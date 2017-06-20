import tensorflow as tf
import numpy as np
from datetime import datetime
import h5py
import os


#    Uni Oldenburg. Rudolf Braun 2016. rab014@gmail.com        
#    For newest version: github.com/Nimitz14/ASR_Network


def ini_weight_var(shape, name=None):
    initial=tf.truncated_normal(shape, stddev=1.41/shape[0])    
    return tf.Variable(initial, name=name, trainable=True)


def ini_bias_var(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name, trainable=True)


def batch_gen(f, t, bs, len_feat, div):
    '''
    Generator function (use in for-loop) for returning `bs` sized training batches until all data is used (one epoch).
    Creates batch from `div` number of sub-batches taken from random portions of the training data (alternative to shuffling).
    `f` and `t` are links to hdf5 files (features and targets respectively).
    '''

    div_bs = int(bs/div)    
    num_iters = int(len(f) / div_bs)
    batch_idx = np.random.permutation(num_iters)*div_bs
    idx_start, idx_end = [], []
    for j in range(div):
        idx_start.append(div_bs*j)
        idx_end.append(div_bs*(j+1))

    for i in range(0, len(batch_idx)-div, div):
        bf, bt = np.zeros((div*div_bs, len_feat), dtype=np.float32), np.zeros((div*div_bs,), dtype=np.int64)
        for j in range(div):
            start = batch_idx[i+j]
            end = start+div_bs
            
            bf[idx_start[j]: idx_end[j]], bt[idx_start[j]: idx_end[j]] = f[start:end], t[start:end]

        yield bf, bt


class Network:

    def init_layers(self, split, splitting, input_split):
        self.weights = []
        self.biases = []
        self.a = []
        self.a_drop = []

        if not split:
            for i, len_layer in enumerate(self.shape[1:-1]):
                self.weights.append(ini_weight_var([self.shape[i], len_layer], 'w'+str(i)))
                self.biases.append(ini_bias_var([len_layer], 'b'+str(i)))
                if i==0:
                    self.a.append(tf.nn.elu(tf.matmul(self.input_layer, self.weights[i]) + self.biases[i]))
                    self.a_drop.append(tf.nn.dropout(self.a[0], self.kp_prob))
                else:
                    self.a.append(tf.nn.elu(tf.matmul(self.a_drop[i-1], self.weights[i]) + self.biases[i]))
                    self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))

            # Output.
            self.weights.append(ini_weight_var([self.shape[-2], self.shape[-1]], 'wl'))
            self.biases.append(ini_bias_var([self.shape[-1]], 'bl'))

            self.output = tf.matmul(self.a_drop[-1], self.weights[-1]) + self.biases[-1]
            self.softm_output = tf.nn.softmax(self.output)

        else:
            print("!\tMulti-stream dropout is set to {}\t!".format(self.split_kp_prob_const))
            switch, self.switch_ct = True, 0
            if input_split == None:
                self.inp_layers = tf.split(1, splitting[0], self.input_layer)
            else:
                self.inp_layers = []
                start = 0
                for _slice in input_split:
                    self.inp_layers.append(tf.slice(self.input_layer, [0, start], [-1, _slice]))
                    start += _slice
            for i, len_layer in enumerate(self.shape[1:-1]):
                div = splitting[i]
                if div != 1:
                    len_piece1 = int(self.shape[i]/div)
                    len_piece2 = int(len_layer/div)
                    self.weights.append([])
                    self.biases.append([])
                    self.a.append([])
                    self.a_drop.append([])
                    for j in range(div):
                        if i == 0:
                            self.weights[i].append(ini_weight_var([input_split[j], len_piece2]))
                        else:
                            self.weights[i].append(ini_weight_var([len_piece1, len_piece2]))
                        self.biases[i].append(ini_weight_var([len_piece2]))
                        if i==0:
                            self.a[i].append(tf.nn.elu(tf.matmul(self.inp_layers[j], self.weights[i][j]) + self.biases[i][j]))
                            self.a_drop[i].append(tf.nn.dropout(self.a[i][j], self.split_kp_prob))
                        else:
                            self.a[i].append(tf.nn.elu(tf.matmul(self.a_drop[i-1][j], self.weights[i][j]) + self.biases[i][j]))
                            self.a_drop[i].append(tf.nn.dropout(self.a[i][j], self.split_kp_prob))
                else:
                    self.weights.append(ini_weight_var([self.shape[i], len_layer]))
                    self.biases.append(ini_bias_var([len_layer]))
                    if switch:
                        self.switch_ct = i
                        switch = False
                        if i == 0:
                            self.joined = tf.concat(1, self.inp_layers)
                        else:
                            self.joined = tf.concat(1, self.a_drop[i-1])
                        self.a.append(tf.nn.elu(tf.matmul(self.joined, self.weights[i]) + self.biases[i]))
                        self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))
                    else:
                        self.a.append(tf.nn.elu(tf.matmul(self.a_drop[i-1], self.weights[i]) + self.biases[i]))
                        self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))

            # Output
            self.weights.append(ini_weight_var([self.shape[-2], self.shape[-1]]))
            self.biases.append(ini_bias_var([self.shape[-1]]))
            self.output = tf.matmul(self.a_drop[-1], self.weights[-1]) + self.biases[-1]
            self.softm_output = tf.nn.softmax(self.output)

    def __init__(self, shape, pretrain=False, split=False, pretrain_params_dict=None, fraction_of_gpu=1, 
                 restore_from_ptparam=False, splitting=None, input_split=None, gpu=None):

        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if pretrain and not restore_from_ptparam:
            with tf.Graph().as_default():
                pretrain_network(shape, splitting, pretrain_params_dict['data_train'], pretrain_params_dict['epochs'],
                                 pretrain_params_dict['batch_size'], pretrain_params_dict['eta'],
                                 pretrain_params_dict['data_val'], pretrain_params_dict['kp_prob'],
                                 pretrain_params_dict['lam'], pretrain_params_dict['save_file'],
                                 fraction_of_gpu=fraction_of_gpu, input_split=input_split)

        self.shape = np.asarray(shape)
        self.split = split

        # Input on 'train time'.
        self.eta = tf.placeholder("float")
        self.input_layer = tf.placeholder("float",shape=[None, self.shape[0]])
        self.kp_prob = tf.placeholder("float")
        self.split_kp_prob = tf.placeholder("float")
        self.split_kp_prob_const = 0.5
        self.lam = tf.placeholder("float")
        self.targets = tf.placeholder("int64", shape=[None,])

        self.init_layers(split, splitting, input_split)

        # Regularization.
        if not self.split:
            num_param = np.sum([f*s+s for f, s in zip(self.shape[:-1], self.shape[1:])])
            self.L2_reg = (tf.add_n([tf.nn.l2_loss(mat) for mat in self.weights]) + tf.add_n([tf.nn.l2_loss(vec) for vec in self.biases]))/num_param
        else:
            self.L2_reg = 0

        # Cross entropy function (1: Individual cross entropy error, 2: Total cross entropy with reg.).
        self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.targets))
        self.cost_m = self.error + self.lam*self.L2_reg

        # Adam SGD.
        self.train_adam = tf.train.AdamOptimizer(self.eta, epsilon=1e-15).minimize(self.cost_m)

        self.get_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softm_output, 1), self.targets),"float"))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction_of_gpu)
   
        params_dict = {}
        if self.split:
            for i in range(len(self.weights)):
                if i == len(self.weights) - 1:    
                    params_dict['wl'] = self.weights[-1]
                    params_dict['bl'] = self.biases[-1]
                else:

                    if i < self.switch_ct:
                        for j in range(splitting[0]):
                            params_dict['w' + str(i) + str(j)] = self.weights[i][j]
                            params_dict['b' + str(i) + str(j)] = self.biases[i][j]
                    else:
                        params_dict['w' + str(i)] = self.weights[i]
                        params_dict['b' + str(i)] = self.biases[i]
        else:
            for i in range(len(self.weights)):
                if i == len(self.weights) - 1:
                    params_dict['wl'] = self.weights[-1]
                    params_dict['bl'] = self.biases[-1]
                else:
                    params_dict['w' + str(i)] = self.weights[i]
                    params_dict['b' + str(i)] = self.biases[i]
        
        self.saver = tf.train.Saver(params_dict)

        if pretrain:
            print("Restoring.")

            get_pt = tf.train.import_meta_graph(pretrain_params_dict['save_file'] + '.meta')

            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            get_pt.restore(self.sess, tf.train.latest_checkpoint('./'))

            in_v = [v for v in tf.global_variables() if v not in (self.weights or self.biases)]
            self.sess.run(tf.variables_initializer(in_v))

        else:
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.global_variables_initializer())

        print("Session running.")
    
    def train(self, data_file, epochs, batch_size, eta, kp_prob=1, eta_policy='const', lam=0, val_file=None,
              eta_chk_pt=1, partial_scoring=True, output_scores=True):
        '''
        Trains network. Minimizes the mean of cross-entropy for softmax outputs.
        Args:
            data_file:      hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
            epochs:         Number of iterations that SGD should perform.
            batch_size:     Number of (feat,targ) pairs to use per SGD iteration.
            eta:            Learning rate.

            kp_prob:        Proportion of neurons that should not be dropped (kept).
                default - 1

            eta_policy:     'const' -> constant learning rate
                            'ES' -> adaptive learning rate (/2 if cost function not decreased after `eta_chk_pt` epochs).
                                Stops after learning rate has been reduced 3 times.
                            'restart' -> Learning rate is reduced (/2) after `epochs`/3. Entire process is repeated 3 times,
                                meaning total number of epochs trained is 3x `epochs`.
                default - 'const'

            lam:            Regularization multiplier.
                default - 0

            val_file:       hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
                default - 'None'

            eta_chk_pt:        Number of epochs at which to check cost to decrease or keep constant the learning rate (eta_policy=="ES").
                default - 1

            partial_scoring: Takes 50 000 samples from the training set and uses them for scoring/error evaluation. If
                            set to `False`, the entire set will be used.
                default - True

            output_scores:   Output error and accuracy values.
                default - True

        '''

        train_opt = self.train_adam

        df = h5py.File(data_file, 'r')
        feats, targs = df['feats'], df['targs']
        
        len_feat = len(feats[0])
        scoring_batch_size = 50000
        print("!\tScoring batch size set to {}.".format(scoring_batch_size))
        
        if partial_scoring:
            sc_f, sc_t = feats[:scoring_batch_size], targs[:scoring_batch_size]    
        else:
            sc_f, sc_t = feats, targs
            
        if not val_file:
            val_data = False
        else:
            val_data = True
            val_d = h5py.File(val_file, 'r')
            val_feats, val_targs = val_d['feats'], val_d['targs']
            val_ct = int(np.ceil(len(val_targs) / scoring_batch_size))
            
        print("Beginning training.")

        if eta_policy == 'ES':
            last_cost = 1e20
            epochs = 100
            original_eta = eta

        if eta_policy == 'restart':
            epochs *= 3

        div = 2
        scores = []
    
        tt = datetime.now()
        for epoch in range(epochs):

            for batch_f, batch_t in batch_gen(feats, targs, batch_size, len_feat, div):
            
                self.sess.run(train_opt, feed_dict={self.input_layer: batch_f, self.targets: batch_t,
                                                    self.eta: eta, self.kp_prob: kp_prob, self.lam: lam, self.split_kp_prob: self.split_kp_prob_const})

            if eta_policy == 'const':
                pass
            elif eta_policy == 'restart':
                if (epoch+1) % int(epochs/9.0) == 0:
                    eta /= 2.0
            elif eta_policy == 'ES':
                if (epoch+1) % eta_chk_pt == 0:
                    new_cost = self.cost_mean(feats, targs)
                if new_cost > last_cost:
                    eta /= 2.0
                    last_cost = new_cost
                if eta == original_eta / 8:
                    break
            else:
                print("No output: `eta_policy` must equal 'const', 'ES' or 'restart' ('const' is default).")

            if output_scores:
                train_score = self.score(sc_f, sc_t)
                if val_data:
                    val_score = 0
                    for i in range(val_ct):
                        start = i*50000
                        end = start + 50000
                        val_score += self.score(val_feats[start:end], val_targs[start:end])
                    val_score /= float(val_ct)

                    print("Training error {}, Training score {}, Val score {}".format(self.cost_mean(sc_f, sc_t),
                                                                            train_score,
                                                                            val_score))

                else:
                    print("Training error {0}, training score {1}".format(self.cost_mean(sc_f, sc_t), train_score))


                if epoch == 0:
                    scores.append(train_score)
                    if val_data:
                        scores.append(val_score)

        df.close()
        if val_data:
            val_d.close()

        print("Training duration: {0}".format(datetime.now() - tt))

        if val_data:
            return scores[0], scores[1]
        else:
            return scores[0]


    def cost_mean(self, feats, targs):
        '''
        Input: Feature vectors and their targets. Returns mean of the cost function outputs
        '''
        return self.sess.run(self.error, feed_dict = {self.input_layer:feats,
                                                      self.targets:targs, 
                                                      self.kp_prob:1,self.lam:0, 
                                                      self.split_kp_prob: 1})
    
    def score(self, feats, targs):
        '''
        Input: Feature vectors and their targets. Returns proportion of outputs that fit targets.
        '''
        return self.sess.run(self.get_acc, feed_dict = {self.input_layer:feats,
                                                        self.targets:targs,
                                                        self.kp_prob:1,self.lam:0, 
                                                        self.split_kp_prob: 1})
    
    def forward_pass(self, inp, apply_log=False):
        '''
        Input: Feature vectors. Returns outputs.
        '''
        if apply_log is True:
            return np.log(self.sess.run(self.softm_output,feed_dict = {self.input_layer:inp, 
                                                                       self.kp_prob:1, 
                                                                       self.split_kp_prob: 1})+1e-15)
        else:
            print("Outputting pre softmax")
            return self.sess.run(self.output,feed_dict = {self.input_layer:inp, 
                                                          self.kp_prob:1, 
                                                          self.split_kp_prob: 1})
    
    def feats_extractor(self, feats_file):
        key_, mat = '', []
        started = False
        with open(feats_file) as f:
            for line in f:
                if line.endswith('[\n'):
                    if started:
                        yield key_, np.asarray(mat)
                        mat = []
                    key_ = line[:-2].strip()
                    started = True
                else:
                    mat.append([float(s) for s in line.strip().strip(']').split()])
    
    
    def output_for_kaldi(self, feats_file, f_name='network_output.txt', len_feat=40, splicing=0, apply_log=True):
        '''
        Takes a txt file of the form
    
            <utterance-ID> [
            <feature vector>
            ...
            <feature vector> ]
            <utterance-ID> [
            ...
    
        extracts the utterances and corresponding feature vectors that match a SNR, calculates the outputs, and saves
        the result in a file 'network_output.txt' in the same form as input. Can be used by latgen-faster-mapped (kaldi).
    
        Args:
            feats_file:        File with utterance-IDs and feature vectors.
            f_name:            Name of output file.
                default - 'network_output.txt'
            len_feat:          Feature length (default 40 is because of Fbank).
                default - 40
            splicing:          Plus-minus splicing to perform.
                default - 0
            apply_log:         Apply logarithm to output [or not].
                default - True
    
        '''
    
        print("!\tFeature length set as: {} | Splicing set as {} \t!".format(len_feat, splicing))

        if apply_log:
            print("Applying the logarithm.")
        else:
            print("Not applying the logarithm.")

        with open(f_name, 'w+') as ofile:
            if splicing != 0:
                zeropad = np.zeros((splicing, len_feat))
                for k, v in self.feats_extractor(feats_file):

                    f_mat = self.splice_frames(v, len_feat, splicing, zeropad)

                    f_mat -= np.mean(f_mat, axis=0)
                    f_mat /= np.std(f_mat, axis=0)

                    ofile.write('{0}  [\n'.format(k))
                    outs = self.forward_pass(f_mat, apply_log=apply_log)
                    for out in outs[:-1]:
                        ofile.write('{0}\n'.format(' '.join(str(c) for c in out)))
                    ofile.write('{0}  ]\n'.format(' '.join(str(c) for c in outs[-1])))

            else:
                for k, v in self.feats_extractor(feats_file):
                    ofile.write('{0}  [\n'.format(k))
                    f = np.asarray(v)

                    f_mat -= np.mean(f_mat, axis=0)
                    f_mat /= np.std(f_mat, axis=0)

                    outs = self.forward_pass(f, apply_log=apply_log)
                    for out in outs[:-1]:
                        ofile.write('{0}\n'.format(' '.join(str(c) for c in out)))
                    ofile.write('{0}  ]\n'.format(' '.join(str(c) for c in outs[-1])))
        
    def splice_frames(self, mat, len_feat, splicing, zeropad):
        len_utt = len(mat)
    
        padded_mat = np.vstack((zeropad, mat, zeropad))
        spliced_mat = np.zeros((len_utt, len_feat * (2 * splicing + 1)), dtype=np.float32)
    
        for i in range(len_utt):
            if i < len_utt - 1:
                spliced_mat[i] = padded_mat[i:i + 2*splicing + 1].flatten()
            else:
                spliced_mat[i] = padded_mat[i:].flatten()
    
        return spliced_mat
    
    def output_for_kaldi_multi(self, feats_files, lens_feat, splicing, f_name='network_output.txt', apply_log=True):
    
        print("!\tFeature lengths set as: {}, {}, {}. \t Splicing set as {}, {}, {} \t!".format(
                lens_feat[0], lens_feat[1], lens_feat[2], splicing[0], splicing[1], splicing[2]))
    
        print("!\tDon't make the mistake of having trained with feature vectors that you normalized twice.\t!")
    
        if apply_log:
            print("Applying the logarithm.")
        else:
            print("Not applying the logarithm.")
        
        with open(f_name, 'w+') as ofile:
            zeropad = [np.zeros((splicing[0], lens_feat[0])), np.zeros((splicing[1], lens_feat[1])),
                        np.zeros((splicing[2], lens_feat[2]))]
    
            print(feats_files[0], feats_files[1], feats_files[2])
    
            for (k1, v1), (k2, v2), (k3, v3) in zip(self.feats_extractor(feats_files[0]), 
                                                    self.feats_extractor(feats_files[1]),
                                                    self.feats_extractor(feats_files[2])):
                
                if k1 != k2 or k2 != k3:
                    print("Something is wrong, utterance IDs: {}, {}, {}.".format(k1, k2, k3))
                    print("Skipping.")
                    continue
    
                if splicing[0] != 0:
                    v1 = self.splice_frames(v1, lens_feat[0], splicing[0], zeropad[0])
                if splicing[1] != 0:            
                    v2 = self.splice_frames(v2, lens_feat[1], splicing[1], zeropad[1])
                if splicing[2] != 0:
                    v3 = self.splice_frames(v3, lens_feat[2], splicing[2], zeropad[2])
    
                f_mat = np.hstack((v1, v2, v3))
                
                f_mat -= np.mean(f_mat, axis=0)
                f_mat /= np.std(f_mat, axis=0)
    
                ofile.write('{}  [\n'.format(k1))
                outs = self.forward_pass(f_mat, apply_log=apply_log)
                for out in outs[:-1]:
                    ofile.write('{}\n'.format(' '.join(str(c) for c in out)))
                ofile.write('{}  ]\n'.format(' '.join(str(c) for c in outs[-1])))
    
    def save(self, name='NN_model', kp_prob=1, lam=0, score=None, score1=None, score2=None, Epochs='N/A', load='None', notes='None', start_sc=None, start_sc2=None):
        '''
        Saves the model (weights (vector of matrices) and bias (matrix)) to 'NN_model.ckpt' or input string.
        '''
        if score1:
            with open('Network_saves.txt','a') as f:
                f.write("File: {:15s} Start Sc: {!s:8s} {!s:8s} Scores: {!s:8s} {!s:8s} {!s:8s} kp_prob: {!s:5s} Lambda: {!s:5s} Epochs: {!s:6s} Loaded from: {:15s}Notes: {}\n".format(
                        name, start_sc, start_sc2, score, score1, score2, kp_prob, lam, Epochs, load, notes))
    
        save_path = self.saver.save(self.sess, "trained_networks/"+name+".ckpt")
        print("Model saved as: %s"%save_path)
    
    
    def load(self, n):
        '''
        Input: File where model is saved. Note network to be loaded must have the same shape as the saved network.
        '''
        self.saver.restore(self.sess,n)
        print("Model loaded.")
    
    
    def stop(self):
        self.sess.close()
        print("Session closed.")
    
    
def pretrain_network(shape, splitting, data_file, epochs, batch_size, eta, val_file='None', kp_prob=1, lam=0,
                 name='params', fraction_of_gpu=1, input_split=None):
    '''
    Trains and saves a Network layer by layer. Training data is partially scored, as well as val data for each epoch.
    Last epoch uses /2 of the selected batch size.

    Based on this: http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf
    Although method not identical (optimal implementation not clear).
    '''

    input_layer = tf.placeholder("float32", shape=[None, shape[0]])
    targets = tf.placeholder("int64", shape=[None,])
    eta_ph = tf.placeholder("float")
    kp_prob_ph = tf.placeholder("float")
    kp_prob_split_ph = tf.placeholder("float")
    kp_prob_split = 0.5

    pt_weights = []
    pt_biases = []
    pt_a = []
    pt_a_drop = []
    sm_out = []
    sm_out_calc = []
    cost = []
    train_opt = []
    acc = []

    df = h5py.File(data_file, 'r')
    feats, targs = df['feats'], df['targs']
    f_t, t_t = feats[:50000], targs[:50000]
    if val_file != 'None':
        vf = h5py.File(val_file, 'r')
        f_sc, t_sc = vf['feats'], vf['targs']
        val_ct = 4 
    len_feat = len(f_t[0])
    sm_weights = []
    sm_bias = []
    num_param = []
    L2_reg = []
    joined = []

    if splitting is None:
        for i, len_layer in enumerate(shape[1:-1]):
            pt_weights.append(ini_weight_var([shape[i], len_layer], 'w' + str(i)))
            pt_biases.append(ini_bias_var([len_layer], 'b' + str(i)))
            if i == 0:
                pt_a.append(tf.nn.elu(tf.matmul(input_layer, pt_weights[i]) + pt_biases[i]))
                pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob_ph))
            else:
                pt_a.append(tf.nn.elu(tf.matmul(pt_a_drop[i - 1], pt_weights[i]) + pt_biases[i]))
                pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob_ph))

            if i == len(shape[1:-1])-1:
                sm_weights.append(ini_weight_var([len_layer, shape[-1]], 'wl'))
                sm_bias.append(ini_bias_var([shape[-1]], 'bl'))
            else:
                sm_weights.append(ini_weight_var([len_layer, shape[-1]]))
                sm_bias.append(ini_bias_var([shape[-1]]))

            sm_out.append(tf.matmul(pt_a_drop[i], sm_weights[i]) + sm_bias[i])  # Tensorflow takes care of softmax
            sm_out_calc.append(tf.nn.softmax(sm_out[i]))
            acc.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sm_out_calc[i], 1), targets), "float")))

            num_param.append(np.sum([f*s+s for f, s in zip(shape[:-1][:i+1], shape[1:i+2])]))
            L2_reg.append(tf.reduce_sum(tf.add_n([tf.nn.l2_loss(mat) for mat in pt_weights[:i+1]])) +
                          tf.reduce_sum(tf.add_n([tf.nn.l2_loss(vec) for vec in pt_biases[:i + 1]])))

            cost.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(sm_out[i], targets)) + lam*L2_reg[i]/num_param[i])
            #cost.append(tf.nn.sparse_softmax_cross_entropy_with_logits(sm_out[i], targets)) 

            train_opt.append(tf.train.AdamOptimizer(eta_ph, epsilon=1e-15).minimize(cost[i]))

    else:
        switch, switch_ct = True, 0

        if input_split == None:
            inp_layers = tf.split(1, splitting[0], input_layer)
        else:
            inp_layers = []
            start = 0
        
            for _slice in input_split:
                inp_layers.append(tf.slice(input_layer, [0, start], [-1, _slice]))
                start += _slice
        for i, len_layer in enumerate(shape[1:-1]):
            div = splitting[i]
            if div != 1:
                len_piece1 = int(shape[i] / div)
                len_piece2 = int(len_layer / div)
                pt_weights.append([])
                pt_biases.append([])
                pt_a.append([])
                pt_a_drop.append([])
                for j in range(div):
                    if i == 0:    
                        pt_weights[i].append(ini_weight_var([input_split[j], len_piece2]))
                    else:
                        pt_weights[i].append(ini_weight_var([len_piece1, len_piece2]))
                    pt_biases[i].append(ini_weight_var([len_piece2]))
                    if i == 0:
                        pt_a[i].append(
                            tf.nn.elu(tf.matmul(inp_layers[j], pt_weights[i][j] + pt_biases[i][j])))
                        pt_a_drop[i].append(tf.nn.dropout(pt_a[i][j], kp_prob_split_ph))
                    else:
                        pt_a[i].append(
                            tf.nn.elu(tf.matmul(pt_a_drop[i - 1][j], pt_weights[i][j]) + pt_biases[i][j]))
                        pt_a_drop[i].append(tf.nn.dropout(pt_a[i][j], kp_prob_split_ph))

                joined.append(tf.concat(1, pt_a_drop[i-1]))
                sm_weights.append(ini_weight_var([len_layer, shape[-1]]))
                sm_bias.append(ini_bias_var([shape[-1]]))
                sm_out.append(tf.matmul(joined[i], sm_weights[i]) + sm_bias[i])
                sm_out_calc.append(tf.nn.softmax(sm_out[i]))

                acc.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sm_out_calc[i], 1), targets), "float")))
                cost.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(sm_out[i], targets)))
                train_opt.append(tf.train.AdamOptimizer(eta_ph, epsilon=1e-15).minimize(cost[i]))

            else:
                pt_weights.append(ini_weight_var([shape[i], len_layer]))
                pt_biases.append(ini_bias_var([len_layer]))
                if switch:
                    switch = False
                    switch_ct = i
                    if i == 0:
                        joined.append(tf.concat(1, inp_layers))
                    else:
                        joined.append(tf.concat(1, pt_a_drop[i - 1]))
                    pt_a.append(tf.nn.elu(tf.matmul(joined[i], pt_weights[i]) + pt_biases[i]))
                    pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob_ph))
                else:
                    pt_a.append(tf.nn.elu(tf.matmul(pt_a_drop[i - 1], pt_weights[i]) + pt_biases[i]))
                    pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob_ph))
                
                # Output
                sm_weights.append(ini_weight_var([len_layer, shape[-1]]))
                sm_bias.append(ini_bias_var([shape[-1]]))
                sm_out.append(tf.matmul(pt_a_drop[i], sm_weights[i]) + sm_bias[i])
                sm_out_calc.append(tf.nn.softmax(sm_out[i]))

                acc.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sm_out_calc[i], 1), targets), "float")))
                cost.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(sm_out[i], targets)))
                train_opt.append(tf.train.AdamOptimizer(eta_ph, epsilon=1e-15, beta1=0.0).minimize(cost[i]))

    l2norm_inv_sc = []
    scale = []
    renorm_op = []
    max_norm = tf.constant(5.0)
    for i in range(len(pt_weights)):    
        l2norm_inv_sc.append(max_norm * tf.rsqrt(tf.reduce_sum(tf.square(pt_weights[i]), reduction_indices=0)))
        scale.append(tf.minimum(l2norm_inv_sc[i], tf.constant(1.0)))
        renorm_op.append((tf.assign(pt_weights[i], tf.mul(pt_weights[i], scale[i])), tf.assign(pt_biases[i], tf.mul(pt_biases[i], scale[i]))))

    param_dict = {}
    if splitting == None:
        switch_ct = 0
        for i in range(len(pt_weights) + 1):
            if i == len(pt_weights):
                param_dict['wl'] = sm_weights[-1]
                param_dict['bl'] = sm_bias[-1]
            else:
                param_dict['w' + str(i)] = pt_weights[i]
                param_dict['b' + str(i)] = pt_biases[i]
    else:
        for i in range(len(pt_weights)+1):
            if i == len(pt_weights):
                param_dict['wl'] = sm_weights[-1]
                param_dict['bl'] = sm_bias[-1]
            else:
                if i < switch_ct:
                    for j in range(splitting[0]):
                        param_dict['w' + str(i)+str(j)] = pt_weights[i][j]
                        param_dict['b' + str(i)+str(j)] = pt_biases[i][j]
                else:
                    param_dict['w'+str(i)] = pt_weights[i]
                    param_dict['b'+str(i)] = pt_biases[i]

    save_op = tf.train.Saver(param_dict)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction_of_gpu)
    div = 2

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        print("Beginning pretraining.")

        for i in range(len(shape[1:-1])):
            eta_var= eta
       
            if i == len(shape[1:-1]) - 1:
                batch_size /= 2

            for epoch in range(epochs):
                for batch_f, batch_t in batch_gen(feats, targs, batch_size, len_feat, div):
                    sess.run(train_opt[i],
                             feed_dict={input_layer: batch_f, 
                                        targets: batch_t, 
                                        eta_ph: eta_var, 
                                        kp_prob_ph: kp_prob, 
                                        kp_prob_split_ph: kp_prob_split})
                
                train_score = sess.run(acc[i], feed_dict={input_layer: f_t, 
                                                          targets: t_t, 
                                                          kp_prob_ph: 1, 
                                                          kp_prob_split_ph: 1})
            
                if val_file != 'None':
                    val_score = 0
                    for j in range(val_ct):
                        start = j*50000
                        end = start + 50000
                        
                        val_score +=sess.run(acc[i], feed_dict={input_layer: f_sc[start:end], 
                                                                targets: t_sc[start:end], 
                                                                kp_prob_ph: 1, 
                                                                kp_prob_split_ph: 1})     
                    
                    val_score/=float(val_ct)
                        
                    print("train {0}, val {1}".format(train_score, val_score))
                else:
                    print("train {0}".format(train_score))
            print()
        save_op.save(sess, name)

    df.close()
    if val_file != 'None':
        vf.close()
