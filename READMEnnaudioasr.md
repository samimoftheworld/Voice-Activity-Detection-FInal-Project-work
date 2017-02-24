For training networks for ASR. Contains a `Network` class that's an OO implementation of tensorflow. Multistream capable. Example program below. Example to format data also below.

# Network

This class uses Tensorflow to create a Neural Network. Trained networks can be saved and restored. Network can be split into different streams (example below).
Assumes training data is in hdf5 file for efficient memory usage.

(For kaldi) Can, given a file of utterance IDs with their feature vectors, output result into a file that kaldi can
	process (specifically `latgen-faster-mapped` from nnet1). 

Inside python use `help(Network)` to bring up the help text showing available functions. Summary of implementation:

Fixed:

	Elu activation function.
	Adam SGD.
	Weight init using `tf.truncated_normal(stddev=1.41/input)` (bias=0). (works better than He et al 2015 init technique when dropout is used) 
	Softmax output.
	

Variable:

	Dropout.
	L2 regularization (not yet available for multi-stream).
	Different learning rate policies (see train function).

A lot of smaller features (as in useful functions etc.) are also there.

## Example

A more practical example (with multistream) can be viewed at the bottom of this page. Simple example:

	from NN import Network 

	
	data_train = '....hdf5'
	data_val = '....hdf5'

	NN = Network([480, 1024, 1024, 1024, 1024, 2000], fraction_of_gpu=0.5)

	# Training cost, score and validation score output every epoch.
	NN.train(data_train, 20, 1024, 1e-4, val_file=data_val, lam=0.1, kp_prob=0.75)

	model_save_file = '...'
	NN.save(model_save_file)

	NN.stop()

More complicated example involving (discriminative) pretraining:

	from NN import Network
	
	data_train = '....hdf5'
	data_val = '....hdf5'
	pretrain_param_save_file = '...'
	pretrain_params_dict = {'data_train': data_train, 'data_val': data_val,		
				'epochs': 20,  		  'batch_size': 1024,  
				'eta': 1e-4,		  'kp_prob': 0.75, 
				'lam': 1, 	'save_file': pretrain_param_save_file}

	NN = Network([520, 1024, 1024, 1024, 1024, 1024, 1024, 2000], pretrain=True,
					pretrain_params_dict=pretrain_params_dict, fraction_of_gpu=0.5)

	# Training cost, score and validation score output every epoch.
	NN.train(data_train, 20, 1024, 1e-4, val_file=data_val, lam=0.1, kp_prob=0.8)

	model_save_file = '...'
	NN.save(model_save_file)

	NN.stop()

## pretrain_network

Tried to understand RBMs, didn't, don't like using stuff I don't understand, so I thought maybe I could try pretrain a network using other methods, such as used in [1](http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf) or [2](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf)

Results so far have been disappointing. Using a kaldi network of the same size and DBN pretraining achieves an 11-12% WER, I could not get past 15% using the same splicing. I believe the underlying problem is I've got 15 hours of training data (aurora4), the linked papers have an order of magnitude more; per [3](http://research.google.com/pubs/pub38131.html) the more data you have the less pretraining matters (with generative being better than discriminative for small amounts of data). Meaning it's hard for me to replicate the papers achieving better than DBN-level results with discriminative pretraining.

## hdf5 data format

Training only works with data that is formatted correctly. The advantage is low RAM usage.
It's not hard to do, but to make it as simple as possible here's an example. It is assumed that the features and targets are numpy arrays in `f` and `t`.

	import h5py

	# This is where we want out data to end up in.
	data_fname = '...' 

	new_data_file = h5py.File(data_fname, 'w')
	
	# Note data must use these keys/names.
	new_data_file.create_dataset('feats', data=f)
	new_data_file.create_dataset('targs', data=t)

	new_data_file.close()

	# That's it !
	

## "Real life" example

The arguments I give the `save` function are appended to a "trained\_networks.txt" file which gives me a nice overview of past results. These arguments are optional.


    from NN import Network
    import h5py
    import argparse
    import numpy as np
    import os

    def run_net(epochs, bs, eta, kp_prob, name, notes, load, gpu):

        lam=1
        data_type = 'data-fbank/'
        data_train_f = data_type+'fbank_train_fixed.hdf5'
        val_file = data_type+'fbank_val.hdf5'

        shape = [680, 1024,1024,1024,1024,1024,1024,2023]

        randkey = np.random.randint(10000)

        pretrain_param_dict = {'data_train': data_train_f, 'epochs': 1, 'batch_size': 4096,
                    'eta': 1e-4, 'data_val': val_file, 'kp_prob': kp_prob,
                    'lam': lam, 'save_file': 'params_fbank'+str(randkey)}
    #    lam = 0

        if load != '':
            NN = Network(shape, fraction_of_gpu=1, gpu=gpu)
            NN.load("trained_networks/"+load+".ckpt")
        else:
            NN = Network(shape, pretrain=False, pretrain_params_dict=pretrain_param_dict, fraction_of_gpu=1, gpu=gpu)

        td = h5py.File(data_train_f, 'r')
        vd = h5py.File(val_file, 'r')

        NN.train(data_train_f, epochs, 4096, eta, kp_prob=kp_prob, lam=lam, val_file=val_file, eta_policy='const')
        NN.train(data_train_f, epochs, bs, eta, kp_prob, val_file=val_file, eta_policy='const')
        NN.train(data_train_f, epochs, bs, eta, kp_prob, val_file=val_file, eta_policy='restart')

        score =  NN.score(td['feats'][:50000], td['targs'][:50000])
        val_score = 0
        val_ct = 4
        for j in range(val_ct):
            start = j*50000
            end = start + 50000
            val_score += NN.score(vd['feats'][start:end], vd['targs'][start:end])
        val_score /= val_ct
        print(score, val_score)
        score2 = 'NaN'
        td.close()
        vd.close()

        NN.save(name, kp_prob, lam, score, val_score, score2, epochs, load, notes)
        NN.stop()

    def main():
        parser = argparse.ArgumentParser(description='Run DNN.')
        parser.add_argument('epochs', type=int, help="Epochs to run.")
        parser.add_argument('batch_size', type=int, help="Batch size to use.")
        parser.add_argument('eta', type=float, help="Learning rate.")
        parser.add_argument('kp_prob', type=float, help="keep probability (1-drop).")
        parser.add_argument('name', type=str, help="Model name.")
        parser.add_argument('gpu', type=str, help="GPU to use.")
        parser.add_argument('--notes', type=str, default='', help="Additional Notes.")
        parser.add_argument('--load', type=str, default='', help="Model to load.")
        args = parser.parse_args()
        run_net(args.epochs, args.batch_size, args.eta, args.kp_prob, args.name, args.notes, args.load, args.gpu)

    main()	
