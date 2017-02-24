
# coding: utf-8

# In[1]:

import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell
import numpy as np
from python_speech_features import mfcc
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# In[2]:

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 13, frames = 25):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            #print fn
            sound_clip,s = librosa.load(fn)
            #print fn
            #print fn.split('/')[5].split('-')[0]
            label = fn.split('/')[5].split('-')[0]
            if(label=='n'):
                label=0
            if(label=='sp'):
                label=1
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    for ho in range(0,frames):
                        labels.append(label)         
    features = np.asarray(mfccs).reshape(len(mfccs)*frames,bands)
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[4]:



parent_dir = '/home/samim/audiotrainingset/'

tr_sub_dirs = ['/home/samim/audiotrainingset/training/']
tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
tr_labels = one_hot_encode(tr_labels)

ts_sub_dirs = ['/home/samim/audiotrainingset/testing/']
ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
ts_labels = one_hot_encode(ts_labels)



# In[5]:

print np.shape(tr_features)
print np.shape(ts_features)
print np.shape(tr_labels)
print np.shape(ts_labels)
#print tr_features
#x=np.array(tr_features)
#with open('/home/samim/test.txt','w') as f:
#    np.savetxt(f,x,fmt='%.18e')
#with open('/home/samim/test1.txt','w') as f:
#    np.savetxt(f,ts_features)
#with open('/home/samim/test2.txt','w') as f:
#    np.savetxt(f,tr_labels)
#with open('/home/samim/test3.txt','w') as f:
#    np.savetxt(f,ts_labels)

#np.savetxt('/home/samim/test.txt',tr_features)
#np.savetxt('/home/samim/test1.txt',ts_features)
#np.savetxt('/home/samim/test2.txt',tr_labels)
#np.savetxt('/home/samim/test3.txt',ts_labels)


# In[6]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[23]:

mfcc_size=13
num_labels=2
batch_size = 128
hidden_size = 1024

graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, mfcc_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	#tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(ts_features)
	
	# Variables.
	W1 = tf.Variable(tf.truncated_normal([mfcc_size, hidden_size]))
	b1 = tf.Variable(tf.zeros([hidden_size]))

	W2 = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
	b2 = tf.Variable(tf.zeros([num_labels]))	
	#W1=tf.cast(W1, tf.float64)
	#b1=tf.cast(b1, tf.float64)
	#W2=tf.cast(W2, tf.float64)
	#b2=tf.cast(b2, tf.float64)
	# Training computation.
	y1 = tf.nn.relu( tf.matmul(tf_train_dataset, W1) + b1 )
	logits = tf.matmul(y1, W2) + b2
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
	
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)

	#y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
	#valid_logits = tf.matmul(y1_valid, W2) + b2
	#valid_prediction = tf.nn.softmax(valid_logits)
	w1=tf.cast(W1, tf.float64)
	B1=tf.cast(b1, tf.float64)
	w2=tf.cast(W2, tf.float64)
	B2=tf.cast(b2, tf.float64)
	y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, w1) + B1)
	test_logits = tf.matmul(y1_test, w2) + B2
	test_prediction = tf.nn.softmax(test_logits)


# Let's run it:
num_steps = 3001

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):
		# Pick an offset within the training data, which has been randomized.
		# Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (tr_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = tr_features[offset:(offset + batch_size), :]
		batch_labels = tr_labels[offset:(offset + batch_size), :]
		#print batch_labels
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print np.shape(batch_data)
			print np.shape(predictions)           
			print np.shape(batch_labels)
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			#print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
			print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), ts_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), ts_labels))

