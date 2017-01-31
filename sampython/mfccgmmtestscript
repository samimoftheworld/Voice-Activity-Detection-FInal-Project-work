import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
def fit_samples(samples):
	gmix = mixture.GMM(n_components=2, covariance_type='full')
	gmix.fit(samples)
	print gmix.means_
	colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
	ax = plt.gca()
	ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
	plt.show()
	
	
#!/usr/bin/env python
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
#note scipy numpy anf python_speech_features must be installed using pip
#wav.read reads the wav file 
#to change the mfcc samplerate and other stuff do
#python def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
#where the columns are
#Parameter		Description
#signal			the audio signal from which to compute features. Should be an N*1 array
#samplerate		the samplerate of the signal we are working with.
#winlen			the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
#winstep			the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
#numcep			the number of cepstrum to return, default 13
#nfilt			the number of filters in the filterbank, default 26.
#nfft			the FFT size. Default is 512
#lowfreq			lowest band edge of mel filters. In Hz, default is 0
#highfreq		highest band edge of mel filters. In Hz, default is samplerate/2
#preemph			apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97
#ceplifter		apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22
#appendEnergy	if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
#returns			A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.


for i in xrange(1,101):
	(rate,sig) = wav.read("/home/sam/vad/n%d.wav"%i)
	#to generate the mfcc
	mfcc_feat = mfcc(sig,rate)
	fit_samples(mfcc_feat)
	np.savetxt('/home/sam/test%d.txt'%i,mfcc_feat)
	
for i in xrange(1,101):
	f = open ( '/home/sam/test%d.txt'%i , 'r')
	l = [ map(int,line.split(' ')) for line in f ]
	fit_samples(l)
	
	
	
	
#script to plot multiple graphs	
	
def q(samples):
	g1 = mlab.bivariate_normal(samples, 1.0, 1.0, -1, -1, -0.8)
	g2 = mlab.bivariate_normal(samples, 1.5, 0.8, 1, 2, 0.6)
	return 0.6*g1+28.4*g2/(0.6+28.4)

def plot_q(samples):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	samples = np.meshgrid(samples)
	Z = q(samples)
	surf = ax.plot_surface(samples, Z, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'),linewidth=0, antialiased=True)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.savefig('3dgauss.png')
	plt.show()

def sample():
	'''Metropolis Hastings'''
	N = 10000
	s = 10
	r = np.zeros(2)
	p = q(r[0], r[1])
	print p
	samples = []
	for i in xrange(N):
		rn = r + np.random.normal(size=2)
		pn = q(rn[0], rn[1])
		if pn >= p:
			p = pn
			r = rn
		else:
			u = np.random.rand()
			if u < pn/p:
				p = pn
				r = rn
		if i % s == 0:
			samples.append(r)
	samples = np.array(samples)
	plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
	'''Plot target'''
	dx = 0.01
	x = np.arange(np.min(samples), np.max(samples), dx)
	y = np.arange(np.min(samples), np.max(samples), dx)
	X, Y = np.meshgrid(x, y)
	Z = q(X, Y)
	CS = plt.contour(X, Y, Z, 10, alpha=0.5)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.savefig("samples.png")
	return samples

def fit_samples(samples):
	gmix = mixture.GMM(n_components=2, covariance_type='full')
	gmix.fit(samples)
	print gmix.means_
	colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
	ax = plt.gca()
	ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
	plt.savefig("class.png")
	plt.show()

