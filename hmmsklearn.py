#hete test and train files contain mfcc from the wav files 
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from matplotlib.colors import LogNorm
from sklearn import mixture
for i in range (1,91):
	with open('/home/samim/train%d.txt'%i,'r') as file:
		train = [[float(digit) for digit in line.split()] for line in file]
		train=np.array(train)
		if (i==1 ):
			X1=train
		else:	
			X1 = np.concatenate([X1, train])
#print X1
for i in range (1,68):
	with open('/home/samim/train1%d.txt'%i,'r') as file:
		train = [[float(digit) for digit in line.split()] for line in file]
		train=np.array(train)
		if (i==1 ):
			X2=train
		else:
			X2 = np.concatenate([X2, train])
#print X2
X = np.concatenate([X1, X2])
lengths = [len(X1), len(X2)]
print X
print lengths
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
X=model.fit(X,lengths)
print X
#Y, Z=model.sample(500) #sample function generates random samples from this 
#model.predict(dataset) #this will return a label fo the data given
#print Y
#print Z
#means = np.array([[0.0,  0.0],[0.0, 11.0],[9.0, 10.0],[11.0, -1.0]])
#plt.plot(Y[:, 0], Y[:, 1], ".-", label="observations", ms=6,mfc="orange", alpha=0.7)
#for i, m in enumerate(means):
#    plt.text(m[0], m[1], 'Component %i' % (i + 1),size=17, horizontalalignment='center',bbox=dict(alpha=.7, facecolor='w'))
#plt.legend(loc='best')
#plt.show()
for i in range (1,7):
	with open('/home/samim/test%d.txt'%i,'r') as file:
		test = [[float(digit) for digit in line.split()] for line in file]
	test=np.array(test)
	X=model.predict(test)
	count1=0
	count0=0
	for j in X:
		if(j==1):
			count1=count1+1
		else:
			count0=count0+1
	print count1,count0
	if(count1>=count0):
		print 'test %d is of label 1'%i
	else:
		print 'test %d is of label 0'%i
