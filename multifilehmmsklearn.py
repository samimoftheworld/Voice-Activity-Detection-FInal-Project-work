
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from matplotlib.colors import LogNorm
from sklearn import mixture
model = hmm.GMMHMM(n_components=2,algorithm="map",n_iter=1000,covariance_type="full")
for i in range (1,100):
	with open('/home/samim/test%d.txt'%i,'r') as file:
		train = [[float(digit) for digit in line.split()] for line in file]
	train=np.array(train)
	X=model.fit(train)
#print X
#Y, Z=model.sample(500) #sample function generates random samples from this 
#model.predict(dataset) #this will return a label fo the data given
with open('/home/samim/test.txt','r') as file:
	test = [[float(digit) for digit in line.split()] for line in file]
test=np.array(train)
#plt.plot(Y[:, 0], Y[:, 1], ".-", label="observations", ms=6,mfc="orange", alpha=0.7)
#for i, m in enumerate(means):
#    plt.text(m[0], m[1], 'Component %i' % (i + 1),size=17, horizontalalignment='center',bbox=dict(alpha=.7, facecolor='w'))
#plt.legend(loc='best')
#plt.show()
