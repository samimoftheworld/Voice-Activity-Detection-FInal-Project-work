import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
with open('/home/samim/test.txt','r') as file:
	train = [[float(digit) for digit in line.split()] for line in file]
train=np.array(train)
clf = mixture.GaussianMixture(n_components=13, covariance_type='full')
clf.fit(train)
print clf.get_params()
"""
Parameters:	

n_components : int, defaults to 1.

    The number of mixture components.

covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’},
   note:- covariance is the joint of 2 matrices id=e the E ie the sigma 
        defaults to ‘full’.

    String describing the type of covariance parameters to use. Must be one of:

    'full' (each component has its own general covariance matrix),
    'tied' (all components share the same general covariance matrix),
    'diag' (each component has its own diagonal covariance matrix),
    'spherical' (each component has its own single variance).

tol : float, defaults to 1e-3.

    The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

reg_covar : float, defaults to 0.

    Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.

max_iter : int, defaults to 100.

    The number of EM iterations to perform.

n_init : int, defaults to 1.

    The number of initializations to perform. The best results are kept.

init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.

    The method used to initialize the weights, the means and the precisions. Must be one of:

    'kmeans' : responsibilities are initialized using kmeans.
    'random' : responsibilities are initialized randomly.

weights_init : array-like, shape (n_components, ), optional

    The user-provided initial weights, defaults to None. If it None, weights are initialized using the init_params method.

means_init: array-like, shape (n_components, n_features), optional :

    The user-provided initial means, defaults to None, If it None, means are initialized using the init_params method.

precisions_init: array-like, optional. :

    The user-provided initial precisions (inverse of the covariance matrices), defaults to None. If it None, precisions are initialized using the ‘init_params’ method. The shape depends on ‘covariance_type’:

    (n_components,)                        if 'spherical',
    (n_features, n_features)               if 'tied',
    (n_components, n_features)             if 'diag',
    (n_components, n_features, n_features) if 'full'

random_state : RandomState or an int seed, defaults to None.

    A random number generator instance.

warm_start : bool, default to False.

    If ‘warm_start’ is True, the solution of the last fitting is used as initialization for the next call of fit(). This can speed up convergence when fit is called several time on similar problems.

verbose : int, default to 0.

    Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater than 1 then it prints also the log probability and the time needed for each step.

verbose_interval : int, default to 10.

    Number of iteration done before the next print.


"""
for i in range (1,100):
	with open('/home/samim/test%d.txt'%i,'r') as file:
		array2d = [[float(digit) for digit in line.split()] for line in file]
	array2d=np.array(array2d)
	lab=clf.predict(array2d)
	#print 'the label %d is'%i,lab
# display predicted scores by the model as a contour plot
#x = np.linspace(-20., 30.,num=13) #generates a array 
#y = np.linspace(-20., 40.,num=13)
#X, Y = np.meshgrid(x, y)
#print y
#print np.reshape(x.T,(13,1))
#XX = np.dot(np.reshape(x,(13,1)), np.reshape(y,(1,13)))
#print XX
#Z = -clf.score_samples(XX)
#print Z
#Z = Z.reshape(x.shape)
#print Z
#CS = plt.contour(x, y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
 #                levels=np.logspace(0, 3, 10))
#CB = plt.colorbar(CS, shrink=0.8, extend='both')
#plt.scatter(train[:, 0], train[:, 1], .8)

#plt.title('Negative log-likelihood predicted by a GMM')
#plt.axis('tight')
#plt.show()

