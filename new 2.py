import numpy as np
import math
from numpy.linalg import inv
#python code to read the mfcc and then make gmm
with open('/home/samim/test.txt','r') as file:
	array2d = [[float(digit) for digit in line.split()] for line in file]
print(array2d) #to print ful array
print(array2d[1]) #to print only one array row
print(array2d[1][3]) #to print only one element
print([i[0] for i in array2d]) #to print only one array column
nog=13 #no of gausians is 13 as we are having a 13 dimentional martix
w= np.random.dirichlet(np.ones(13),size=1) #this is the weignts of the gausians and the function generates random numbers with total sum = 1
numrows = len(array2d)    # length of rows in your example
numcols = len(array2d[0]) # length of columns in your example
mean=np.random.uniform(1,10,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
sig=np.random.uniform(1,10,size=(1,13)) #this is variance sigma ie variance
siginv=np.reshape(sig, (13,1)) #to get a inverse martix of sigma
pi=math.pow(2*3.14, 13/2) #the value of (2*pi)^(nog/2)
sigma=np.multiply(sig, siginv) #this will give me a sigma matrix a multiplication of the variances
x=array2d[0]
xu=x-mean
xt=xu.transpose()
sigmainv=inv(sigma)
print(len(sigmainv))
print(len(sigmainv[0]))
print(len(xt))
print(len(xt[0]))
print(len(xu))
print(len(xu[0]))
expval=np.multiply(xt,sigmainv) #the expontian value in the fx(x)
expval1=np.multiply(expval,xu) #the expontian value in the fx(x)
print(expval1)	


"""def multivar( nog ): #this functions clacutates the multivariate function of the gmm
	mean=np.random.uniform(1,10,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
	sig=np.random.uniform(1,10,size=(1,13)) #this is variance sigma ie variance
	siginv=np.reshape(sig, (13,1)) #to get a inverse martix of sigma
	pi=math.pow(2*pi, 13/2) #the value of (2*pi)^(nog/2)
	sigma=np.multiply(sig, siginv) #this will give me a sigma matrix a multiplication of the variances
	x=[i[0] for i in array2d]
	xu=x-mean
	"""

