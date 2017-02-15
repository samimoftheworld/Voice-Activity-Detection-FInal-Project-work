#a code for gmm creation from mfcc with correct methods
import numpy as np
import math
from scipy import linalg
from mpmath import *
def multivar(array2d,mean,sigma): #this functions clacutates the multivariate function of the gmm ie the fxx value
	#check=1
	#while (check==1):
		#w= np.random.dirichlet(np.ones(13),size=1) #this is the weignts of the gausians and the function generates random numbers with total sum = 1
		#numrows = len(array2d)    # length of rows in your example
		#numcols = len(array2d[0]) # length of columns in your example
		#mean=np.random.uniform(1,4,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
		#sig=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
		##print mean
		##print sig
		#siginv=np.reshape(sig, (13,1)) #to get a inverse martix of sigma
		#pi=math.pow(2*3.14, 13/2) #the value of (2*pi)**(nog/2) 
		#sigma=np.multiply(sig, siginv) #this will give me a sigma matrix a multiplication of the variances
		#si=np.linalg.det(sigma) #for the determinant of the sigma matrix
		##print si
		#if(si>0):
		#	check=0
			##print si
	pi = (2*math.pi)**6.5
	x=array2d
	xu=x-mean
	#print mean
	#print x
	#print xu
	#xt=xu.transpose()
	#print np.shape(xt)
	sigmainv=linalg.pinv2(sigma)
	#print(len(sigmainv))
	#print(len(sigmainv[0]))
	#print(len(xt))
	#print(len(xt[0]))
	#print(len(xu))
	#print(len(xu[0]))
	#print np.shape(xu)
	expval=np.dot(xu,sigmainv) #the expontian value in the fx(x)
	#print(np.shape(expval))
	expval1=np.dot(expval,xu.T) #the expontian value in the fx(x)
	#print(expval1)
	#si=np.linalg.det(sigma) #for the determinant of the sigma matrix
	si=np.linalg.det(sigma)/2
	expo=np.exp(expval1) #divided by 10000000000000 just to procede remove later
	#expo=math.e**expval1 #the final value of exponent
	#mp.dps=300
	#expo=exp(expval1)
	##print expval1
	##print expo
	fxx=(1/pi*si)*expo
	##print fxx
	return fxx[0]

def resterm(w,fxxn,fxxd,numrows): #here fxxn is the neumerator fxx and the fxxd is the denomination fxx in eqn rn(1)=w1*fxx1(xn,u1,e1)/(w1*fxx1(xn,u1,e1)+w2*fxx2(xn,u2,e2))
	a=[x * w[0] for x in fxxn] #this does w[0]*fxxn 
	b=[x * w[1] for x in fxxd] #this does w[1]*fxxd
	r=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
	#r2=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
	for i in range (0,numrows):
		rterm=a[i]/(a[i]+b[i])
		r.append(rterm)
	#print r
	return r
		
def getsigma(r,array2d,numrows,mean):
	#print array2d[0]
	lal=array2d[0]
	lala=np.reshape(lal,(13,1))
	lal=np.reshape(lal,(1,13))
	#print lal
	#print lala
	brr=np.dot(lala,lal)
	#print brr
	brr=np.subtract(brr,brr) #just to makle a array of mfcc sum
	for j in range (0,numrows):
		arr=array2d[j]
		xu=arr-mean
		mul=np.dot(xu,np.transpose(xu))
		a=mul*r[j] #this does r[j]*arr 
		brr=brr+a #sum of all arrays 4
	#a=np.dot(r,array2d) #this does w[0]*fxxn 
		b=0
		for i in range (0,numrows):
			b=b+r[i]
	#print brr
	sig=np.divide(brr,b)
	return sig
	
def getmean(r,array2d,numrows):
	brr=array2d[0]
	brr=np.subtract(brr,brr) #just to makle a array of mfcc sum
	for j in range (0,numrows):
		arr=array2d[j]
		a=[x* r[j] for x in arr] #this does r[j]*arr 
		brr=brr+a #sum of all arrays 4
	#a=np.dot(r,array2d) #this does w[0]*fxxn 
		b=0
		for i in range (0,numrows):
			b=b+r[i]
	mean=np.divide(brr,b)
	return mean
		

def getweight(r,numrows):
	b=0
	for i in range (0,numrows):
		b=b+r[i]
	weight=b/numrows
	print b
	print numrows
	return weight
	
	

#python code to read the mfcc and then make gmm
with open('/home/samim/test.txt','r') as file:
	array2d = [[float(digit) for digit in line.split()] for line in file]
#print(array2d) #to print ful array
#print(array2d[1]) #to print only one array row
#print(array2d[1][3]) #to print only one element
#print([i[0] for i in array2d]) #to print only one array column
nog=2 #no of gausians is 13 as we are having a 13 dimentional martix
#array2d=linalg.svd(array2d)
#array2d[3][2]
numrows = len(array2d)    # length of rows in your example
fxx1=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
fxx2=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
#r1=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
#r2=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx

#w= np.random.dirichlet(np.ones(13),size=1) #this is the weignts of the gausians and the function generates random numbers with total sum = 1
w = [0.5,0.5]
#numrows = len(array2d)    # length of rows in your example
#numcols = len(array2d[0]) # length of columns in your example
mean1=np.random.uniform(1,4,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
mean2=np.random.uniform(1,4,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
sig1=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
siginv1=np.reshape(sig1, (13,1)) #to get a inverse martix of sigma
sigma1=np.multiply(sig1, siginv1) #this will give me a sigma matrix a multiplication of the variances
sig2=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
siginv2=np.reshape(sig2, (13,1)) #to get a inverse martix of sigma
sigma2=np.multiply(sig2, siginv2) #this will give me a sigma matrix a multiplication of the variances
ck=1
while(ck==1):
	for i in range (0,numrows):
		#print array2d[i]
		sam=multivar(array2d[i],mean1,sigma1)
		fxx1.append(sam)
		sam=multivar(array2d[i],mean2,sigma2)
		fxx2.append(sam)
	#print sam
	#print fxx1 #all the fxx stored here
	#print fxx2 #all the fxx stored here
	r1=(resterm(w,fxx1,fxx2,numrows)) #to get the responsibility term rn1 value of fxx
	r2=(resterm(w,fxx2,fxx1,numrows)) #to get the responsibility term rn2 value of fxx
	u1=getmean(r1,array2d,numrows)
	u2=getmean(r2,array2d,numrows)
	w1=getweight(r1,numrows)
	w2=getweight(r2,numrows)
	sigma11=getsigma(r1,array2d,numrows,mean1)
	sigma22=getsigma(r2,array2d,numrows,mean2)
	print 'prev stuff'
	print w[0],w[1],mean1,mean2,sigma1,sigma2
	print 'current stuff'
	print w1,w2,u1,u2,sigma11,sigma22
	if(np.array_equal(w1,w[0])):
		if(np.array_equal(w2,w[1])):
			if(np.array_equal(u1,mean1)):
				if(np.array_equal(u2,mean2)):
					if(np.array_equal(sigma1,sigma11)):
						if(np.array_equal(sigma2,sigma22)):
							ck=0
	w[0]=w1
	w[1]=w2
	mean1=u1
	mean2=u2
	#print sigma11
	sigma1=sigma11
	sigma2=sigma22
print 'the best mean is '
print mean1
print mean2
print 'the best sigma is '
print sigma1
print sigma2
print 'the best weight is '
print w	
"""def multivar( nog ): #this functions clacutates the multivariate function of the gmm
	mean=np.random.uniform(1,10,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
	sig=np.random.uniform(1,10,size=(1,13)) #this is variance sigma ie variance
	siginv=np.reshape(sig, (13,1)) #to get a inverse martix of sigma
	pi=math.pow(2*pi, 13/2) #the value of (2*pi)^(nog/2)
	sigma=np.multiply(sig, siginv) #this will give me a sigma matrix a multiplication of the variances
	x=[i[0] for i in array2d]
	xu=x-mean
	"""

