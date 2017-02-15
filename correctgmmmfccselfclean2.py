#a code for gmm creation from mfcc with correct methods but cleaner
import numpy as np
import math
from scipy import linalg
from mpmath import *
def multivar(array2d,mean,sigma): #this functions clacutates the multivariate function of the gmm ie the fxx value
	pi = (2*math.pi)**6.5
	x=array2d
	xu=x-mean
	sigmainv=linalg.inv(sigma)
	#print sigmainv
	expval=np.dot(xu,sigmainv) #the expontian value in the fx(x)
	expval1=(-0.5)*np.dot(expval,xu.T) #the expontian value in the fx(x)
	#expval1=(-0.5)*expval1
	#print expval,xu,expval1
	si=(np.linalg.det(sigma))**0.5
	#print si
	expo=np.exp(expval1)	 #divided by 10000000000000 just to procede remove later
	#print pi
	print expo
	fxx=(1/(pi*si))*expo
	print fxx
	#print np.shape(fxx)
	return fxx

def resterm(w1,w2,fxxn,fxxd,numrows): #here fxxn is the neumerator fxx and the fxxd is the denomination fxx in eqn rn(1)=w1*fxx1(xn,u1,e1)/(w1*fxx1(xn,u1,e1)+w2*fxx2(xn,u2,e2))
	a=[x * w1 for x in fxxn] #this does w[0]*fxxn 
	b=[x * w2 for x in fxxd] #this does w[1]*fxxd
	r=np.zeros(numrows) #depends on the total no of gausians since we have 2 gausians here so 2 fxx
	for i in range (0,numrows):
		#print a[i]+b[i]
		r[i]=a[i]/(a[i]+b[i])
		
	print a[0],b[0]
	return r
		
def getsigma(r,array2d,numrows,mean):
	#lal=array2d[0]
	#lala=np.reshape(lal,(13,1))
	#lal=np.reshape(lal,(1,13))
	#brr=np.dot(lala,lal)
	#brr=np.subtract(brr,brr) #just to makle a array of mfcc sum
	brr=np.zeros((13,13))
	#print 'this is brr'
	#print brr
	for j in range (0,numrows):
		arr=array2d[j]
		xu=arr-mean
		mul=np.dot(np.transpose(xu),xu)
		#print mul
		a=mul*r[j] #this does r[j]*arr 
		#print a
		brr=brr+a #sum of all arrays 4
	#print brr
	b=0
	for i in range (0,numrows):
			b=b+r[i]
	sig=brr/b
	return sig
	
def getmean(r,array2d,numrows):
	#brr=array2d[0]
	#brr=np.subtract(brr,brr) #just to makle a array of mfcc sum
	#print 'this is brr'
	#print brr
	brr = np.zeros((1,13))
	
	for j in range (0,numrows):
		a=r[j]*array2d[j]
		#a=[x* r[j] for x in arr] #this does r[j]*arr 
		brr=brr+a #sum of all arrays / data in the mfcc ie E(rn1*xn)
		#brr=[x + y for x, y in zip(brr,a)] #this adds the matrices together[x + y for x, y in zip(first, second)]
	#print 'this is brr'
	#print brr
	b=np.sum(r)
	#print 'this is b'
	#print b
	mean=brr/b
	return mean		

def getweight(r,numrows):
	b=0
	#for i in range (0,numrows):
	#	b=b+r[i]
	#b = np.sum(r)
	weight=np.sum(r)/numrows
	print weight	
	print numrows
	return weight
	
	

#python code to read the mfcc and then make gmm
with open('/home/samim/test.txt','r') as file:
	array2d = [[float(digit) for digit in line.split()] for line in file]
print type(array2d)
array2d=np.array(array2d)
print np.shape(array2d)
print type(array2d)
nog=2 #no of gausians is 13 as we are having a 13 dimentional martix
numrows = len(array2d)    # length of rows in your example
fxx1=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx
fxx2=[]*numrows #depends on the total no of gausians since we have 2 gausians here so 2 fxx

#w = [0.5,0.5]
ww1=0.5
ww2=0.5
mean1=np.random.uniform(1,4,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
mean2=np.random.uniform(1,4,size=(1,13))  #this is mean of the gausiansthe function generates random numbers in a nxm dimention martix in size=(n,m) format for the input matrix
#sig1=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
#siginv1=np.reshape(sig1, (13,1)) #to get a inverse martix of sigma
#sigma1=np.multiply(sig1, siginv1) #this will give me a sigma matrix a multiplication of the variances
sigma1=np.eye(13)
#sig2=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
#siginv2=np.reshape(sig2, (13,1)) #to get a inverse martix of sigma
#sigma2=np.multiply(sig2, siginv2) #this will give me a sigma matrix a multiplication of the variances
sigma2=np.eye(13)
#check=1
#while (check==1):
#	sig1=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
#	siginv1=np.reshape(sig1, (13,1)) #to get a inverse martix of sigma
#	sigma1=np.multiply(sig1, siginv1) #this will give me a sigma matrix a multiplication of the variances
#	si1=np.linalg.det(sigma1) #for the determinant of the sigma matrix
#	if(si1>0):
#			check=0
#check=1
#while (check==1):
#	sig2=np.random.uniform(1,4,size=(1,13)) #this is variance sigma ie variance
#	siginv2=np.reshape(sig2, (13,1)) #to get a inverse martix of sigma
#	sigma2=np.multiply(sig2, siginv2) #this will give me a sigma matrix a multiplication of the variances
#	si2=np.linalg.det(sigma2) #for the determinant of the sigma matrix
#	if(si2>0):
#			check=0
ck=1
while(ck==1):
	for i in range (0,numrows):
		sam=multivar(array2d[i],mean1,sigma1)
		fxx1.append(sam)
		sam=multivar(array2d[i],mean2,sigma2)
		fxx2.append(sam)
	r1=(resterm(ww1,ww2,fxx1,fxx2,numrows))	#to get the responsibility term rn1 value of fxx
	print r1
	r2=(resterm(ww2,ww1,fxx2,fxx1,numrows)) #to get the responsibility term rn2 value of fxx
	print r1[0],r2[0]
	u1=getmean(r1,array2d,numrows)
	u2=getmean(r2,array2d,numrows)
	w1=getweight(r1,numrows)
	w2=getweight(r2,numrows)
	sigma11=getsigma(r1,array2d,numrows,mean1)
	sigma22=getsigma(r2,array2d,numrows,mean2)
	print sigma11
	#print 'prev stuff'
	#print w[0],w[1],mean1,mean2,sigma1,sigma2
	#print 'current stuff'
	#print w1,w2,u1,u2,sigma11,sigma22
	if(np.array_equal(w1,ww1)):
		if(np.array_equal(w2,ww2)):
			if(np.array_equal(u1,mean1)):
				if(np.array_equal(u2,mean2)):
					if(np.array_equal(sigma1,sigma11)):
						if(np.array_equal(sigma2,sigma22)):
							ck=0
	ww1=w1
	ww2=w2
	mean1=u1
	mean2=u2
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
