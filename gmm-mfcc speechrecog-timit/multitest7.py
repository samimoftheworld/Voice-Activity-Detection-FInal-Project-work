#testing multiple file by creationg a gmm for more data
from Smacpy import Smacpy
import cPickle as pickle
import numpy as np
#model = Smacpy("/home/samim/audiotrainingset/training", {'n-1.wav':'nonspeech',  'n-2.wav':'nonspeech',  'n-3.wav':'nonspeech',  'n-4.wav':'nonspeech',  'n-5.wav':'nonspeech',  'n-6.wav':'nonspeech',  'n-7.wav':'nonspeech',  'n-8.wav':'nonspeech',  'n-9.wav':'nonspeech',  'n-10.wav':'nonspeech',  'n-11.wav':'nonspeech',  'n-12.wav':'nonspeech',  'n-13.wav':'nonspeech',  'n-14.wav':'nonspeech',  'n-15.wav':'nonspeech',  'n-16.wav':'nonspeech',  'n-17.wav':'nonspeech',  'n-18.wav':'nonspeech',  'n-19.wav':'nonspeech',  'n-20.wav':'nonspeech',  'n-21.wav':'nonspeech',  'n-22.wav':'nonspeech',  'n-23.wav':'nonspeech',  'n-24.wav':'nonspeech',  'n-25.wav':'nonspeech',  'n-26.wav':'nonspeech',  'n-27.wav':'nonspeech',  'n-28.wav':'nonspeech',  'n-29.wav':'nonspeech',  'n-30.wav':'nonspeech',  'n-31.wav':'nonspeech',  'n-32.wav':'nonspeech',  'n-33.wav':'nonspeech',  'n-34.wav':'nonspeech',  'n-35.wav':'nonspeech',  'n-36.wav':'nonspeech',  'n-37.wav':'nonspeech',  'n-38.wav':'nonspeech',  'n-39.wav':'nonspeech',  'n-40.wav':'nonspeech',  'n-41.wav':'nonspeech',  'n-42.wav':'nonspeech',  'n-43.wav':'nonspeech',  'n-44.wav':'nonspeech',  'n-45.wav':'nonspeech',  'n-46.wav':'nonspeech',  'n-47.wav':'nonspeech',  'n-48.wav':'nonspeech',  'n-49.wav':'nonspeech',  'n-50.wav':'nonspeech',  'n-51.wav':'nonspeech',  'n-52.wav':'nonspeech',  'n-53.wav':'nonspeech',  'n-54.wav':'nonspeech',  'n-55.wav':'nonspeech',  'n-56.wav':'nonspeech',  'n-57.wav':'nonspeech',  'n-58.wav':'nonspeech',  'n-59.wav':'nonspeech',  'n-60.wav':'nonspeech',  'n-61.wav':'nonspeech',  'n-62.wav':'nonspeech',  'n-63.wav':'nonspeech',  'n-64.wav':'nonspeech',  'n-65.wav':'nonspeech',  'n-66.wav':'nonspeech',  'n-67.wav':'nonspeech',  'n-68.wav':'nonspeech',  'n-69.wav':'nonspeech',  'n-70.wav':'nonspeech',  'n-71.wav':'nonspeech',  'n-72.wav':'nonspeech',  'n-73.wav':'nonspeech',  'n-74.wav':'nonspeech',  'n-75.wav':'nonspeech',  'n-76.wav':'nonspeech',  'n-77.wav':'nonspeech',  'n-78.wav':'nonspeech',  'n-79.wav':'nonspeech',  'n-80.wav':'nonspeech',  'n-81.wav':'nonspeech',  'n-82.wav':'nonspeech',  'n-83.wav':'nonspeech',  'n-84.wav':'nonspeech',  'n-85.wav':'nonspeech',  'n-86.wav':'nonspeech',  'n-87.wav':'nonspeech',  'n-88.wav':'nonspeech',  'n-89.wav':'nonspeech',  'n-90.wav':'nonspeech', 'n-109.wav':'nonspeech','n-110.wav':'nonspeech',   'sp-1.wav':'speech',  'sp-2.wav':'speech',  'sp-3.wav':'speech',  'sp-4.wav':'speech',  'sp-5.wav':'speech',  'sp-6.wav':'speech',  'sp-7.wav':'speech',  'sp-8.wav':'speech',  'sp-9.wav':'speech',  'sp-10.wav':'speech',  'sp-11.wav':'speech',  'sp-12.wav':'speech',  'sp-13.wav':'speech',  'sp-14.wav':'speech',  'sp-15.wav':'speech',  'sp-16.wav':'speech',  'sp-17.wav':'speech',  'sp-18.wav':'speech',  'sp-19.wav':'speech',  'sp-20.wav':'speech',  'sp-21.wav':'speech',  'sp-22.wav':'speech',  'sp-23.wav':'speech',  'sp-24.wav':'speech',  'sp-25.wav':'speech',  'sp-26.wav':'speech',  'sp-27.wav':'speech',  'sp-28.wav':'speech',  'sp-29.wav':'speech',  'sp-30.wav':'speech',  'sp-31.wav':'speech',  'sp-32.wav':'speech',  'sp-33.wav':'speech',  'sp-34.wav':'speech',  'sp-35.wav':'speech',  'sp-36.wav':'speech',  'sp-37.wav':'speech',  'sp-38.wav':'speech',  'sp-39.wav':'speech',  'sp-40.wav':'speech',  'sp-41.wav':'speech',  'sp-42.wav':'speech',  'sp-43.wav':'speech',  'sp-44.wav':'speech',  'sp-45.wav':'speech',  'sp-46.wav':'speech',  'sp-47.wav':'speech',  'sp-48.wav':'speech',  'sp-49.wav':'speech',  'sp-50.wav':'speech',  'sp-51.wav':'speech',  'sp-52.wav':'speech',  'sp-53.wav':'speech',  'sp-54.wav':'speech',  'sp-55.wav':'speech',  'sp-56.wav':'speech',  'sp-57.wav':'speech',  'sp-58.wav':'speech',  'sp-59.wav':'speech',  'sp-60.wav':'speech',  'sp-61.wav':'speech',  'sp-62.wav':'speech',  'sp-63.wav':'speech',  'sp-64.wav':'speech',  'sp-65.wav':'speech',  'sp-66.wav':'speech',  'sp-67.wav':'speech', })
#pickle.dump( model, open( "model.p", "wb" ) )
model= pickle.load( open( "model32filter/model35.p", "rb" ) )
x = ["" for y in range(2870)] #strign array in python
y=np.zeros((2870,2))
j=1
err1=0
err2=0
for i in xrange(1,66):
	x[j],y[j]=model.classify('/home/samim/audiotrainingset/chunktimit/ntestchunk%d.wav'%i)
	j=j+1
for i in xrange(1,66):
	x[j],y[j]=model.classify('/home/samim/audiotrainingset/chunktimit/sptestchunk%d.wav'%i)
	j=j+1
for i in xrange(1,66):
	print 'So the %d file is calssified as'%i,x[i],'with sp and np frames as ',y[i] ,'error = ',(y[i][0]/(y[i][0]+y[i][1]))
	err1=err1+(y[i][0]/(y[i][0]+y[i][1]))
for i in xrange(66,j):
	print 'So the %d file is calssified as'%(i-110),x[i],'with sp and np frames as ',y[i],'error = ',(y[i][1]/(y[i][0]+y[i][1]))
	err2=err2+(y[i][1]/(y[i][0]+y[i][1]))
test=0
for i in xrange(1,66):
	if(x[i]!='nonspeech'):
		test=test+1
for i in xrange(66,j):
	if(x[i]!='speech'):
		test=test+1
print test
print j
error=float(test)/float(j)
print error
accuracy=float(1.0-error)
print 'accuracy:',accuracy
print 'average NS error',err1/110
print 'average SP error',err2/66

print 'total frame error',((err1+err2)/j)

#j=1
#for i in xrange(1,1500):
#	x[j]=model.classify('/home/samim/audiotrainingset/testchunk/chunk%d.wav'%i)
#	j=j+1
#test=1
#for i in xrange(1,1200):
#	if(x[i]!='nonspeech'):
#		test=test+1
#for i in xrange(1200,1500):
#	if(x[i]!='speech'):
#		test=test+1
#print 'for my audio file '
#for i in xrange(1,59):
#	print 'So the %d file is calssified as'%i,x[i]
#print test
#error=float(test)/float(j)
#print error
#accuracy=float(1.0-error)
#print 'accuracy:',accuracy
