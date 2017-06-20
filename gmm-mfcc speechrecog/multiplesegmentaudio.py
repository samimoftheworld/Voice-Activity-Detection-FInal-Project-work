#segment multiple file in python
from pydub import AudioSegment
from pydub.silence import split_on_silence
#sound = AudioSegment.from_mp3("my_file.mp3")
k=1
for i in range(1,91): #here i have taken 100 files to be sample
	sound = AudioSegment.from_wav("/home/samim/project-work-in-cdac/audiotrainingset/training/n%d.wav"%i) #importing the wan file
	length =len(sound) #find length of the wav file in mili second 1s=1000ms
	le=length/10000+1 #take lower limit of the length/1000 
	#seg=1 #declare segment
	for j in range(0,le):
		if(j==le):
			le1=(j+1)*10000+1 #initial time for the last frame
			seg=sound[le1:length] #the remaining time stored here
			seg.export("/home/samim/chunk%d.wav"%(k+1), format="wav") #export the segment
			k=k+1
		elif(le==1):
			le1=(j+1)*10000+1 #initial time for the last frame
			seg=sound[le1:length] #the remaining time stored here
			seg.export("/home/samim/chunk%d.wav"%(k+1), format="wav") #export the segment
			k=k+1
		else:
			le1=j*10000+1 #initial time
			le2=(j+1)*10000 #final time
			seg=sound[le1:le2] #chunk out segment
			seg.export("/home/samim/chunk%d.wav"%(k+1), format="wav") #export the segment
			k=k+1
