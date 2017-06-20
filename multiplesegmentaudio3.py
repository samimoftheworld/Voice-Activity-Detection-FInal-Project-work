#segment multiple file in python
from pydub import AudioSegment
from pydub.silence import split_on_silence
#sound = AudioSegment.from_mp3("my_file.mp3")
k=1
for i in range(1,67): #here i have taken 100 files to be sample
	sound = AudioSegment.from_wav("/home/samim/audiotrainingset/training/sp-%d.wav"%i) #importing the wan file
	length =len(sound) #find length of the wav file in mili second 1s=1000ms
	le=1000 #take lower limit of the length/1000 
	#seg=1 #declare segment
	#for j in range(0,le):
	#	if(j==le):
	seg1=sound[0:le]
	seg=sound[le:length] #the remaining time stored here
	seg.export("/home/samim/audiotrainingset/chunktrain/sptrainchunk%d.wav"%i, format="wav") #export the segment
	seg1.export("/home/samim/audiotrainingset/chunktrain/sptestchunk%d.wav"%i, format="wav") #export the segment
	