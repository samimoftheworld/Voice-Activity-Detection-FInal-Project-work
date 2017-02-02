#segment a single file in python
from pydub import AudioSegment
from pydub.silence import split_on_silence
#sound = AudioSegment.from_mp3("my_file.mp3")
sound = AudioSegment.from_wav("/home/samim/my_file.wav") #importing the wan file
length = len(sound) #find length of the wav file in mili second 1s=1000ms
le=length/10000+1 #take lower limit of the length/1000 
seg=1 #declare segment
for i in range(0,le):
	if(i==le):
		le1=(i+1)*10000+1 #initial time for the last frame
		seg=sound[le1:length] #the remaining time stored here
		seg.export("/home/samim/chunk%d.wav"%i+1, format="wav") #export the segment
	else:
		le1=i*10000+1 #initial time
		le2=(i+1)*10000 #final time
		seg=sound[le1:le2] #chunk out segment
		seg.export("/home/samim/chunk%d.wav"%i, format="wav") #export the segment
