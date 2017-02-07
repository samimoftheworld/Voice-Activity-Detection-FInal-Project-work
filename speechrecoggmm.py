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
from Smacpy import Smacpy
model = Smacpy("/home/samim/audiotrainingset/training", {'n1.wav':'nonspeech', 'n2.wav':'nonspeech', 'n3.wav':'nonspeech', 'n4.wav':'nonspeech', 'n5.wav':'nonspeech', 'n6.wav':'nonspeech', 'n7.wav':'nonspeech', 'n8.wav':'nonspeech', 'n9.wav':'nonspeech', 'n10.wav':'nonspeech', 'n11.wav':'nonspeech', 'n12.wav':'nonspeech', 'n13.wav':'nonspeech', 'n14.wav':'nonspeech', 'n15.wav':'nonspeech', 'n16.wav':'nonspeech', 'n17.wav':'nonspeech', 'n18.wav':'nonspeech', 'n19.wav':'nonspeech', 'n20.wav':'nonspeech', 'n21.wav':'nonspeech', 'n22.wav':'nonspeech', 'n23.wav':'nonspeech', 'n24.wav':'nonspeech', 'n25.wav':'nonspeech', 'n26.wav':'nonspeech', 'n27.wav':'nonspeech', 'n28.wav':'nonspeech', 'n29.wav':'nonspeech', 'n30.wav':'nonspeech', 'n31.wav':'nonspeech', 'n32.wav':'nonspeech', 'n33.wav':'nonspeech', 'n34.wav':'nonspeech', 'n35.wav':'nonspeech', 'n36.wav':'nonspeech', 'n37.wav':'nonspeech', 'n38.wav':'nonspeech', 'n39.wav':'nonspeech', 'n40.wav':'nonspeech', 'n41.wav':'nonspeech', 'n42.wav':'nonspeech', 'n43.wav':'nonspeech', 'n44.wav':'nonspeech', 'n45.wav':'nonspeech', 'n46.wav':'nonspeech', 'n47.wav':'nonspeech', 'n48.wav':'nonspeech', 'n49.wav':'nonspeech', 'n50.wav':'nonspeech', 'n51.wav':'nonspeech', 'n52.wav':'nonspeech', 'n53.wav':'nonspeech', 'n54.wav':'nonspeech', 'n55.wav':'nonspeech', 'n56.wav':'nonspeech', 'n57.wav':'nonspeech', 'n58.wav':'nonspeech', 'n59.wav':'nonspeech', 'n60.wav':'nonspeech', 'n61.wav':'nonspeech', 'n62.wav':'nonspeech', 'n63.wav':'nonspeech', 'n64.wav':'nonspeech', 'n65.wav':'nonspeech', 'n66.wav':'nonspeech', 'n67.wav':'nonspeech', 'n68.wav':'nonspeech', 'n69.wav':'nonspeech', 'n70.wav':'nonspeech', 'n71.wav':'nonspeech', 'n72.wav':'nonspeech', 'n73.wav':'nonspeech', 'n74.wav':'nonspeech', 'n75.wav':'nonspeech', 'n76.wav':'nonspeech', 'n77.wav':'nonspeech', 'n78.wav':'nonspeech', 'n79.wav':'nonspeech', 'n80.wav':'nonspeech', 'n81.wav':'nonspeech', 'n82.wav':'nonspeech', 'n83.wav':'nonspeech', 'n84.wav':'nonspeech', 'n85.wav':'nonspeech', 'n86.wav':'nonspeech', 'n87.wav':'nonspeech', 'n88.wav':'nonspeech', 'n89.wav':'nonspeech', 'n90.wav':'nonspeech', 'sp1.wav':'speech', 'sp2.wav':'speech', 'sp3.wav':'speech', 'sp4.wav':'speech', 'sp5.wav':'speech', 'sp6.wav':'speech', 'sp7.wav':'speech', 'sp8.wav':'speech', 'sp9.wav':'speech', 'sp10.wav':'speech', 'sp11.wav':'speech', 'sp12.wav':'speech', 'sp13.wav':'speech', 'sp14.wav':'speech', 'sp15.wav':'speech', 'sp16.wav':'speech', 'sp17.wav':'speech', 'sp18.wav':'speech', 'sp19.wav':'speech', 'sp20.wav':'speech', 'sp21.wav':'speech', 'sp22.wav':'speech', 'sp23.wav':'speech', 'sp24.wav':'speech', 'sp25.wav':'speech', 'sp26.wav':'speech', 'sp27.wav':'speech', 'sp28.wav':'speech', 'sp29.wav':'speech', 'sp30.wav':'speech', 'sp31.wav':'speech', 'sp32.wav':'speech', 'sp33.wav':'speech', 'sp34.wav':'speech', 'sp35.wav':'speech', 'sp36.wav':'speech', 'sp37.wav':'speech', 'sp38.wav':'speech', 'sp39.wav':'speech', 'sp40.wav':'speech', 'sp41.wav':'speech', 'sp42.wav':'speech', 'sp43.wav':'speech', 'sp44.wav':'speech', 'sp45.wav':'speech', 'sp46.wav':'speech', 'sp47.wav':'speech', 'sp48.wav':'speech', 'sp49.wav':'speech', 'sp50.wav':'speech', 'sp51.wav':'speech', 'sp52.wav':'speech', 'sp53.wav':'speech', 'sp54.wav':'speech', 'sp55.wav':'speech', 'sp56.wav':'speech', 'sp57.wav':'speech', 'sp58.wav':'speech', 'sp59.wav':'speech', 'sp60.wav':'speech', 'sp61.wav':'speech', 'sp62.wav':'speech', 'sp63.wav':'speech', 'sp64.wav':'speech', 'sp65.wav':'speech', 'sp66.wav':'speech', 'sp67.wav':'speech'})
x = ["" for y in range(le)] #strign array in python
j=1
for i in xrange(0,le):
	if(i==6):
		break
	x[j]=model.classify('/home/samim/chunk%d.wav'%i)
	j=j+1
for i in xrange(0,le):
	print 'So the %d file is calssified as'%i,x[i]
