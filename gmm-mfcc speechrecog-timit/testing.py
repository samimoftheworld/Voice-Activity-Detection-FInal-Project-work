from scikits.audiolab import Sndfile
import numpy as np
framelen=10
sf2 = Sndfile('/home/samim/audiotrainingset/testchunk/chunk1.wav', "r")
chunk = sf2.read_frames(20, dtype=np.float32)
print chunk
sf = Sndfile('/home/samim/audiotrainingset/testchunk/chunk1.wav', "r")
while(True):
	
	try:
		chunk = sf.read_frames(framelen, dtype=np.float32)
		print chunk
		if len(chunk) != framelen:
			print("Not read sufficient samples - returning")
			break
	except RuntimeError:
		break

