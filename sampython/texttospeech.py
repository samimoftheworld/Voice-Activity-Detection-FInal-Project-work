#program to change text into speech
#importing a file from file dialog box
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

#reading a file

file = open(file_path, "r")

from gtts import gTTS
tts = gTTS(text=file.read( ), lang='en')


#to save a file in mp3 format
f = 'output.mp3'
tts.save(f)

