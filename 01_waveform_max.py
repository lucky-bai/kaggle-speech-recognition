import numpy as np
import wave
import os

ROOTDIR = 'test/audio/'

for wavfile in filter(lambda x: '.wav' in x, os.listdir(ROOTDIR)):
  wavfile = ROOTDIR + '/' + wavfile
  spf = wave.open(wavfile,'r')

  signal = spf.readframes(-1)
  signal = np.fromstring(signal, 'Int16')

  #print wavfile, max(abs(signal))
  print max(abs(signal))
  #if max(abs(signal)) == 12:
    #print list(signal)
