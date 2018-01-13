import numpy as np
import wave
import os
import csv

THRESHOLD = 1138
ROOTDIR = 'test/audio/'


csvw = csv.writer(open('silence.csv', 'w'))
csvw.writerow(['fname', 'label'])


ix = 0
for wavfile in filter(lambda x: '.wav' in x, os.listdir(ROOTDIR)):
  owavfile = ROOTDIR + '/' + wavfile
  spf = wave.open(owavfile,'r')

  signal = spf.readframes(-1)
  signal = np.fromstring(signal, 'Int16')

  pred = 'no'
  if max(abs(signal)) < THRESHOLD:
    pred = 'silence'

  csvw.writerow([wavfile, pred])

  ix += 1
  if ix % 100 == 0:
    print ix
