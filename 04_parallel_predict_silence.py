# Same as predict_silence but do in parallel
import numpy as np
import wave
import os
import csv
import multiprocessing as mp

THRESHOLD = 1138
ROOTDIR = 'test/audio/'

num_processed = mp.Value('i', 0)

def process_wav(wavfile):
  global num_processed
  num_processed.value += 1
  if num_processed.value % 100 == 0:
    print(num_processed.value)

  owavfile = ROOTDIR + '/' + wavfile
  spf = wave.open(owavfile,'r')

  signal = spf.readframes(-1)
  signal = np.fromstring(signal, 'Int16')

  pred = 'no'
  if max(abs(signal)) < THRESHOLD:
    pred = 'silence'

  return wavfile, pred



wavfiles = filter(lambda x: '.wav' in x, os.listdir(ROOTDIR))

pool = mp.Pool(processes = 50)
results = [pool.apply(process_wav, args = (f,)) for f in wavfiles]


csvw = csv.writer(open('silence.csv', 'w'))
csvw.writerow(['fname', 'label'])

for wavfile, pred in results:
  csvw.writerow([wavfile, pred])

