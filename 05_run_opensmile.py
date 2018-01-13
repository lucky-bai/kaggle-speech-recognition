# Run opensmile voice detection for the whole data
import os
import time
import multiprocessing as mp

OPENSMILE_PATH = '../opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
CONF_FILE_PATH = '../opensmile-2.3.0/scripts/vad/vad_opensource.conf'
TEST_FILE_PATH = '../subset/clip_6d7e364c4.wav'
#DATA_PATH = '../subset/'
DATA_PATH = 'test/audio/'

#SMILE_CMD = '%s -C %s -I %s -O smile.csv' % (OPENSMILE_PATH, CONF_FILE_PATH, TEST_FILE_PATH)
SMILE_CMD = '%s -C %s -I %s -O %s 2> /dev/null'

fs = os.listdir(DATA_PATH)

def process_file(f):
  ff = DATA_PATH + f
  csvpath = 'smile_out/%s.csv' % f
  cmd = SMILE_CMD % (OPENSMILE_PATH, CONF_FILE_PATH, ff, csvpath)
  os.system(cmd)

#pool = mp.Pool(processes = 16)
#results = [pool.apply(process_file, args = (f,)) for f in fs]

ix = 0
for f in fs:
  print(ix, f)
  process_file(f)
  ix += 1

