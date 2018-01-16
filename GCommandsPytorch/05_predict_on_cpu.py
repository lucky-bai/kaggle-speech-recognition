# Ran out of GPU hours :(
from __future__ import print_function
from argparse import Namespace
import torch
import torch.optim as optim
import torch.utils.data as data
from gcommand_loader import GCommandLoader, spect_loader
import numpy as np
from model import LeNet, VGG
from train import train, test
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pdb
import sys
import csv
import multiprocessing as mp

ANSWER_DICT = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven shiela six stop three tree two up wow yes zero'.split()

args = Namespace(
  test_path = '../test/audio',
  window_size = .02,
  window_stride = .01,
  window_type = 'hamming',
  normalize = True,
  test_batch_size = 1,
  cuda = False
)

test_files = os.listdir(args.test_path)

checkpoint = torch.load('checkpoint/model6_lb085.t7', map_location=lambda storage, loc: storage)
model = checkpoint['net']

ix = mp.Value('i', 0)
lock = mp.Lock()

def process_one(f):
  global csvw
  global csvf

  ff = args.test_path + '/' + f
  spect = spect_loader(ff, args.window_size, args.window_stride, args.window_type, args.normalize, 101)
  data = spect.unsqueeze(0)

  data = Variable(data, volatile = True)
  output = model(data)

  # Conf: higher is better
  preds = output.data.max(1)[1].tolist()
  confidences = output.data.max(1)[0].tolist()

  pred = ANSWER_DICT[preds[0]]
  conf = confidences[0]
  with lock:
    print(ix.value, f, pred, conf)
    csvw.writerow([f, pred, conf])
    csvf.flush()
    ix.value += 1

with open('checkpoint/model6_lb085.csv', 'w') as csvf:
  csvw = csv.writer(csvf)
  csvw.writerow(['fname', 'label', 'confidence'])
  csvf.flush()

  pool = mp.Pool(processes = 50)
  pool.map(process_one, test_files)
