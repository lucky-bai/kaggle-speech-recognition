# Like predict, but also output the probability
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

ANSWER_DICT = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven shiela six stop three tree two up wow yes zero'.split()

args = Namespace(
  test_path = '../test/audio',
  window_size = .02,
  window_stride = .01,
  window_type = 'hamming',
  normalize = True,
  test_batch_size = 100,
  cuda = True
)

test_files = os.listdir(args.test_path)

class TestDataLoader(data.Dataset):
  def __getitem__(self, index):
    ff = args.test_path + '/' + test_files[index]
    spect = spect_loader(ff, args.window_size, args.window_stride, args.window_type, args.normalize, 101)
    return test_files[index], spect

  def __len__(self):
    return len(test_files)


test_loader = torch.utils.data.DataLoader(
  TestDataLoader(),
  batch_size = args.test_batch_size,
  shuffle = None,
  num_workers = 20,
  pin_memory = args.cuda,
  sampler = None
)


#checkpoint = torch.load('checkpoint/model_6.t7')
checkpoint = torch.load('checkpoint/model_all_train_lb086_v2.t7')
model = checkpoint['net']

csvw = csv.writer(open('nn_output.csv', 'w'))
csvw.writerow(['fname', 'label', 'confidence'])

ix = 0
for f, data in test_loader:
  data = Variable(data, volatile = True).cuda()
  output = model(data)

  # Conf: higher is better
  preds = output.data.max(1)[1].tolist()
  confidences = output.data.max(1)[0].tolist()

  for j in xrange(len(preds)):
    pred = ANSWER_DICT[preds[j]]
    conf = confidences[j]
    print(ix, f[j], pred, conf)
    ix += 1
    csvw.writerow([f[j], pred, conf])
