import pandas as pd
import pdb

RECOGNIZED = 'yes no up down left right on off stop go'.split()

vad_data = pd.read_csv('output/max_vads.csv')
nn_data = pd.read_csv('GCommandsPytorch/nn_output.csv')

# Sort everything by label
vad_data = vad_data.sort_values(by = ['fname'])
nn_data = nn_data.sort_values(by = ['fname'])

VAD_THRESHOLD = 0.7
UNKNOWN_THRESHOLD = -1

labels = []
for ix in xrange(len(vad_data)):
  clip_name = vad_data.iloc[ix]['fname']
  nn_val = nn_data.iloc[ix]['label']
  nn_conf = nn_data.iloc[ix]['confidence']
  max_vad = vad_data.iloc[ix]['vad']

  ans = None
  if max_vad < VAD_THRESHOLD:
    ans = 'silence'
  elif nn_conf < UNKNOWN_THRESHOLD:
    ans = 'unknown'
  elif nn_val in RECOGNIZED:
    ans = nn_val
  else:
    ans = 'unknown'

  labels.append(ans)
  #print(clip_name, nn_val, silence_val, ans)

submit_data = pd.DataFrame.from_items([
  ('fname', nn_data['fname']),
  ('label', labels)
])

submit_data.to_csv('output/submission.csv', index = False)
