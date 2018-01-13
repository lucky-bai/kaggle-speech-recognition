import pandas as pd
import pdb

# Don't run this, max_vad version is better!
assert(False)

RECOGNIZED = 'yes no up down left right on off stop go'.split()

silence_data = pd.read_csv('output/silence.csv')
nn_data = pd.read_csv('output/nn_output.csv')

# Sort everything by label
silence_data = silence_data.sort_values(by = ['fname'])
nn_data = nn_data.sort_values(by = ['fname'])

labels = []
for ix in xrange(len(silence_data)):
  clip_name = silence_data.iloc[ix]['fname']
  silence_val = silence_data.iloc[ix]['label']
  nn_val = nn_data.iloc[ix]['label']

  ans = None
  if silence_val == 'silence':
    ans = 'silence'
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
