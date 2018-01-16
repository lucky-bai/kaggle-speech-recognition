import pandas as pd
import pdb

data1 = pd.read_csv('GCommandsPytorch/checkpoint/model6_lb085.csv')
data2 = pd.read_csv('GCommandsPytorch/checkpoint/model_all_train_lb086.csv')
data3 = pd.read_csv('GCommandsPytorch/checkpoint/model_all_train_lb086_v2.csv')

data1 = data1.sort_values(by = ['fname'])
data2 = data2.sort_values(by = ['fname'])
data3 = data3.sort_values(by = ['fname'])

labels = []
for ix in xrange(len(data1)):
  clip_name = data1.iloc[ix]['fname']
  val1 = data1.iloc[ix]['label']
  val2 = data2.iloc[ix]['label']
  val3 = data3.iloc[ix]['label']

  if val1 == val2:
    ans = val1
  elif val1 == val3:
    ans = val1
  elif val2 == val3:
    ans = val2
  else:
    ans = val3

  #print(clip_name, val1, val2, val3, ans)
  labels.append(ans)

submit_data = pd.DataFrame.from_items([
  ('fname', data1['fname']),
  ('label', labels)
])

submit_data.to_csv('output/ensemble_of_3.csv', index = False)
