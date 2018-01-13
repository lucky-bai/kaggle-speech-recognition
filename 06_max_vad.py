# For each file in smile_out, produce the max
import os
import pandas as pd

fs = os.listdir('smile_out')

fnames = []
vads = []

for f in fs:
  ff = f[:-4]
  if len(vads) % 100 == 0:
    print(len(vads))

  data = pd.read_csv('smile_out/' + f, header = None)
  max_vad = data.max(axis = 0)[1]
  vad_quantile = data.quantile(q = 0.85, axis = 0)[1]

  fnames.append(ff)
  #vads.append(max_vad)
  vads.append(vad_quantile)


out = pd.DataFrame.from_items(
  [('fname', fnames), ('vad', vads)]
)
out.to_csv('output/vads_85.csv', index = False)
