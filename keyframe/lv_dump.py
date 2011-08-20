import cPickle as pickle
import matplotlib.pyplot as mp
import numpy as np


with open('lv.pkl') as fp:
    label_values = pickle.load(fp)

values = {}
for l, v in label_values:
    values.setdefault(l, []).append(v)
dims = len(v)
for l in values:
    values[l] = np.array(values[l])

for d in range(dims):
    mp.clf()
    mp.hist([values[x][:, d] for x in values], 100, color=['red', 'blue'])
    mp.savefig('hist-%d.png' % d)
