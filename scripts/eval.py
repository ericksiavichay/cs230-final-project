'''Script for Calculating Evaluation Metrics
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FILENAME = 'bin_age_test_result.npy'

# load the file 
f = np.load(FILENAME, allow_pickle=True)
fmap = f.item()
if 'R2' in fmap:
	del fmap['R2']

# extract true and pred values
true, pred = [], []
for sn in fmap:
	pred.append(fmap[sn][0])
	true.append(fmap[sn][1])

# calculate age pred
precision, recall, fb, support = precision_recall_fscore_support(true, pred, pos_label=0, average='binary')

print('Evaluation summary for ' + FILENAME)
print('Precision: {0:.4f}'.format(precision))
print('Recall: {0:.4f}'.format(recall))
print('F1 score: {0:.4f}'.format(fb))