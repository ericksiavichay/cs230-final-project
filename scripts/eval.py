'''Script for Calculating Evaluation Metrics
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns

FILENAME = '..\\test_result_sex_no_aug.npy'

# load the file 
f = np.load(FILENAME, allow_pickle=True)
fmap = f.item()
if 'R2' in fmap:
	del fmap['R2']

class_names = ['male', 'female']
# extract true and pred values
true, pred = [], []
for sn in fmap:
	pred.append(fmap[sn][0])
	true.append(fmap[sn][1])
print(sum(true))

# calculate age pred
precision, recall, fb, support = precision_recall_fscore_support(true, pred, pos_label=1, average='binary')
cm=confusion_matrix(true, pred, labels=[0, 1])
print(cm.ravel())
print('Evaluation summary for ' + FILENAME)
print('Precision: {0:.4f}'.format(precision))
print('Recall: {0:.4f}'.format(recall))
print('F1 score: {0:.4f}'.format(fb))
print(cm)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['male', 'female']); ax.yaxis.set_ticklabels(['male', 'female']);
plt.show()