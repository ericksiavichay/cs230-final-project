'''Script for Calculating Adjacency / Diagonal matrix
'''

import numpy as np
import glob

print("import hcp input data")
data = None
for file in glob.glob('data/*left.npy'):
	filename = file.split("_")
	left = np.load(file)
	right = np.load('_'.join(filename[:-1]) + '_right.npy')
	d = np.concatenate((left, right), axis=0)
	if data is None:
		data = d
	else:
		data = np.concatenate((d, data), axis=1)
print('sanity check: current data has shape of ' + str(data.shape))

print('compute triangular matrix')
n_regions = len(d)
A = np.zeros((n_regions, n_regions))
for i in range(n_regions):
	for j in range(i, n_regions):
		if i==j: 
			A[i][j] = 1
		else:
			A[i][j] = abs(np.corrcoef(data[i,:], data[j,:])[0][1]) # get value from corrcoef matrix
			A[j][i] = A[i][j]

print('compute hat adjacency matrix')
A_hat = A + np.identity(n_regions)

print('compute diagnoal hat')
D = np.zeros((n_regions, n_regions))
for i in range(n_regions):
	D[i][i] = np.sum(A[i])

print('export adjacency matrix')
np.save('adj_matrix.npy', A) 
np.save('adj_hat.npy', A_hat)
np.save('d_hat.npy', D)