'''Script for Calculating Adjacency / Diagonal matrix
for half of hemisphere
'''

import glob
import numpy as np


print("import hcp input data for half")
data = None
for file in glob.glob('data/*left.npy'):
	filename = file.split("_")
	left = np.load(file)
	right = np.load('_'.join(filename[:-1]) + '_right.npy')
	d = left + right/2.
	if data is None:
		data = d
	else:
		data = np.concatenate((d, data), axis=1)
print('sanity check: current data has shape of ' + str(data.shape)) # shape = (180, 1314326)

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
np.save('adj/adj_h_matrix.npy', A) 
np.save('adj/adj_h_hat.npy', A_hat)
np.save('adj/d_h_hat.npy', D)
   
