'''Script for Calculating Adjacency / Diagonal matrix
'''

import numpy as np
import glob

print("import hcp input data")
data = None
for file in glob.glob('../data/*.npy'):
    filename = file.split("_")
    d = np.load(file)
    if d.shape[1] == 1200:
        if data is None:
            data = d
        else:
            data = np.concatenate((d, data), axis=1)
print('sanity check: current data has shape of ' + str(data.shape))

print('compute triangular matrix')
n_regions = len(d)
A = np.zeros((1, n_regions, n_regions))
for i in range(n_regions):
    for j in range(i, n_regions):
        if i==j:
            A[0][i][j] = 1
        else:
            A[0][i][j] = abs(np.corrcoef(data[i,:], data[j,:])[0][1]) # get value from corrcoef matrix
            A[0][j][i] = A[0][i][j]

print('compute hat adjacency matrix')
A_hat = np.zeros((1, n_regions, n_regions))
A_hat[0] = A[0] + np.identity(n_regions)

print('compute diagnoal hat')
D = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    D[i][i] = np.sum(A[0][i])

print('export adjacency matrix')
np.save('adj_matrix.npy', A)
np.save('adj_hat.npy', A_hat)
np.save('d_hat.npy', D)