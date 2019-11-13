import numpy as np

n_reg = 5
A = np.zeros((n_reg, n_reg))

for i in range(2):
	A[i] = np.array([1,1,0,0,0])
for j in range(2,5):
	A[j] = np.array([0,0,1,1,1])

print('export adjacency matrix')
np.save('adj/dummy_adj.npy', A) 