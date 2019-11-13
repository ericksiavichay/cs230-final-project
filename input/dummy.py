import numpy as np, math, random
import pandas as pd, pickle as pk
from sklearn.model_selection import train_test_split

X, Y, sample_names = [],[],[]
label_dict_train, label_dict_test = {}, {}

AMP = 1000
SIZE = 1000

for t in range(SIZE):
	g1 = []
	g2 = []

	for i in range(100):
		g1_m = [[np.random.rand()] for i in range(5)]
		g1_m[0][0] += AMP*math.sin(i)
		g1_m[1][0] += AMP*math.sin(i)
		
		g2_m = [[np.random.rand()] for i in range(5)]
		g2_m[2][0] += AMP*math.sin(i)
		g2_m[3][0] += AMP*math.sin(i)
		g2_m[4][0] += AMP*math.sin(i)

		g1.append(g1_m)
		g2.append(g2_m)

	s = np.array(g1)
	
	sample_names += ['{}_1'.format(t), '{}_2'.format(t)]
	X.append([np.array(g1)])
	X.append([np.array(g2)])
	Y += [0, 1]

X = np.array(X)


X_train, X_test, Y_train, Y_test, test_samples, train_samples = train_test_split(X, Y, sample_names, test_size=0.2, random_state=0)
label_dict_train['label'] = Y_train
label_dict_train['sample_name'] = train_samples
label_dict_test['label'] = Y_test
label_dict_test['sample_name'] = test_samples

print('save dataset')
np.save('dummy_train_data.npy', X_train)
np.save('dummy_test_data.npy', X_test)
pickle_out = open("dummy_train_label.pkl","wb")
pk.dump(label_dict_train, pickle_out)
pickle_out.close()
pickle_out = open("dummy_test_label.pkl","wb")
pk.dump(label_dict_test, pickle_out)
pickle_out.close()