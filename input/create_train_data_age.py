import numpy as np, glob
import pandas as pd, pickle as pk
from sklearn.model_selection import train_test_split

#converting the npy files to shape (N, C, T, V, M)
#creating pickle file for labels

MIN_AGE = 22
TIMELENGTH = 1200
TIMESLICE = 100
BRAIN_REGION = 22
AGES = [22, 35]


print('create X and Y dataset')
X, Y, sample_names = [], [], []
label_dict_train, label_dict_test = {}, {}

df = pd.read_csv('data/HCP_1200.csv')

for file in glob.glob('data/*.npy'):
	filename = file.split("_")
	d = np.load(file)

	row = df[df['subject'] == int(filename[0][-6:])]

	if row.empty:
		print("{} is not in HCP_1200.csv".format(filename[0][-6:]))

	elif d.shape[1] > TIMESLICE:
		for i in range(int(d.shape[1]/TIMESLICE)):
			start = i*TIMESLICE
			end = (i+1)*TIMESLICE
			if len(d[:,start:end][0]) == TIMESLICE:
				sample_names.append(int(row.values[0][0]))
				Y.append(int(int(row.values[0][2])-MIN_AGE))
				X.append([np.array([d[:,start:end]]).T])
				


X = np.array(X)

X_train, X_test, Y_train, Y_test, test_samples, train_samples = train_test_split(X, Y, sample_names, test_size=0.2, random_state=0)
label_dict_train['label'] = Y_train
label_dict_train['sample_name'] = train_samples
label_dict_test['label'] = Y_test
label_dict_test['sample_name'] = test_samples


print('save dataset')
np.save('train_data.npy', X_train)
np.save('test_data.npy', X_test)
pickle_out = open("train_label.pkl","wb")
pk.dump(label_dict_train, pickle_out)
pickle_out.close()
pickle_out = open("test_label.pkl","wb")
pk.dump(label_dict_test, pickle_out)
pickle_out.close()