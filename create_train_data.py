import numpy as np
import glob
import pandas as pd
import pickle as pk

#converting the npy files to shape (N, C, T, V, M)
#creating pickle file for labels

index = 0
label_dict = {}
label_dict['sample_name'] = []
label_dict['label'] = []
#number of samples to run
num_samples = 50
train_data = np.zeros((num_samples, 1, 1200, 360, 1))
df = pd.read_csv('data/HCP_1200.csv')
for file in glob.glob('data/hcp_tc_npy/*left.npy'):
    if index == num_samples:
        break
    filename = file.split("_")
    left = np.load(file)
    row = df[df['subject'] == int(filename[2][-6:])]
    if left.shape[1] != 1200:
        print("{} does not have 1200 timesteps".format(filename[2][-6:]))
    if row.empty:
        print("{} is not in HCP_1200.csv".format(filename[2][-6:]))
    if left.shape[1] == 1200 and not row.empty:
        label_dict['sample_name'].append(row.values[0][0])
        if row.values[0][1] == 'M':
            label_dict['label'].append(0)
        else:
            label_dict['label'].append(1)
        right = np.load('_'.join(filename[:-1]) + '_right.npy')
        d = np.concatenate((left, right), axis=0)
        train_data[index][0] = np.array([d]).T
        index += 1
assert(index == num_samples)

np.save('train_data.npy', train_data)
np.save('sample_name.npy', label_dict['sample_name'])
np.save('label.npy', label_dict['label'])
pickle_out = open("train_label.pkl","wb")
pk.dump(label_dict, pickle_out)
pickle_out.close()