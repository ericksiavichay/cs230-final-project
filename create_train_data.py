import numpy as np
import glob
import pandas as pd
import pickle as pk

#converting the npy files to shape (N, C, T, V, M)
#creating pickle file for labels

index = 0
label_dict_train = {}
label_dict_test = {}
label_dict_train['sample_name'] = []
label_dict_train['label'] = []
label_dict_test['sample_name'] = []
label_dict_test['label'] = []
#number of samples to run
num_train_samples = 5
num_test_samples = 5
train_data = np.zeros((num_train_samples, 1, 1200, 360, 1))
test_data = np.zeros((num_test_samples, 1, 1200, 360, 1))
df = pd.read_csv('data/HCP_1200.csv')
training_data = True
testing_data = False
index2 = 0
for file in glob.glob('data/*left.npy'):
    if index == num_train_samples:
        training_data = False
        testing_data = True
    if index2 == num_test_samples:
        break
    filename = file.split("_")
    left = np.load(file)
    row = df[df['subject'] == int(filename[0][-6:])]
    if left.shape[1] != 1200:
        print("{} does not have 1200 timesteps".format(filename[0][-6:]))
    if row.empty:
        print("{} is not in HCP_1200.csv".format(filename[0][-6:]))
    if training_data:
        if left.shape[1] == 1200 and not row.empty:
            label_dict_train['sample_name'].append(row.values[0][0])
            label_dict_train['label'].append(row.values[0][2])
            # if row.values[0][1] == 'M':
            #     label_dict_train['label'].append(0)
            # else:
            #     label_dict_train['label'].append(1)
            right = np.load('_'.join(filename[:-1]) + '_right.npy')
            d = np.concatenate((left, right), axis=0)
            print(test_data[index2][0].shape)
            train_data[index][0] = np.array([d]).T
            index += 1
    if testing_data:
        if left.shape[1] == 1200 and not row.empty:
            label_dict_test['sample_name'].append(row.values[0][0])
            label_dict_test['label'].append(row.values[0][2])
            # if row.values[0][1] == 'M':
            #     label_dict_test['label'].append(0)
            # else:
            #     label_dict_test['label'].append(1)
            right = np.load('_'.join(filename[:-1]) + '_right.npy')
            d = np.concatenate((left, right), axis=0)
            test_data[index2][0] = np.array([d]).T
            print(test_data[index2][0].shape)
            index2 += 1


assert(index == num_train_samples)
assert(index2 == num_test_samples)

print(test_data)

np.save(('train_data_age_%d.npy', num_train_samples), train_data)
np.save(('test_data_age_%d.npy', num_test_samples), test_data)
#np.save('sample_name.npy', label_dict['sample_name'])
#np.save('label.npy', label_dict['label'])
pickle_out = open(("train_label_%d.pkl", num_train_samples),"wb")
pk.dump(label_dict_train, pickle_out)
pickle_out.close()
pickle_out = open(("test_label_%d.pkl", num_train_samples),"wb")
pk.dump(label_dict_test, pickle_out)
pickle_out.close()