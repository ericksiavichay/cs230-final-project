import numpy as np, glob
import pandas as pd, pickle as pk
from sklearn.model_selection import train_test_split

#converting the npy files to shape (N, C, T, V, M)
#creating pickle file for labels

MIN_AGE = 22
TIMELENGTH = 1200
BRAIN_REGION = 360


print('create X and Y dataset')
X, Y = [], []
df = pd.read_csv('data/HCP_1200.csv')

for file in glob.glob('data/*left.npy'):
    if (index > 5): break
    filename = file.split("_")
    left = np.load(file)

    row = df[df['subject'] == int(filename[0][-6:])]

    if row.empty:
        print("{} is not in HCP_1200.csv".format(filename[0][-6:]))
    else:
        Y.append(int(row.values[0][2])-MIN_AGE)

        right = np.load('_'.join(filename[:-1]) + '_right.npy')
        d = np.concatenate((left, right), axis=0)
        X.append([np.array([d]).T])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
label_dict_train['sample_name'] = Y_train
label_dict_test['sample_name'] = Y_test

print('save dataset')
np.save('train_data_age.npy', X_train)
np.save('test_data_age.npy'. X_test)
#np.save('sample_name.npy', label_dict['sample_name'])
#np.save('label.npy', label_dict['label'])
pickle_out = open("train_label_age.pkl","wb")
pk.dump(label_dict_train, pickle_out)
pickle_out.close()
pickle_out = open("test_label_age.pkl","wb")
pk.dump(label_dict_test, pickle_out)
pickle_out.close()