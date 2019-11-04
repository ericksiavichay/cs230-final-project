from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, UpSampling3D, Input, ZeroPadding3D, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dropout
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.constraints import unit_norm, max_norm
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam

import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import sys
import argparse
import os
import glob 
import os.path
from scipy.ndimage import zoom
import time

import dcor
#from distcorr import distcorr

# In[2]:
def augment_by_transformation2(data,age,sex,n):
    print("(just for test) Done!\n");
    return data,age,sex


def augment_by_transformation(data,n):
    augment_scale = 1

    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        new_data = np.zeros((m,data.shape[1],data.shape[2],data.shape[3],1))
        for i in range(0,m):
            idx = np.random.randint(0,raw_n)
            new_data[i] = data[idx].copy()
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,0),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(0,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.shift(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5))

        # output an example
        array_img = nib.Nifti1Image(np.squeeze(new_data[3,:,:,:,0]),np.diag([1, 1, 1, 1]))  
        filename = 'augmented_example.nii.gz'
        nib.save(array_img,filename)

        data = np.concatenate((data, new_data), axis=0)
        return data

def inv_correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

class GAN():
    
    def __init__(self):

        optimizer = Adam(0.0002)
	optimizer_distiller = Adam(0.0001)
	optimizer_regressor = Adam(0.0001)

        L2_reg = 0.1
	ft_bank_baseline = 16
	latent_dim = 16

        # Build and compile the cf predictorinv_inv
        self.regressor = self.build_regressor()
        self.regressor.compile(loss='mse', optimizer=optimizer)

        # Build the feature encoder
        input_image = Input(shape=(patch_x/2,patch_y,patch_z,1), name='input_image')
        feature = Conv3D(ft_bank_baseline, activation='relu', kernel_size=(3, 3, 3),padding='same')(input_image)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline*2, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline*4, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature_dense = Flatten()(feature)
        
        self.encoder = Model(input_image, feature_dense)

        # For the distillation model we will only train the encoder

        self.regressor.trainable = False
        cf = self.regressor(feature_dense)
        self.distiller = Model(input_image, cf)
        self.distiller.compile(loss=correlation_coefficient_loss, optimizer=optimizer)

        # Build and Compile the classifer  
        #self.encoder.load_weights('encoder.h5');
        #self.encoder.trainable = False
        input_feature_clf = Input(shape=(512,), name='input_feature_dense')
        feature_clf = Dense(latent_dim*4, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(input_feature_clf)
        feature_clf = Dense(latent_dim*2, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
        prediction_score = Dense(1, name='prediction_score',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
        self.classifier = Model(input_feature_clf, prediction_score)

        # Build the entir workflow
        prediction_score_workflow = self.classifier(feature_dense)
        label_workflow = Activation('sigmoid', name='r_mean')(prediction_score_workflow)
        self.workflow = Model(input_image, label_workflow)
        self.workflow.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    def build_regressor(self):
	latent_dim = 16
        inputs_x = Input(shape=(512,))
	feature = Dense(latent_dim*4, activation='tanh')(inputs_x)
    	feature = Dense(latent_dim*2, activation='tanh')(feature)
        cf = Dense(1)(feature)

        return Model(inputs_x, cf)

    def train(self, epochs, training, testing, testing_raw, batch_size=64, fold=0):
        [train_data_aug, train_dx_aug] = training
	[test_data_aug,  test_dx_aug]  = testing
	[test_data    ,  test_dx    ]   = testing_raw
        
        test_data_aug_flip = np.flip(test_data_aug,1)
        test_data_flip = np.flip(test_data,1)

        idx_perm = np.random.permutation(int(train_data_aug.shape[0]/2))
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Disstiller
            # ---------------------

            # Select a random batch of images
            idx_perm = np.random.permutation(int(train_data_aug.shape[0]/2))
            idx = idx_perm[:int(batch_size/2)]
	    idx = np.concatenate((idx,idx+int(train_data_aug.shape[0]/2)))

            training_feature_batch = train_data_aug[idx]
            dx_batch = train_dx_aug[idx]

            c_loss = self.workflow.train_on_batch(training_feature_batch[:,:32,:,:], dx_batch)
            training_feature_batch = np.flip(training_feature_batch,1)
            c_loss = self.workflow.train_on_batch(training_feature_batch[:,:32,:,:], dx_batch)

            # Plot the progress
            if epoch % 10 == 0:
                c_loss_test_1 = self.workflow.evaluate(test_data_aug[:,:32,:,:],      test_dx_aug, verbose = 0, batch_size = batch_size)    
                c_loss_test_2 = self.workflow.evaluate(test_data_aug_flip[:,:32,:,:], test_dx_aug, verbose = 0, batch_size = batch_size)    

		# feature dist corr
	        features_dense = self.encoder.predict(train_data_aug[train_dx_aug == 0,:32,:,:],  batch_size = batch_size)
                print ("%d [Acc: %f,  Test Acc: %f %f]" % (epoch, c_loss[1], c_loss_test_1[1], c_loss_test_2[1]))
		sys.stdout.flush()

                self.classifier.save_weights("res/classifier.h5")
                self.encoder.save_weights("res/encoder.h5")
                self.workflow.save_weights("res/workflow.h5")

                prediction = self.workflow.predict(test_data[:,:32,:,:],  batch_size = 64)
                filename = 'res/prediction_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,prediction)
                prediction = self.workflow.predict(test_data_flip[:,:32,:,:],  batch_size = 64)
                filename = 'res/prediction_flip_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,prediction)

                filename = 'res/dx_'+str(fold)+'.txt'
                np.savetxt(filename,test_dx)    


if __name__ == '__main__':
    start_time = time.time()

    ## summarize data
    file_idx = np.genfromtxt('./info/subjects.txt', dtype='str') #np.loadtxt('./subject_ids.txt')  
    sex = np.loadtxt('./info/sex.txt') 
    age = np.loadtxt('./info/age.txt') 
    dx = sex;

    exist = np.ones(len(file_idx), dtype=bool)
    cnt = 0
    for subject_idx in file_idx:
        filename_full = '/fs/neurosci01/qingyuz/hcp/hcp_900_structural/img_64/'+subject_idx+'.nii.gz'
	if (not os.path.isfile(filename_full)):
		exist[cnt] = 0
	cnt += 1

    print("Total = %d" % np.count_nonzero(exist));
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    sys.stdout.flush()

    file_idx = file_idx[exist];
    age = age[exist];
    dx  = dx[exist];
    print("Total female = %d" % sum(sex))

    ## loading data
    np.random.seed(seed=0)

    subject_num = file_idx.shape[0]
    patch_x = 64
    patch_y = 64
    patch_z = 64
    min_x = 0
    min_y = 0
    min_z = 0

    augment_size = 1024
    data = np.zeros((subject_num, patch_x, patch_y, patch_z,1))
    i = 0
    for subject_idx in file_idx:
        filename_full = '/fs/neurosci01/qingyuz/hcp/hcp_900_structural/img_64/'+subject_idx+'.nii.gz'

        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i,:,:,:,0] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data[i,:,:,:,0] = (data[i,:,:,:,0] - np.mean(data[i,:,:,:,0])) / np.std(data[i,:,:,:,0])

        # output an example
        array_img = nib.Nifti1Image(np.squeeze(data[i,:,:,:,0]),np.diag([1, 1, 1, 1]))  
        filename = 'processed_example.nii.gz'
        nib.save(array_img,filename)

        i += 1

    print ("Loaded %d data images." % i)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    sys.stdout.flush()

    ## cross-validation
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    pred = np.zeros((dx.shape))

    fold = 1
    for train_idx, test_idx in skf.split(data, dx):  
        print('CV fold %d' % fold)
        sys.stdout.flush()

        train_data = data[train_idx]
        train_dx = sex[train_idx]
        train_subid = file_idx[train_idx]

        test_data = data[test_idx]
        test_dx = sex[test_idx]
        test_subid = file_idx[test_idx]

        print(test_subid[0])
        sys.stdout.flush()
        print("Augmenting data ...")
        sys.stdout.flush()

        # augment data
        train_data_pos = train_data[train_dx==1];
        train_data_neg = train_data[train_dx==0];
        del train_data

        
        train_data_pos_aug = augment_by_transformation(train_data_pos,augment_size)
        del train_data_pos
        train_data_neg_aug = augment_by_transformation(train_data_neg,augment_size)
        del train_data_neg
        train_data_aug = np.concatenate((train_data_neg_aug, train_data_pos_aug), axis=0)
        del train_data_pos_aug
        del train_data_neg_aug
        
        train_dx_aug = np.zeros((augment_size * 2,))
        train_dx_aug[augment_size:] = 1

        test_data_pos = test_data[test_dx==1];
        test_data_neg = test_data[test_dx==0];

        test_data_pos_aug = augment_by_transformation(test_data_pos,500)
        del test_data_pos
        test_data_neg_aug = augment_by_transformation(test_data_neg,500)
        del test_data_neg
        test_data_aug = np.concatenate((test_data_neg_aug, test_data_pos_aug), axis=0)
        del test_data_pos_aug
        del test_data_neg_aug

        test_dx_aug = np.zeros((1000,))
        test_dx_aug[500:] = 1

        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        print("Begin Training on fold")
        sys.stdout.flush()

        gan = GAN()
        gan.train(epochs=1001, training=[train_data_aug, train_dx_aug], testing=[test_data_aug, test_dx_aug], testing_raw=[test_data, test_dx],  batch_size=96, fold=fold)

        del train_data_aug
        del test_data_aug
        del test_data

        print("Training finished on fold")
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        fold = fold + 1
