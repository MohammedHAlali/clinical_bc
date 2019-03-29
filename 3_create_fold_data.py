"""
This code is to read full data file, divides it into folds of training and testing data. 
Because of memory limitation, we couldn't include this in the actual training session, which would be better.
The output will be 5 files of training images data (images, labels), and their corresponding 5 files for testing (images, labels).
So, in total 4x5=20 files will be created after running this code.
@author: Mohammed H. Alali
March, 2019
"""

from sklearn.model_selection import KFold
import numpy as np



data_images = np.load('data_images.npz')['arr_0']
data_labels = np.load('data_labels.npz')['arr_0']

print('shapes: ', data_images.shape, '\n', data_labels.shape)

#set up k-fold cross-valiation
kfold = KFold(n_splits=5, shuffle=True)
fold = 0
for train_index, test_index in kfold.split(X=data_images, y=data_labels):
	print('fold = ', fold)
	fold = fold + 1
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = data_images[train_index], data_images[test_index]
	y_train, y_test = data_labels[train_index], data_labels[test_index]
	print('shapes: ', X_train.shape, '\n', X_test.shape, '\n', y_train.shape, '\n', y_test.shape)
	np.savez_compressed('train_images_fold{}'.format(fold), X_train)
	np.savez_compressed('train_labels_fold{}'.format(fold), y_train)
	np.savez_compressed('test_images_fold{}'.format(fold), X_test)
	np.savez_compressed('test_labels_fold{}'.format(fold), y_test)

print('done all')
