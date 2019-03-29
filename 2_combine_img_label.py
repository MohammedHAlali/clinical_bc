'''
modified from: https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data.git

This code reads images (.npy) and their corresponding labels from (clean_features.csv)
to prepare breast cancer data of diagnostic images and multi-labels

@author: Mohammed H. Alali
March, 2019
'''

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv

#from skimage import io
from sklearn.utils import shuffle
#from keras.preprocessing import image
#from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
#from keras.applications.imagenet_utils import preprocess_input

# Define data path
data_path = '../data/diagnostic_images/npy'
label_file = 'clean_features.csv'
data_dir_list = os.listdir(data_path)
print('number of images: ', len(data_dir_list))

all_binary_features = []
all_feature_values = []
features_names = []
labels_dict = {} # a dictionary to store each case id as key with all case's features as value
with open(label_file, 'r') as csvfile:
	csv_reader = csv.reader(csvfile)
	feature_names = next(csv_reader)
	#print('feature names count: ', len(feature_names))
	for i, row in enumerate(csv_reader):
		labels_dict[row[0]] = row[1:]
		print('features count: for case {} is {}'.format(row[0], len(labels_dict[row[0]])))

print('total number of labels: ', len(labels_dict))
print('features for case TCGA-A2-A0CK is: ', labels_dict['TCGA-A2-A0CK'])


img_data_list=[]
label_data_list=[]

for i, f in enumerate(data_dir_list):
	print('{} -reading img of patient id: {}'.format(i, f[:12]))
	#im = np.load(os.path.join(data_path, f))
	#print('image {} shape {}'.format(f, im.shape))
	img_data_list.append(np.load(os.path.join(data_path, f)))
	#get label
	label = labels_dict[f[:12]]
	label_data_list.append(label)

print('image data list shapes')
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#normalization
print ('final: ', img_data.shape)

print('label data list shapes')
label_data = np.array(label_data_list).astype(np.float16)
#img_data = img_data.astype('float32')
print (label_data.shape)

#Shuffle the dataset
x,y = shuffle(img_data,label_data)

print('x (images) shape: ', x.shape)
print('y (labels) shape: ', y.shape)

np.savez_compressed('data_images', x)
np.savez_compressed('data_labels', y)
np.savetxt('data_labels.csv', y, delimiter=',')

print('done')
