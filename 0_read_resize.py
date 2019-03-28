# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:03:54 2019
this code is to read BC diagnostic images and resize them to (1024, 2048)
@author: moham
"""

im_path = '/work/deogun/alali/breast/data/diagnostic_images/png'
out_path = '/work/deogun/alali/breast/data/diagnostic_images/npy/'
#from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

images_list = os.listdir(im_path)
#print('list of images: ', images_list)

print('processing all of the data')
for count, f in enumerate(images_list):
	#if(count > len(images_list)/2):
	#	break
	print('count = ', count)
	im = plt.imread(os.path.join(im_path,f))
	lr_im = np.fliplr(im) #flip left-right
	ud_im = np.flipud(im) #flip up-down
	ud_lr_im = np.fliplr(ud_im) #flip up-down, then left-right
	print('shape of image {} is {}'.format(f, im.shape))
	print('shape of flipped lr image {} is {}'.format(f, lr_im.shape))
	print('shape of flipped ud image {} is {}'.format(f, ud_im.shape))
	tf.reset_default_graph()
	with tf.Session() as sess:
		# Normalization
		norm_im = tf.image.per_image_standardization(im)
		norm_lr_im = tf.image.per_image_standardization(lr_im)
		norm_ud_im = tf.image.per_image_standardization(ud_im)
		norm_ud_lr_im = tf.image.per_image_standardization(ud_lr_im)
		print('{} im normalized: {}'.format(count, norm_im))
		#resize every image and its copies
		resized_tf = tf.image.resize_image_with_crop_or_pad(norm_im, 2048, 4096)
		lr_tf = tf.image.resize_image_with_crop_or_pad(norm_lr_im, 2048, 4096)
		ud_tf = tf.image.resize_image_with_crop_or_pad(norm_ud_im, 2048, 4096)
		ud_lr_tf = tf.image.resize_image_with_crop_or_pad(norm_ud_lr_im, 2048, 4096)
		#print('resized tensor: ', resized_tf.shape)
		resized_img = sess.run(resized_tf)
		lr_np = sess.run(lr_tf)
		ud_np = sess.run(ud_tf)
		ud_lr_np = sess.run(ud_lr_tf)
		print('resized image type ={} size ={}: '.format(type(resized_img),resized_img.shape))
		print('resized image lr type ={} size ={}: '.format(type(lr_np),lr_np.shape))
		print('resized image ud type ={} size ={}: '.format(type(ud_np),ud_np.shape))
		if(count % 10 == 0):
			plt.imshow(lr_np)
			plt.savefig('out/resized{}.png'.format(count))
			plt.close()
		resized_img.dump(out_path+f[:-4]+'.npy')
		lr_np.dump(out_path+f[:-4]+'_lr.npy')
		ud_np.dump(out_path+f[:-4]+'_ud.npy')
		ud_lr_np.dump(out_path+f[:-4]+'_ud_lr.npy')
		
print('done')
