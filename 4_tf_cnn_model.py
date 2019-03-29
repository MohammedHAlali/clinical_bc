'''
design a TF cnn model to work on BC images and labels
'''

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#tf.device('/cpu:0')

out_dir = 'out/'
tf.reset_default_graph()

#default parameters
exp_num = 1
learning_rate = 0.1
epochs = 1
batch_size = 1
num_classes = 49
fold = 1
#tf Graph Input
X = tf.placeholder(tf.float32, [None, 2048, 4096, 4], name='input_placeholder')
Y = tf.placeholder(tf.float32, [None, num_classes], name='output_placeholder')
train_mode = tf.placeholder(tf.bool)
if(len(sys.argv) > 1):
	for i in range(1, len(sys.argv)):
		if(sys.argv[i] == 'exp'):
			exp_num = sys.argv[i+1]
			print('experiment number: ', exp_num)
		elif(sys.argv[i] == 'lr'):
			learning_rate = float(sys.argv[i+1])
			print('learning rate: ', learning_rate)
		elif(sys.argv[i] == 'epo'):
			epochs = int(sys.argv[i+1])
			print('epochs: ', epochs)
		elif(sys.argv[i] == 'batch'):
			batch_size = int(sys.argv[i+1])
			print('batch size: ', batch_size)
		elif(sys.argv[i] == 'fold'):
			fold = int(sys.argv[i+1])
			print('fold = ', fold)
else:
	raise ValueError('ERROR: please pass parameters as: exp 1 lr 0.1 epo 1 batch 10. You passed: ', sys.argv)



X_train = np.load('data/train_images_fold{}.npz'.format(fold))['arr_0']
y_train = np.load('data/train_labels_fold{}.npz'.format(fold))['arr_0']
X_test = np.load('data/test_images_fold{}.npz'.format(fold))['arr_0']
y_test = np.load('data/test_labels_fold{}.npz'.format(fold))['arr_0']
y_train = y_train.astype(np.float16)
y_test = y_test.astype(np.float16)

print('shapes: ', X_train.shape, '\n', X_test.shape, '\n', y_train.shape, '\n', y_test.shape)

# for checking only
print('reading X_train[0-5]')
#save 5 images for testing and debugging
for k in range(5):
        plt.imshow(X_train[k])
        plt.savefig(out_dir+'x_train{}.png'.format(k))

def cnn_model(X):
	print('input x: ', X)
	#k_init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)

	conv1 = tf.layers.conv2d(X,  filters=2,
                                     kernel_size=[5,5], 
                                     strides=2, 
                                     padding='same', 
                                     activation=tf.nn.relu,
			             name='conv1')
	print(conv1)
	conv2 = tf.layers.conv2d(conv1, 
                                 filters=4,
                                 kernel_size=[5,5], 
                                 strides=2, 
                                 padding='same', 
                                 activation=tf.nn.relu,
                                 name='conv2')
	print(conv2)

	conv3 = tf.layers.conv2d(conv2,
                                 filters=8,
                                 kernel_size=[5,5], 
                                 strides=2,
                                 padding='same', 
                                 activation=tf.nn.relu,
                                 name='conv3')
	print(conv3)
	conv4 = tf.layers.conv2d(conv3,
                                 filters=16,
                                 kernel_size=[3,3],
                                 strides=3,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv4')
	print(conv4)
	conv5 = tf.layers.conv2d(conv4,
                                 filters=32,
                                 kernel_size=[3,3],
                                 strides=3,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv5')
	print(conv5)
	
	conv6 = tf.layers.conv2d(conv5,
                                 filters=64,
                                 kernel_size=[1,1],
                                 strides=3,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv6')
	print(conv6)
	
	flatten_dim = np.prod(conv6.get_shape().as_list()[1:])
	print('flatten dimension: ', flatten_dim)
	flat = tf.reshape(conv6, [-1, flatten_dim])
	print('flat layer: ', flat)
	
	dropout = tf.layers.dropout(flat, train_mode) #dropout 50% of the size
	print('dropout: ', dropout)
	dense1 = tf.layers.dense(dropout, 5000, activation=tf.nn.relu, name='dense1') 
	print(dense1)
	#print(dense1.get_shape().as_list()[1])
	#print('type: ', type(dense1.get_shape().as_list()[1]))
	dense2 = tf.layers.dense(dense1, 1000, tf.nn.relu, name='dense2')
	print(dense2)
	dense3 = tf.layers.dense(dense2, 200, tf.nn.relu, name='dense3')
	print(dense3)
	#dense4 = tf.layers.dense(dense3, dense3.get_shape().as_list()[1]//5, tf.nn.relu, name='dense4')
	#print(dense4)
	logits = tf.layers.dense(dense3, num_classes, name='logits')
	print(logits)
	return logits	

# Construct model
logits = cnn_model(X)
output = tf.nn.sigmoid(logits)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.square(Y - output), axis=0)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)
print('optimizer: ', optimizer)

#evaluate model
corrected_pred = tf.equal(output, Y) 
accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float16), axis=0)

# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()
#local_init = tf.local_variables_initializer()


# Start training
with tf.Session() as sess:
	#print('session started: ', sess)
    
	# Run the initializer
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	#print('init run')
	all_epochs_loss_per_feature = [] #stores the loss value of each feature (length = 49), averaged per epoch
	all_epochs_acc_per_feature = []
	all_epoch_loss_one_value = [] #stores the loss values of all features (length = 1) averaged per epoch
	# starting Training
	for e in range(1, epochs+1):
		#sess.run(local_init)
		print('============ epoch : ', e)
		epoch_loss = []
		epoch_acc = []
		iterations = len(X_train)//batch_size
		for i in range(iterations):
			#print('iteration: ', i)
			batch_x = X_train[(i*batch_size):(i*batch_size)+batch_size]
			batch_y = y_train[(i*batch_size):(i*batch_size)+batch_size]
			#print('batch_y: ', batch_y)
			#print('batch x shape: ', batch_x.shape)
			#print('batch y shape: ', batch_y.shape)
			myloss, myacc, _ = sess.run([loss_op, accuracy, train_op], 
							feed_dict={X: batch_x, Y: batch_y, train_mode:True})
			epoch_loss.append(myloss)
			#print('mymid_layer: ', mymid_layer)
			#myacc = np.equal(batch_y, myout)
			#myacc = np.squeeze(myacc.astype(np.float16))
			epoch_acc.append(myacc)			
		#results for every epoch
		avg_epoch_acc = np.mean(epoch_acc, axis=0)
		avg_epoch_loss = np.mean(epoch_loss, axis=0)
		avg_epoch_loss_total = np.mean(epoch_loss)
		print("epoch {} average loss = {} loss per feature= ".format(e, avg_epoch_loss_total) + \
                 		"{}".format(avg_epoch_loss) + \
                  		", Training Accuracy= " + \
                  		"{}".format(avg_epoch_acc))
        	
		#For every epoch
		all_epochs_loss_per_feature.append(avg_epoch_loss)
		all_epochs_acc_per_feature.append(avg_epoch_acc)
		all_epoch_loss_one_value.append(avg_epoch_loss_total)
		print('all epoch loss: ', all_epoch_loss_one_value)
		print('length of all_epoch_loss_one_value: ', len(all_epoch_loss_one_value))
	#after all epochs are done
	#visualize all epochs loss, in one value (len = 1)
	x = np.arange(1, epochs+1)
	y = all_epoch_loss_one_value
	print('x = num of epochs: ', x)
	print('y= average feature epoch loss: ', y)
	plt.plot(x, y)
	plt.ylabel('mean training loss for all features')
	plt.xlabel('number of epochs')
	plt.title('Training loss')
	plt.show()
	plt.savefig(out_dir+'exp{}_training_mean_loss_fold{}.png'.format(exp_num, fold))
	plt.close()
	#visualizing all epochs loss for every feature, (len = 49)
	x = np.arange(1, num_classes+1)
	y = np.mean(all_epochs_loss_per_feature, axis=0)
	#print('x=num classes={}, shpape={}'.format(x, x.shape))
	#print('y=all epoch_loss={}, shape={}'.format(y, y.shape))
	plt.bar(x, y)
	plt.ylabel('training loss per feature')
	plt.xlabel('feature index')
	plt.title('Training Results for all epochs')
	plt.show()
	plt.savefig(out_dir+'exp{}_training_loss_fold{}.png'.format(exp_num, fold))
	plt.close()

	# Visualize Accuracy for all training epochs
   	#x = np.arange(1, num_classes+1)
	y2 = np.mean(all_epochs_acc_per_feature, axis=0)
	np.savetxt(out_dir+'exp{}_train_accuracy_fold{}.csv'.format(exp_num, fold), y2, delimiter=',')
	print('x=num classes={}, shape={}'.format(x, x.shape))
	print('y=all epoch accuracy={}, shape={}'.format(y2, y2.shape))
	plt.bar(x, y2)
	plt.ylabel('training accuracy')
	plt.xlabel('feature index')
	plt.title('Training Accuracy for all epochs')
	plt.show()
	plt.savefig(out_dir+'exp{}_training_acc_fold{}.png'.format(exp_num, fold))
	plt.close()
	print("Training Finished!")

    	# Calculate test accuracy
	test_acc_list = []
	print('==========Testing==========')
	for i in range(len(X_test)):
		test_x = np.expand_dims(X_test[i], axis=0)
		test_y = np.expand_dims(y_test[i], axis=0)
		# for checking only
		#print('reading  X_test[10, 20]')
		#save 5 images for testing and debugging
		if i % 10 == 0:
        		plt.imshow(X_test[i])
        		plt.savefig(out_dir+'x_test{}.png'.format(i))
		#try to insert images only, without labels
		pred, myacc= sess.run([output, accuracy], feed_dict={X: test_x, Y: test_y, train_mode:False})
		#print('testing iteration {}, myacc = {}'.format(i, myacc))
		#print('tf accuracy: ', tfaccu)
		#print('testing iteration {}, myacc = {}'.format(i, myacc))
		test_acc_list.append(myacc)
		#print('actual test y: ', test_y)
		#print('predicted test_label {} = {} '.format(i, pred))

	x = np.arange(num_classes)
	y = np.squeeze(np.mean(test_acc_list, axis=0))
	np.savetxt(out_dir+'exp{}_test_accuracy_fold{}.csv'.format(exp_num, fold), y, delimiter=',')
	print('y=test accuracy list={}, shape={}'.format(y, y.shape))
	plt.bar(x, y)
	plt.ylabel('accuracy')
	plt.xlabel('feature index')
	plt.title('Testing Accuracy Results')
	plt.show()
	plt.savefig(out_dir+'exp{}_testing_accuracy_fold{}.png'.format(exp_num, fold))
	plt.close()

print('$$$$$$ done everything $$$$$$')
