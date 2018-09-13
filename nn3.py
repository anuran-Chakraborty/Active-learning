import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

train_x=None
train_y=None
test_x=None
test_y=None
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_classes = 10
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float',[None, 10])
y = tf.placeholder('float')

prediction=None



# Nothing changes
def neural_network_model(data):

	
	hidden_1_layer = {'f_fum':n_nodes_hl1,
	                  'weight':tf.Variable(tf.random_normal([10, n_nodes_hl1])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'f_fum':n_nodes_hl2,
	                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'f_fum':n_nodes_hl3,
	                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'f_fum':n_classes,
    	            'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        	        'bias':tf.Variable(tf.random_normal([n_classes])),}
    

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    # output=tf.nn.sigmoid(output)
	return output

def train_neural_network(x):
	global prediction
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end,:])
				batch_y = np.array(train_y[start:end])

				# batch_y = batch_y.reshape(len(batch_y))
				# n_values = int(np.max(batch_y)) + 1
				# batch_y = np.eye(n_values)[np.array(batch_y, dtype=np.int32)]

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))	
		save_path = saver.save(sess, "./model/model.ckpt")
		print("Model saved in path: %s" % save_path)	

		return (prediction.eval({x:test_x}))
		
	    

def train(trn_x,trn_y,tst_x,tst_y,ep,num_class):
	global train_x
	global train_y
	global test_x
	global test_y
	hm_epochs = ep
	n_classes = num_class

	train_x=trn_x
	train_y=trn_y
	test_x=tst_x
	test_y=tst_y

	# train_x=train_x.values
	# train_y=train_y.values

	# test_x=test_x.values
	# test_y=test_y.values

	# for i in range(len(train_y)):
	#     str=train_y[i]
	#     train_y[i]=str[-1]

	# for i in range(len(test_y)):
	#     str=test_y[i]
	#     test_y[i]=str[-1]    

	train_y = train_y.reshape(len(train_y))
	n_values_train = 10
	train_y = np.eye(n_values_train)[np.array(train_y, dtype=np.int32)]  #converting to one hot


	test_y = test_y.reshape(len(test_y))
	n_values = 10
	test_y = np.eye(n_values)[np.array(test_y, dtype=np.int32)]  #converting to one hot

	return train_neural_network(x)

def validate(validate_x):
	global prediction
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "./model/model.ckpt")
		print("Model restored.")
		return (prediction.eval({x:validate_x}))

