import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import random
import csv
import shutil
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math



condition_list = ['walk', 'walk_dual','OA', 'OA_dual']

subjects = ['MN02', 'MN03','MN04','MN05','MN06','MN07',
'MN08','MN09','MN10','MN11']

joints = ['LAnkle', 'LFemur', 'LFoot', 'LHip', 'LKnee', 'LTibia',
'RAnkle', 'RFemur', 'RFoot', 'RHip', 'RKnee', 'RTibia',
'Lumbar_bending', 'Lumbar_flexion', 'Lumbar_rotation', 
'Pelvis_list', 'Pelvis_rotation', 'Pelvis_tilt'] 

parts = ['kinematics', 'kinetics', 'power']

# Opening JSON file
f = open('dataset.json',)
data = json.load(f)

# ##################### extract one data as example #################

# follow the convention data['condition']['subject']['joint']
# you can do whatever cross-reference as you need
data_01 = data['walk']['MN02']['LFoot']['kinematics']
data_02 = data['walk']['MN03']['LFoot']['kinematics']
data_03 = data['walk']['MN04']['LFoot']['kinematics']
data_04 = data['walk']['MN05']['LFoot']['kinematics']

# every data is saved as a 2D list
# print('col: ', len(data_01)) 
# print('row: ', len(data_01[0]))

# preprocess data into 2 cycles one sequence
def preprocess_sequence(data):
	seq_len = math.floor(len(data)/2)
	data_temp = []
	idx = 0
	for i in range (seq_len):
		seq = data[idx][:-1] + data[idx+1]
		data_temp.append(seq)
		idx = idx+2
	data = data_temp
	return data

training_sequence = []
for i in range (4):
	seq = preprocess_sequence(data['walk']['MN0'+str(i+2)]['LFoot']['kinematics'])
	training_sequence.append(seq)

data_01 = training_sequence[0]
data_02 = training_sequence[1]
data_03 = training_sequence[2]
data_04 = training_sequence[3]



import matplotlib.pyplot as plt

##################### to plot the example data #################

fig = plt.figure(figsize=(2, 2))

plt.subplot(2, 2, 1)
x = list(range(1, len(data_01[0])+1 ))
for i in range (len(data_01)):
	plt.plot(x, data_01[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN02')

plt.subplot(2, 2, 2)
x = list(range(1, len(data_02[0])+1 ))
for i in range (len(data_02)):
	plt.plot(x, data_02[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN03')


plt.subplot(2, 2, 3)
x = list(range(1, len(data_03[0])+1 ))
for i in range (len(data_03)):
	plt.plot(x, data_03[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN04')


plt.subplot(2, 2, 4)
x = list(range(1, len(data_04[0])+1 ))
for i in range (len(data_04)):
	plt.plot(x, data_04[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN05')

plt.show()


# ########################### prepare the training data ##################
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return X, y

def get_training_data(data_sequence, n_steps):
	for i in range (len(data_sequence)):
		if i == 0:
			X , y = split_sequence(data_sequence[i], n_steps)
		else:
			X1, y1 = split_sequence(data_sequence[i], n_steps)
			X = X + X1
			y = y + y1
	return X, y

def get_model_input(training_sequence, n_steps):
	for i in range (len(training_sequence)):
		if i == 0:
			X, y = get_training_data(training_sequence[i], 10)
		else:
			X_temp, y_temp = get_training_data(training_sequence[i], 10)
			X = X + X_temp
			y = y + y_temp
	return X, y



# ########################### build model ##################
class MCDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
	    self.rate = rate
	    super(MCDropout, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
	    config = super().get_config().copy()
	    # config = super(MCDropout, self).get_config()
	    config.update({
	        'rate': self.rate
	    })
	    return config


def rnn_model():
	# define model
	model = keras.Sequential()
	model.add(layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(10, 1)))
	model.add(MCDropout(rate=0.2))
	model.add(layers.LSTM(100, activation='relu'))
	model.add(MCDropout(rate=0.2))
	model.add(layers.Dense(1))
	model.compile(optimizer='adam', loss='mse')
	print(model.summary())
	return model


# # # ############################ Training #######################
# X, y = get_model_input(training_sequence, 10)

# print('the shape of training data: ', len(X) , ' ', len(X[0]))
# X = np.array(X)
# y = np.array(y)

# # reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))

# model = rnn_model()
# model.fit(X, y, batch_size=100, epochs=30)
# model.save('walk_rnn.h5')



# ############################### load the model for prediction #######################

def predict_proba(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    # print(np.stack(preds))
    min_pred = max(np.stack(preds))
    max_pred = min(np.stack(preds))
    return np.stack(preds).mean(axis=0), min_pred, max_pred


model = tf.keras.models.load_model('walk_rnn.h5', custom_objects={'MCDropout': MCDropout})
print(model.summary())


# load some data for testing 
test_sequence = []
for i in range (4):
	seq = preprocess_sequence(data['walk']['MN0'+str(i+6)]['LFoot']['kinematics'])
	test_sequence.append(seq[1])

X =[]
y = []

y_pred = []
y_min = []
y_max = []


print("check-01")
# process the data and feed to the model
for i in range (len(test_sequence)):
	X_temp , y_temp = split_sequence(test_sequence[i], 10)
	X.append(X_temp)
	y.append(y_temp)

for i in range (len(y)):
	pred = []
	ymin = []
	ymax = []
	print("check-0"+str(i+2))
	for j in range (len(y[i])):
		x_input = np.array(X[i][j])
		x_input = x_input.reshape((1, 10, 1))
		# yhat = model.predict(x_input)
		yhat, min_pred, max_pred = predict_proba(x_input, model, 10)
		# print(yhat[0], min_pred[0])
		pred.append(yhat[0][0])
		ymin.append(min_pred[0][0])
		ymax.append(max_pred[0][0])
	y_pred.append(pred)
	y_min.append(ymin)
	y_max.append(ymax)


# # print(len(y))
# # print(len(y_pred))
# # print(y_pred[0])


# # # ##################### to plot the example data #################

# fig = plt.figure()

# x = list(range(1, len(y)+1 ))
# x = np.array(x)

# plt.plot(x, y, '.-b', label='ground truth')
# plt.plot(x, y_pred, '--r', label='prediction')

# y_pred = np.array(y_pred)
# y_min = np.array(y_min)
# y_max = np.array(y_max)

# print()

# plt.fill_between(x, y_pred - (y_pred - y_min), y_pred + (y_max - y_pred),
#                  color='red', alpha=0.2)
# plt.grid()
# plt.xlabel('step')
# plt.ylabel('LFoot position - MN03')
# plt.legend()

# plt.show()


fig = plt.figure(figsize=(2, 2))

for i in range (len(y)):
	plt.subplot(2, 2, i+1)
	x = list(range(1, len(y[i])+1 ))
	x = np.array(x)

	plt.plot(x, y[i], '.-b', label='ground truth')
	plt.plot(x, y_pred[i], '--r', label='prediction')

	pred = np.array(y_pred[i])
	ymin = np.array(y_min[i])
	ymax = np.array(y_max[i])
	plt.fill_between(x, pred - (pred - ymin), pred + (ymax - pred),
                 color='red', alpha=0.2)
	plt.grid()
	plt.xlabel('step')
	plt.ylabel('LFoot position - MN0'+str(i+6))
	plt.legend()
plt.show()


