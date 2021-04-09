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


condition_list = ['walk', 'walk_dual','OA', 'OA_dual']

subjects = ['MN02', 'MN03','MN04','MN05','MN06','MN07',
'MN08','MN09','MN10','MN11']

joints = ['LAnkle', 'LFemur', 'LFoot', 'LHip', 'LKnee', 'LTibia',
'RAnkle', 'RFemur', 'RFoot', 'RHip', 'RKnee', 'RTibia',
'Lumbar_bending', 'Lumbar_flexion', 'Lumbar_rotation', 
'Pelvis_list', 'Pelvis_rotation', 'Pelvis_tilt'] 

# Opening JSON file
f = open('dataset.json',)
data = json.load(f)

# ##################### extract one data as example #################

# follow the convention data['condition']['subject']['joint']
# you can do whatever cross-reference as you need
data_01 = data['walk']['MN02']['LFoot']
data_02 = data['walk']['MN03']['LFoot']
data_03 = data['walk']['MN04']['LFoot']

# every data is saved as a 2D list
# print('col: ', len(data_01)) 
# print('row: ', len(data_01[0]))


import matplotlib.pyplot as plt

##################### to plot the example data #################

fig = plt.figure(figsize=(3, 1))

plt.subplot(3, 1, 1)
x = list(range(1, len(data_01[0])+1 ))
for i in range (len(data_01)):
	plt.plot(x, data_01[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN02')

plt.subplot(3, 1, 2)
x = list(range(1, len(data_02[0])+1 ))
for i in range (len(data_02)):
	plt.plot(x, data_02[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN03')


plt.subplot(3, 1, 3)
x = list(range(1, len(data_03[0])+1 ))
for i in range (len(data_03)):
	plt.plot(x, data_03[i])
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN04')

plt.show()


########################### prepare the training data ##################
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


########################### build model ##################
def rnn_model():
	# define model
	model = keras.Sequential()
	model.add(layers.LSTM(50, activation='relu', input_shape=(10, 1)))
	model.add(layers.Dense(1))
	model.compile(optimizer='adam', loss='mse')
	# print(model.summary())
	return model


############################ Training #######################
# X, y = get_training_data(data_01, 10)
# X = np.array(X)
# y = np.array(y)
# print(X.shape)

# # reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))

# model = rnn_model()
# model.fit(X, y, epochs=10, verbose=0)
# model.save('rnn.h5')



############################### load the model for prediction #######################
model = keras.models.load_model('rnn.h5')
print(model.summary())

test_sequence = data_02[10]
X , y = split_sequence(test_sequence, 10)

y_pred = []

for i in range (len(y)):
	x_input = np.array(X[i])
	x_input = x_input.reshape((1, 10, 1))
	yhat = model.predict(x_input, verbose=0)
	y_pred.append(yhat[0])

print(len(y))
print(len(y_pred))
# print(y_pred[0])


# ##################### to plot the example data #################

fig = plt.figure()

x = list(range(1, len(y)+1 ))
plt.plot(x, y, '.-b', label='ground truth')
plt.plot(x, y_pred, '--r', label='prediction')
plt.grid()
plt.xlabel('step')
plt.ylabel('LFoot position - MN03')
plt.legend()

plt.show()



