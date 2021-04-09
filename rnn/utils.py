import os
import glob
import random
import csv
import shutil
import json
import pandas as pd


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



##################### extract one data as example #################

# follow the convention data['condition']['subject']['joint']
# you can do whatever cross-reference as you need
data_01 = data['walk']['MN02']['LFoot']['kinematics']
data_02 = data['walk']['MN03']['LFoot']['kinematics']
data_03 = data['walk']['MN04']['LFoot']['kinematics']

# every data is saved as a 2D list
print('col: ', len(data_01)) 
print('row: ', len(data_01[0]))


import matplotlib.pyplot as plt

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