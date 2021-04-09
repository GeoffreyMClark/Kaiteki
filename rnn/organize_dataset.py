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

parent_dir = './whole-body-kinematics-kinetics-power/walk - walk with CT - Obstacle avoidance - Obstacle avoidance with CT/all joints angles torques powers/'

d1 = {}
for i  in range (len(condition_list)):
	d2 = {}
	for j in range (len(subjects)):
		d3 = {}
		for k in range (len(joints)):

			# load data using pandas and save data into a list
			filename = joints[k]+'_'+subjects[j]+"_"+condition_list[i]+'.xlsx'
			df_temp = pd.read_excel(parent_dir+filename, engine='openpyxl', header=None)

			d4 = {}
			data_k = []
			data_T = []
			data_P = []
			for kk in range (df_temp.shape[1]):
				data_k.append(df_temp[kk].values.tolist()[0:101])
				data_T.append(df_temp[kk].values.tolist()[101:202])
				data_P.append(df_temp[kk].values.tolist()[202:303])
			d4[parts[0]] = data_k
			d4[parts[1]] = data_T
			d4[parts[2]] = data_P 
			# put the list to a dictionary and make dictionary structure
			d3[joints[k]] = d4
		d2[subjects[j]] = d3
	d1[condition_list[i]] = d2


json = json.dumps(d1)
f = open("dataset.json","w")
f.write(json)
f.close()

