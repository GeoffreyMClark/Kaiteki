import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


csvs = ['LKnee_MN04_walk_dual.xlsx',  'LKnee_MN04_walk.xlsx']
dfs = [np.array(pd.read_excel(i, header=None, engine='openpyxl')) for i in csvs]

# df0 = dfs[0]
# angle0 = df0[:101,:]
angles = [np.array(x[:101,:]) for x in dfs]
mean_angles = [np.mean(x, axis=1) for x in angles]
var_angles = [np.var(x, axis=1) for x in angles]
min_angles = [mean_angles[i] - var_angles[i] for i in range(len(mean_angles))]
max_angles = [mean_angles[i] + var_angles[i] for i in range(len(mean_angles))]

torques = [np.array(x[101:202,:]) for x in dfs]
mean_torques = [np.mean(x, axis=1) for x in torques]
var_torques = [np.var(x, axis=1) for x in torques]
min_torques = [mean_torques[i] - var_torques[i] for i in range(len(mean_torques))]
max_torques = [mean_torques[i] + var_torques[i] for i in range(len(mean_torques))]

# plt.plot(angles[0])
# plt.show()

# min_angle0 = mean_angle - variance
# max_angle0 = mean_angle + variance

fig1 = plt.figure(1,figsize=(30,15))
ax1 = fig1.add_subplot(111)
ax1.plot(mean_angles[0], color='r',label='ankle - walk_dual')
# ax1.plot(mean_angles[1], color='b',label='ankle - walk')
ax1.fill_between(range(101),max_angles[0], min_angles[0], color='r', alpha=0.1)
# ax1.fill_between(range(101),max_angles[1], min_angles[1], color='b', alpha=0.1)

# ax2 = fig1.add_subplot(212)
# ax2.plot(mean_torques[0], color='r',label='ankle - walk_dual')
# ax2.plot(mean_torques[1], color='b',label='ankle - walk')
# ax2.fill_between(range(101),max_torques[0], min_torques[0], color='r', alpha=0.1)
# ax2.fill_between(range(101),max_torques[1], min_torques[1], color='b', alpha=0.1)
plt.show()

