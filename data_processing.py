import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.interpolate import griddata


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)
    return data



def cutAndProcessKeitekiData():
    kinematics_raw = loadData('data/MN06_kinematics.csv')
    kinetics_raw = loadData('data/MN06_kinetics.csv')
    GRFforces_raw = loadData('data/MN06_forces_2000Hz_filtered by 20Hz cutoff_fixed for opensim.csv')
    accel_raw = loadData('data/MN06_walkacc.csv')
    emg_raw = loadData('data/MN06_walkemg.csv')

    # Kinematics Comuptation
    pelvis_tilt = kinematics_raw[:,1]
    hip_angle_R = kinematics_raw[:,7]
    thigh_angle_R = kinematics_raw[:,1] + kinematics_raw[:,7]
    knee_angle_R = kinematics_raw[:,10]
    tibia_angle_R = kinematics_raw[:,1] + kinematics_raw[:,7] + kinematics_raw[:,10]
    ankle_angle_R = kinematics_raw[:,11]
    foot_angle_R = kinematics_raw[:,1] + kinematics_raw[:,7] + kinematics_raw[:,10] + kinematics_raw[:,11]
    hip_angle_L = kinematics_raw[:,14]
    thigh_angle_L = kinematics_raw[:,1] + kinematics_raw[:,14]
    knee_angle_L = kinematics_raw[:,17]
    tibia_angle_L = kinematics_raw[:,1] + kinematics_raw[:,14] + kinematics_raw[:,17]
    ankle_angle_L = kinematics_raw[:,18]
    foot_angle_L = kinematics_raw[:,1] + kinematics_raw[:,14] + kinematics_raw[:,17] + kinematics_raw[:,18]

    # Kinetics Computation
    hip_moment_R = kinetics_raw[:,7]
    knee_moment_R = kinetics_raw[:,16]
    ankle_moment_R = kinetics_raw[:,18]
    hip_moment_L = kinetics_raw[:,10]
    knee_moment_L = kinetics_raw[:,17]
    ankle_moment_L = kinetics_raw[:,19]

    # Forces
    forceX_R = GRFforces_raw[0:-1:20,3]
    forceZ_R = GRFforces_raw[0:-1:20,4]
    forceY_R = GRFforces_raw[0:-1:20,5]
    forceX_L = GRFforces_raw[0:-1:20,9]
    forceZ_L = GRFforces_raw[0:-1:20,10]
    forceY_L = GRFforces_raw[0:-1:20,11]


    # Plotting basic raw data
    # plt.figure(1)
    # plt.plot(ankle_angle_R)
    # plt.plot(forceZ_R)
    # plt.figure(2)
    # plt.plot(ankle_angle_L)
    # plt.plot(forceZ_L)
    # plt.show()

    # Cut data into strides
    step_num=[]; phase_max=100; NDOF=26
    # cut_final = np.array([],dtype=np.int64).reshape(0,phase_max+1,NDOF)
    cut_final = []
    phase = np.linspace(0,phase_max,phase_max+1).reshape(1,phase_max+1)
    for i in range(3,ankle_angle_R.size):
        if forceZ_R[i-1] < 20 and forceZ_R[i] >= 20 and forceZ_R[i-3] <= 20:
            step_num.append(i-2)
            if len(step_num) == 2:
                xo = np.linspace(step_num[0],step_num[1],phase_max+1)
                xp = np.linspace(step_num[0],step_num[1],step_num[1]-step_num[0]+1)

                y1_cut = pelvis_tilt[step_num[0]:step_num[1]+1]
                y2_cut = hip_angle_R[step_num[0]:step_num[1]+1]
                y3_cut = thigh_angle_R[step_num[0]:step_num[1]+1]
                y4_cut = knee_angle_R[step_num[0]:step_num[1]+1]
                y5_cut = tibia_angle_R[step_num[0]:step_num[1]+1]
                y6_cut = ankle_angle_R[step_num[0]:step_num[1]+1]
                y7_cut = foot_angle_R[step_num[0]:step_num[1]+1]
                y8_cut = hip_angle_L[step_num[0]:step_num[1]+1]
                y9_cut = thigh_angle_L[step_num[0]:step_num[1]+1]
                y10_cut = knee_angle_L[step_num[0]:step_num[1]+1]
                y11_cut = tibia_angle_L[step_num[0]:step_num[1]+1]
                y12_cut = ankle_angle_L[step_num[0]:step_num[1]+1]
                y13_cut = foot_angle_L[step_num[0]:step_num[1]+1]

                y14_cut = hip_moment_R[step_num[0]:step_num[1]+1]
                y15_cut = knee_moment_R[step_num[0]:step_num[1]+1]
                y16_cut = ankle_moment_R[step_num[0]:step_num[1]+1]
                y17_cut = hip_moment_L[step_num[0]:step_num[1]+1]
                y18_cut = knee_moment_L[step_num[0]:step_num[1]+1]
                y19_cut = ankle_moment_L[step_num[0]:step_num[1]+1]

                y20_cut = forceX_R[step_num[0]:step_num[1]+1]
                y21_cut = forceZ_R[step_num[0]:step_num[1]+1]
                y22_cut = forceY_R[step_num[0]:step_num[1]+1]
                y23_cut = forceX_L[step_num[0]:step_num[1]+1]
                y24_cut = forceZ_L[step_num[0]:step_num[1]+1]
                y25_cut = forceY_L[step_num[0]:step_num[1]+1]


                y1 = np.interp(xo,xp,y1_cut).reshape(1,phase_max+1)
                y2 = np.interp(xo,xp,y2_cut).reshape(1,phase_max+1)
                y3 = np.interp(xo,xp,y3_cut).reshape(1,phase_max+1)
                y4 = np.interp(xo,xp,y4_cut).reshape(1,phase_max+1)
                y5 = np.interp(xo,xp,y5_cut).reshape(1,phase_max+1)
                y6 = np.interp(xo,xp,y6_cut).reshape(1,phase_max+1)
                y7 = np.interp(xo,xp,y7_cut).reshape(1,phase_max+1)
                y8 = np.interp(xo,xp,y8_cut).reshape(1,phase_max+1)
                y9 = np.interp(xo,xp,y9_cut).reshape(1,phase_max+1)
                y10 = np.interp(xo,xp,y10_cut).reshape(1,phase_max+1)
                y11 = np.interp(xo,xp,y11_cut).reshape(1,phase_max+1)
                y12 = np.interp(xo,xp,y12_cut).reshape(1,phase_max+1)
                y13 = np.interp(xo,xp,y13_cut).reshape(1,phase_max+1)

                y14 = np.interp(xo,xp,y14_cut).reshape(1,phase_max+1)
                y15 = np.interp(xo,xp,y15_cut).reshape(1,phase_max+1)
                y16 = np.interp(xo,xp,y16_cut).reshape(1,phase_max+1)
                y17 = np.interp(xo,xp,y17_cut).reshape(1,phase_max+1)
                y18 = np.interp(xo,xp,y18_cut).reshape(1,phase_max+1)
                y19 = np.interp(xo,xp,y19_cut).reshape(1,phase_max+1)

                y20 = np.interp(xo,xp,y20_cut).reshape(1,phase_max+1)
                y21 = np.interp(xo,xp,y21_cut).reshape(1,phase_max+1)
                y22 = np.interp(xo,xp,y22_cut).reshape(1,phase_max+1)
                y23 = np.interp(xo,xp,y23_cut).reshape(1,phase_max+1)
                y24 = np.interp(xo,xp,y24_cut).reshape(1,phase_max+1)
                y25 = np.interp(xo,xp,y25_cut).reshape(1,phase_max+1)

                cut_temp = np.concatenate((phase,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25),axis=0).reshape(NDOF, phase_max+1)
                # cut_final = np.concatenate((cut_final,cut_temp),axis=0)
                cut_final.append(cut_temp)
                del step_num[0]
    return cut_final


if __name__ == '__main__':
    demonstrationTrajectories = cutAndProcessKeitekiData()


    fig, axs = plt.subplots(len(demonstrationTrajectories[0]), 1)
    for i in range(1,len(demonstrationTrajectories)-1):
        for j in range(demonstrationTrajectories[i].shape[0]):
            axs[j].plot(demonstrationTrajectories[i][j,:])
    plt.show()