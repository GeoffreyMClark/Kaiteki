"""
Functional Principal Component Analysis
=======================================
Explores the two possible ways to do functional principal component analysis.
"""

# Author: Yujian Hong
# License: MIT

import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import plot_fpca_perturbation_graphs
from skfda.exploratory.visualization import _get_component_perturbations
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial
from skfda.exploratory.visualization import Boxplot
from skfda.representation.grid import FDataGrid
from skfda.exploratory.visualization._utils import _get_figure_and_axes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import csv
import json
import pdb


##############################################################################
# In this example we are going to use functional principal component analysis to
# explore datasets and obtain conclusions about said dataset using this
# technique.
#
# First we are going to fetch the Berkeley Growth Study data. This dataset
# correspond to the height of several boys and girls measured from birth to
# when they are 18 years old. The number and time of the measurements are the
# same for each individual. To better understand the data we plot it.
condition_list = ['walk', 'walk_dual','OA', 'OA_dual']

subjects = ['MN02', 'MN03','MN04','MN05','MN06','MN07',
'MN08','MN09','MN10','MN11']

joints = ['LAnkle', 'LFemur', 'LFoot', 'LHip', 'LKnee', 'LTibia',
'RAnkle', 'RFemur', 'RFoot', 'RHip', 'RKnee', 'RTibia',
'Lumbar_bending', 'Lumbar_flexion', 'Lumbar_rotation',
'Pelvis_list', 'Pelvis_rotation', 'Pelvis_tilt']

parts = ['kinematics', 'kinetics', 'power']

# Opening JSON file
f = open('../dataset.json',)
data = json.load(f)


# csvs = ['LKnee_MN04_walk_dual.xlsx',  'LKnee_MN04_walk.xlsx']
# dfs = [np.array(pd.read_excel(i, header=None, engine='openpyxl')) for i in csvs]

k_all = np.array(data['walk_dual']['MN04']['LKnee']).T
ka = k_all[:101,:]
# plt.plot(ka[:101,:])
# plt.show()

# df0 = dfs[0]
# angle0 = df0[:101,:]
# angles = [np.array(x[:101,:]) for x in dfs]
# mean_angles = [np.mean(x, axis=1) for x in angles]
# var_angles = [np.var(x, axis=1) for x in angles]
# min_angles = [mean_angles[i] - var_angles[i] for i in range(len(mean_angles))]
# max_angles = [mean_angles[i] + var_angles[i] for i in range(len(mean_angles))]


# torques = [np.array(x[101:202,:]) for x in dfs]
# mean_torques = [np.mean(x, axis=1) for x in torques]
# var_torques = [np.var(x, axis=1) for x in torques]
# min_torques = [mean_torques[i] - var_torques[i] for i in range(len(mean_torques))]
# max_torques = [mean_torques[i] + var_torques[i] for i in range(len(mean_torques))]

# plt.plot(angles[0])
# plt.show()

# a0 = angles[0].T
a0 = ka.T
df = FDataGrid(a0)

dataset = skfda.datasets.fetch_growth()
# y = dataset['target']
# fd = dataset['data']
# df = dataset['data']
fd = copy.deepcopy(df)
fd.plot()
plt.show()





##############################################################################
# FPCA can be done in two ways. The first way is to operate directly with the
# raw data. We call it discretized FPCA as the functional data in this case
# consists in finite values dispersed over points in a domain range.
# We initialize and setup the FPCADiscretized object and run the fit method to
# obtain the first two components. By default, if we do not specify the number
# of components, it's 3. Other parameters are weights and centering. For more
# information please visit the documentation.
# fpca_discretized = FPCA(n_components=2)
# fpca_discretized.fit(fd)
# fpca_discretized.components_.plot()

##############################################################################
# In the second case, the data is first converted to use a basis representation
# and the FPCA is done with the basis representation of the original data.
# We obtain the same dataset again and transform the data to a basis
# representation. This is because the FPCA module modifies the original data.
# We also plot the data for better visual representation.
# dataset = fetch_growth()
# fd = copy.deepcopy(df)
# basis = skfda.representation.basis.Fourier(n_basis=7)
# basis_fd = fd.to_basis(basis)
# basis_fd.plot()

##############################################################################
# We initialize the FPCABasis object and run the fit function to obtain the
# first 2 principal components. By default the principal components are
# expressed in the same basis as the data. We can see that the obtained result
# is similar to the discretized case.
# fpca = FPCA(n_components=2)
# fpca.fit(basis_fd)
# fpca.components_.plot()

##############################################################################
# To better illustrate the effects of the obtained two principal components,
# we add and subtract a multiple of the components to the mean function.
# We can then observe now that this principal component represents the
# variation in the mean growth between the children.
# The second component is more interesting. The most appropriate explanation is
# that it represents the differences between girls and boys. Girls tend to grow
# faster at an early age and boys tend to start puberty later, therefore, their
# growth is more significant later. Girls also stop growing early

# plot_fpca_perturbation_graphs(basis_fd.mean(),
#                               fpca.components_,
#                               30,
#                               fig=plt.figure(figsize=(6, 2 * 4)))

##############################################################################
# We can also specify another basis for the principal components as argument
# when creating the FPCABasis object. For example, if we use the Fourier basis
# for the obtained principal components we can see that the components are
# periodic. This example is only to illustrate the effect. In this dataset, as
# the functions are not periodic it does not make sense to use the Fourier
# basis
# dataset = fetch_growth()
# fd = dataset['data']
# fd = copy.deepcopy(df)
# basis = skfda.representation.basis.Fourier(n_basis=7)
# basis_fd = fd.to_basis(basis)
# fpca = FPCA(n_components=2, components_basis=Fourier(n_basis=7))
# fpca.fit(basis_fd)
# fpca.components_.plot()

# plot_fpca_perturbation_graphs(basis_fd.mean(),
#                               fpca.components_,
#                               30,
#                               fig=plt.figure(figsize=(6, 2 * 4)))

# mean = basis_fd.mean()
# multiple = 30
# components = fpca.components_
# perts = mean.copy()
# evr = fpca.explained_variance_ratio_
# num1 = int(30*evr[0])
# num2 = int(30*evr[0])
# perts = perts.concatenate(perts[0] + int(30*evr[0]) * components[0] + int(30*evr[1])*components[1])
# perts = perts.concatenate(perts[0] - int(30*evr[0]) * components[0] - int(30*evr[1])*components[1])


# # fig, axes = _get_figure_and_axes(None,None,None)
# # # if not axes:
# # axes = fig.subplots(nrows=len(components-1))
# perts.plot()
# # mean = basis_fd.mean()
# plt.show()

# import pdb
# pdb.set_trace()
# # aux = _get_component_perturbations(mean, fpca.components_, 1, 30)
# # eval_points = np.linspace(0,1,101)
# # mat = mean(eval_points)
# # fig1 = plt.figure()

# # plt.plot(mat[0], color='k')
# # plt.plot(ka, color='r')
# plt.show()

##############################################################################
# We can observe that if we switch to the Monomial basis, we also lose the
# key features of the first principal components because it distorts the
# principal components, adding extra maximums and minimums. Therefore, in this
# case the best option is to use the BSpline basis as the basis for the
# principal components
# dataset = fetch_growth()
# # fd = dataset['data']
# fd = copy.deepcopy(df)
# basis_fd = fd.to_basis(BSpline(n_basis=7))
# fpca = FPCA(n_components=2, components_basis=Monomial(n_basis=4))
# fpca.fit(basis_fd)
# fpca.components_.plot()
# pdb.set_trace()
# plt.show()
