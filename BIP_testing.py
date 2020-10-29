import matplotlib.pyplot as plt
import numpy as np
from data_processing import cutAndProcessKeitekiData
import copy

# Import the library.
import intprim


def plot_partial_trajectory(infered, test, mean, phase, dof_names):
    mean_phase = np.linspace(0,1,100)
    test_phase = np.linspace(0,phase,test.shape[1])
    infer_phase = np.linspace(phase,1,infered.shape[1])
    optimized_phase = np.linspace(phase,1,80)
    centers = primitive.basis_model.centers

    fig, axs = plt.subplots(len(dof_names), 1,figsize=(30,15))
    for i in range(len(dof_names)):
        axs[i].plot(mean_phase,mean[i],'k')
        axs[i].plot(test_phase,test[i],'b')
        axs[i].plot(infer_phase, infered[i],'r--')
        axs[i].set_ylabel(dof_names[i], fontsize=18)
        axs[i].grid(True)  
    axs[-1].set_xlabel('Phase', fontsize=18)
    
    plt.show()

# Set a seed for reproducibility
np.random.seed(213413414)

# Define the data axis names.
total_names = ["Pelvis Tilt", "hip_angle_R", "thigh_angle_R", "knee_angle_R", "tibia_angle_R", "ankle_angle_R", "foot_angle_R", "hip_angle_L", "thigh_angle_L", "knee_angle_L", "tibia_angle_L", "ankle_angle_L", "foot_angle_L", "hip_moment_R", "knee_moment_R", "ankle_moment_R", "hip_moment_L", "knee_moment_L", "ankle_moment_L", "forceX_R", "forceZ_R", "forceY_R", "forceX_L", "forceZ_L", "forceY_L"]
select = [21,4,5]
active_dofs = np.array([0,1])
dof_names = np.asarray([e for i, e in enumerate(total_names) if i in select])

# Generate basis functions
basis_model_gaussian = intprim.basis.GaussianModel(12, 0.0025, dof_names)
# basis_model_gaussian.plot()
# plt.show()

# Generate data
training_trajectories = cutAndProcessKeitekiData()

# for i in range(len(select)):
#     temp_dof = full_trajectories[:,:,select[i]].reshape(full_trajectories.shape[0],full_trajectories.shape[1],1)
#     training_trajectories = np.concatenate((training_trajectories, temp_dof), axis=2)

# Define a scaling group where the DoFs are scaled separately.
scaling_group = []
for i in range(len(select)):
    scaling_group.append([i])

# perform basis selection
selection = intprim.basis.Selection(dof_names, scaling_groups = scaling_group)

# Initialize a BIP instance.
primitive = intprim.BayesianInteractionPrimitive(basis_model_gaussian, scaling_groups = scaling_group)

# First compute the scaling
for trajectory in training_trajectories:
    primitive.compute_standardization(trajectory[select,:])

# Then add the demonstrations
for trajectory in training_trajectories:
    primitive.add_demonstration(trajectory[select,:])

# Then add the demonstrations for selection.
for trajectory in training_trajectories:
    selection.add_demonstration(trajectory[select,:])

# Compute observation noise
# observation_noise = np.diag(selection.get_model_mse(basis_model_gaussian, np.array([0, 1, 2])))
observation_noise = np.diag(selection.get_model_mse(basis_model_gaussian, np.array(list(range(0,len(select))))))
print(observation_noise)

# Compute the phase mean and phase velocities from the demonstrations.
phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
phase_mean = 0.0
phase_var = 1e-4
process_var = 1e-8

# Define the initial mean/variance of the temporal state.
initial_phase_mean = [phase_mean, phase_velocity_mean]
initial_phase_var = [phase_var, phase_velocity_var]

# Initialize an ensemble Kalman filter
filter_enkf = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
    basis_model = basis_model_gaussian,
    initial_phase_mean = initial_phase_mean,
    initial_phase_var = initial_phase_var,
    proc_var = process_var,
    initial_ensemble = primitive.basis_weights)

################ Initialize the BIP instance for inference ################
# Set the filter using a deep copy that way we can re-use the filter for inference multiple times without having to re-create it from scratch.
# This is simply for convenience.
primitive.set_filter(copy.deepcopy(filter_enkf))

# Create test trajectories.
traj_select = int(np.random.uniform(0,len(training_trajectories)))
test_trajectories = [training_trajectories[traj_select][select,:]]
test_trajectory_partial = np.array(test_trajectories[0], copy = True)

# We must make sure to inflate the corresponding observation noise entry as well so we ignore the zero values.
test_observation_noise = observation_noise
for i in range(len(select)):
    if not i in active_dofs:
        test_trajectory_partial[i,:] = 0.0
        test_observation_noise[i, i] = 10000.0




################ Perform inference and plot the results ################
mean_trajectory = primitive.get_mean_trajectory()
prev_observed_index = 0
observed_index=20
inferred_trajectory, phase, traj_mean, var = primitive.generate_probable_trajectory_recursive(
    test_trajectory_partial[:, prev_observed_index:observed_index],
    test_observation_noise,
    active_dofs,
    num_samples = test_trajectory_partial.shape[1] - observed_index)

plot_partial_trajectory(inferred_trajectory, test_trajectory_partial[:, :observed_index], mean_trajectory, phase, dof_names)

pass





















