# import gymnasium as gym
# import numpy as np
#
# # env = gym.make("VariableFriction-v5", render_mode="human")
# env = gym.make("VariableFriction-v5")
# env.reset()
#
# for _ in range(10):
#     next_env_dict, r_dict, terminated, _, info_ = env.step(np.array([-1.38833278e-03, -1.23331150e-01,  2.00000000e-03,  7.06825390e-01,
#         7.07388061e-01,  4.33150262e-17,  4.32805725e-17]))
#     print("Goal: ", env.goal)
#     # base_poses = [0, 0.10136, 0, 0, 0, 0]
#     # base_camera = np.concatenate([base_poses[:3], np.zeros(4)])
#     # print(r_dict['goal_pose'])
#     # print(next_env_dict["achieved_goal"])

# def transform_angle(a):
#     if a > 90:
#         return a-270
#     else:
#         return 90+a

# Examples
# angles = [-90, 110, 90]
# transformed_angles = [a%360 for a in angles]
# print(transformed_angles)

import numpy as np
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Number of points to generate for plotting
# num_points = 1000
#
# # Generate x values uniformly
# x_values = np.random.uniform(-0.04, 0.04, num_points)
#
# # Calculate corresponding y values
# y_upper = 1.625 * np.abs(x_values) - 0.315
# y_values = np.random.uniform(-0.24, y_upper)
#
# # Adjust y using np.clip (though in the original code it seems to have a mistake in order)
# y_values = np.clip(y_values - 0.02, -0.24, -0.30)  # Note: This clip range seems incorrect as it won't change any values based on your original code logic
#
# # Plotting the points
# plt.figure(figsize=(8, 6))
# plt.scatter(x_values, y_values, alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sampling Area for Coordinate Generation')
# plt.grid(True)
#
# # Show x and y boundaries
# plt.axhline(y=-0.24, color='r', linestyle='--')
# plt.fill_between(np.linspace(-0.04, 0.04, 500), -0.24, 1.625 * np.abs(np.linspace(-0.04, 0.04, 500)) - 0.315, color='gray', alpha=0.5)
# plt.show()


import numpy as np

# List of NumPy arrays
# dict = {
#     'obs': []
# }
# observations = [
#     np.array([-0.5953975, -0.14035055, 0.63491931, -0.16413231, -0.07769869, -0.00529537, 0.14501552, -0.00585843, -0.25550377, -1.10302038]),
#     np.array([-0.26332136, 0.17718582, 0.19188872, -0.16409328, 1.02929784, 0.01621559, -0.91327727, -0.0059587, 0.27116507, -1.1099319])
# ]
# # Combining into a single 2D NumPy array
# combined_array = np.stack(observations)
# dict['obs'] = combined_array
#
# print(dict)

from scipy.spatial.transform import Rotation as R
import numpy as np
import rotations
#
print(R.from_quat([9.99999482e-01, 2.54176735e-05 , 5.45175729e-04  ,8.59493241e-04]).as_euler('xyz', degrees=True))

# quat_a = np.array([0.99580748, -0., -0., -0.09147382])
# quat_b = np.array([0.0032,     0. ,        0., 0.])
# euler_a = rotations.quat2euler(quat_a)
# euler_b = rotations.quat2euler(quat_b)
# if euler_a.ndim == 1:
#     euler_a = euler_a[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
# if euler_b.ndim == 1:
#     euler_b = euler_b[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
# euler_a[:,:2] = euler_b[:,:2]  # make the second and third term of euler angle the same
# quat_a = rotations.euler2quat(euler_a)
# quat_a = quat_a.reshape(quat_b.shape)
#
# # print(quat_a, quat_b)
# quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
# angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
#
#
# print(angle_diff)

# euler_desired = np.array([-156.05052894  ,  0.   ,         0.        ])
# quat_transform = R.from_euler('xyz', euler_desired, degrees=True).as_quat()
# print(quat_transform)



# print(R.from_quat(quat_transform).as_euler('xyz', degrees=True))
# quat_transform_quat = quat_transform.as_quat()
# quat_new = R.from_quat(quat_transform_quat) * R.from_quat(q_old)
# quat_new_quat = R.from_quat(q_old).as_quat()
# quat_new_quat = quat_new.as_quat()

# rotation_matrix = R.from_quat(quat_new_quat.as_quat()).as_matrix()
# rotation_matrix_permuted = rotation_matrix[:, [2, 1, 0]]  # Swap x and z columns
# check = R.from_matrix(rotation_matrix_permuted).as_quat()


# print(R.from_quat([1.15537045e-01,
#         9.93303172e-01,  6.08222775e-17,  7.07460361e-18]).as_euler('xyz', degrees=True))
# print(R.from_quat(q_old).as_euler('xyz', degrees=True))
# print(check)

# # Current orientation in Euler angles
# current_euler = np.array([180, 0, 120])
# # Desired orientation in Euler angles
# desired_euler = np.array([-150, 0, 0])
#
# # Convert Euler angles to rotation matrices
# current_rotation_matrix = R.from_euler('xyz', current_euler, degrees=True).as_matrix()
# desired_rotation_matrix = R.from_euler('xyz', desired_euler, degrees=True).as_matrix()
#
# # Calculate the rotation matrix to go from current to desired orientation
# transformation_matrix = desired_rotation_matrix @ np.linalg.inv(current_rotation_matrix)
#
# # Convert the transformation matrix back to Euler angles
# transformation_euler = R.from_matrix(transformation_matrix).as_quat()
#
# print(transformation_euler)



# goal_a = np.array([8.91395147e-04, - 2.69582251e-01, - 1.55608955e-05 ,
#
#          9.99999482e-01, 2.54176735e-05 , 5.45175729e-04  ,8.59493241e-04,])
# goal_b = np.array([-0.03867779, -0.30000001 , 0.002   ,    0.94791305 , 0.  ,        0.,
#   0.31852921])
# goal_a[2] = goal_b[2]
#
# d_pos = np.zeros_like(goal_a[..., 0])
#
# delta_pos = goal_a[..., :3] - goal_b[..., :3]
# d_pos = np.linalg.norm(delta_pos, axis=-1)
#
# quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
#
# euler_a = rotations.quat2euler(quat_a)
# euler_b = rotations.quat2euler(quat_b)
# if euler_a.ndim == 1:
#     euler_a = euler_a[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
# if euler_b.ndim == 1:
#     euler_b = euler_b[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
# euler_a[:,:2] = euler_b[:,:2]  # make the second and third term of euler angle the same
# quat_a = rotations.euler2quat(euler_a)
# quat_a = quat_a.reshape(quat_b.shape)
#
# # print(quat_a, quat_b)
# quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
# angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
# d_rot = angle_diff
# # assert d_pos.shape == d_rot.shape
#
# print("Check: ", np.rad2deg(angle_diff), np.rad2deg(euler_a), np.rad2deg(euler_b))

# import pickle
# import matplotlib.pyplot as plt
# # dataset_dir = '/Users/qiyangyan/Desktop/TrainingFiles/Real4/Real4/demonstration/1episode_seed=0_real_cube'
# dataset_dir = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel.pkl'
# with open(dataset_dir, 'rb') as f:
#     dataset = pickle.load(f)
# for key in dataset.keys():
#     print(key)
#
# for i, item in enumerate(dataset['terminals']):
#     if item == 1:
#         # print(dataset['desired_goals'][i])
#         print(R.from_quat(dataset['desired_goals'][i][3:7]).as_euler('xyz', degrees=True))

' Plot robot state '
# # episode_start_idx = np.where(dataset['terminals'] == 1)[0][0]
# episode_start_idx = 0
# episode_end_idx = np.where(dataset['terminals'] == 1)[0][0]
# episode_len = episode_end_idx - episode_start_idx
# # end_of_first_episode = np.where(dataset['terminals'] == 1)[0][0]
# # print(end_of_first_episode)
#
# print(R.from_quat(dataset['observations'][3][11:15]).as_euler('xyz', degrees=True))
#
# # Set up the figure with a 4x2 grid of subplots
# fig, axs = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
#
# # Plot each requested index in a separate subplot
# for i, idx in enumerate(range(0, 8)):
#     ax = axs[i // 2, i % 2]  # Calculate position in the 4x2 grid
#     ax.plot(range(episode_len), dataset['observations'][episode_start_idx:episode_end_idx][:, idx], label=f'Index {idx+1}')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Plot of Index {idx+1}')
#     ax.grid(True)
#
# # Set common labels
# plt.xlabel('Element Index')
# plt.title('Velocity of Joint')
# plt.legend(loc='upper right')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
