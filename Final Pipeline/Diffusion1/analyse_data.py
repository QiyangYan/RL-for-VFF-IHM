import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Function to load data from a pickle file
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Replace 'path_to_file.pkl' with the actual file path
# pickle_file_path = '/Users/qiyangyan/Desktop/Training Files/Diffusion1/hopper-medium-v2.pkl'
# pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-1686demos'
# pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-1817demos'
# pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-1135demos'
# pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-2mmPolicy_5mmThreshold'
pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-2mmPolicy_2mmThreshold'
data = load_pickle(pickle_file_path)

# Print the loaded data
# print("Loaded data:", data)

# If you want to understand the structure of the loaded data
if data is not None:
    print("Type of the loaded data:", type(data))
    if isinstance(data, dict):
        print("Keys in the dictionary:", data.keys())
    elif isinstance(data, list):
        print("Number of elements in the list:", len(data))
    elif isinstance(data, (set, tuple)):
        print("Number of elements:", len(data))
    else:
        print("Data:", data)

for key in data.keys():
    print(np.shape(data[key]))

# print(data['observations'][:7])
# print("1", data['desired_goals'][0])
# print(data['rewards'][:7])
# print(data['terminals'][6])

''' Print actions '''
print(data['observations'][:5])

''' Measure number of episodes '''
num = 0
for i, item in enumerate(data['terminals']):
    if item:
        num += 1
print(num)

''' Measure episode length '''
episode_len = []
last = 0
for i, item in enumerate(data['terminals']):
    if item:
        episode_len.append(i - last)
        last = i
print("Mean: ", np.mean(episode_len))
print("STD: ", np.std(episode_len))
print("Max: ", np.max(episode_len))
print("Min: ", np.min(episode_len))

''' Normalize state and desired goal '''
def normalize(data, mean, std):
    std = np.where(std == 0, 1, std)
    return (data-mean)/std

goal_mean = np.mean(data['desired_goals'], axis=0)
goal_std = np.std(data['desired_goals'], axis=0)
desired_goal_norm = normalize(data['desired_goals'], goal_mean, goal_std)
data['desired_goals'] = desired_goal_norm
desired_goal_norm_param = {
    'mean': goal_mean,
    'std': goal_std,
}

obs_mean = np.mean(data['observations'], axis=0)
obs_std = np.std(data['observations'], axis=0)
obs_norm = normalize(data['observations'], obs_mean, obs_std)
data['observations'] = obs_norm
obs_norm_param = {
    'mean': obs_mean,
    'std': obs_std,
}

''' Normalize desired goal '''
# print("Check action: ", np.max(data['actions'][0]), np.min(data['actions'][0]), np.max(data['actions'][1]), np.min(data['actions'][1]))
#
# print(np.max(data['desired_goals'][:, 0]), np.min(data['desired_goals'][:, 0]))
# print(np.max(data['desired_goals'][:, 1]), np.min(data['desired_goals'][:, 1]))
#
# sum = 0
# for item in data['desired_goals']:
#     sum+=item[0]
# print(sum/len(data['desired_goals']))
#
# print("Desired Goal: ", np.mean(data['desired_goals'], axis=0), np.std(data['desired_goals'], axis=0))

# def normalize(data, mean, std):
#     std = np.where(std == 0, 1, std)
#     return (data-mean)/std
#
# goal_mean = np.mean(data['desired_goals'], axis=0)
# goal_std = np.std(data['desired_goals'], axis=0)
# desired_goal_norm = normalize(data['desired_goals'], goal_mean, goal_std)
# desired_goal_norm_param = {
#     'mean': goal_mean,
#     'std': goal_std,
# }
# with open('../Real4/Real4/models/desired_goal_norm_param.pkl', 'wb') as f:
#     pickle.dump(desired_goal_norm_param, f)

# print(goal_mean, goal_std)
# print("1: ", desired_goal_norm)
# print("2: ", data['desired_goals'])
#
# print(np.max(desired_goal_norm[:, 0]), np.min(desired_goal_norm[:, 0]))
# print(np.max(desired_goal_norm[:, 1]), np.min(desired_goal_norm[:, 1]))

''' Plot target pose region '''
# plt.figure()
# for i, item in enumerate(data['terminals']):
#     if item:
#         plt.scatter(data['desired_goals'][i][0], data['desired_goals'][i][1])
# plt.xlabel("y", fontsize='large', fontweight='bold')
# plt.ylabel("x", fontsize='large', fontweight='bold')
# plt.title("Target Position Distribution", fontweight='bold')
# plt.grid(which='major')
# file_path = "/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/target_position_distribution.png"
# plt.savefig(file_path)
# plt.close()  # Good practice to close the plot to free resources
# print("Plot saved successfully.")

# path = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/result/VariableFriction.pth'
# data = torch.load(path, map_location=torch.device('cpu'))
# print(data.keys())
# print(data['state_normalizer_mean'], data['state_normalizer_std'])