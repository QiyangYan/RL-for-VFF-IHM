import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_dim1(true_action_first_elements):
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 1, 1)

    # indices = list(range(len(action_first_elements)))  # Creating a list of indices
    # plt.scatter(indices, action_first_elements, label='Action First Element', color='blue', s=20)  # Scatter plot
    indices = list(range(len(true_action_first_elements)))  # Creating a list of indices
    plt.scatter(indices, true_action_first_elements, label='True Action First Element', color='orange',
                s=10)  # Scatter plot
    plt.title('First Element of Actions')

    # indices = list(range(len(action_indicator)))  # Creating a list of indices
    # plt.scatter(indices, action_indicator, label='Action Third Element', color='blue', s=10)  # Scatter plot
    # indices = list(range(len(true_action_indicator)))  # Creating a list of indices
    # plt.scatter(indices, true_action_indicator, label='True Action Third Element', color='orange',
    #             s=5)  # Scatter plot
    # plt.title('Third Element of Actions')

    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)  # Add grid for better readability
    plt.legend()
    # plt.tight_layout()
    plt.show()

def discretize_action_to_control_mode_E2E(action):
    # Your action discretization logic here
    action_norm = (action + 1) / 2
    if 1 / 6 > action_norm >= 0:
        control_mode = 0
        friction_state = 1  # left finger high friction
    elif 2 / 6 > action_norm >= 1 / 6:
        control_mode = 1
        friction_state = 1
    elif 3 / 6 > action_norm >= 2 / 6:
        control_mode = 2
        friction_state = -1
    elif 4 / 6 > action_norm >= 3 / 6:
        control_mode = 3
        friction_state = -1
    elif 5 / 6 > action_norm >= 4 / 6:
        control_mode = 4
        friction_state = 0
    else:
        assert 1 >= action_norm >= 5 / 6, f"Wrong action size: {action, action_norm}"
        control_mode = 5
        friction_state = 0
    # print(action_norm)
    return friction_state, control_mode

''' Plot target pose region '''
pickle_file_path = {
    # "train": '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10k_demos_random_slide',
    # "test": '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10k_demos_random_slide_testDataset',
    # "21_1": '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos',
    "test_demo": "/Users/qiyangyan/Desktop/TrainingFiles/Real4/Real4/demonstration/cubecylinder_3k5_demos_seed=0",
}

' Plot '
# plt.figure(figsize=(12, 14))
# for idx, (key, path) in enumerate(pickle_file_path.items()):
#     print("Key: ", key)
#     with open(path, 'rb') as file:
#         data = pickle.load(file)
#     plt.subplot(2, 1, idx + 1)
#     count = 0
#     for i, item in enumerate(data['terminals']):
#         if item == 1:
#             plt.scatter(data['desired_goals'][i][0], data['desired_goals'][i][1])
#             count += 1
#         if count > 5000:
#             break
#     plt.xlabel("y", fontsize='large', fontweight='bold')
#     plt.ylabel("x", fontsize='large', fontweight='bold')
#     plt.title(f"Target Position Distribution ({key})", fontweight='bold')
#     plt.xlim([-0.1, 0.1])
#     plt.ylim([-0.33, -0.215])
#     plt.grid(which='major')
#
# plt.tight_layout()
# plt.show()

with open(pickle_file_path['test_demo'], 'rb') as file:
    data = pickle.load(file)

for key in data.keys():
    print(key)
    print(np.shape(data[key]))


action = data["actions"]
plt.scatter(range(len(action[:,1])), action[:,1])
discretize_action = []
# for item in action:
    # print(item)
    # if discretize_action_to_control_mode_E2E(item[1])[1] == 4:
    #     print(item[0])
    # discretize_action.append(discretize_action_to_control_mode_E2E(item)[1])
# plt.scatter(range(len(discretize_action)), np.array(discretize_action))
plt.show()

# file_path = "/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/target_position_distribution.png"
# plt.savefig(file_path)
# plt.close()  # Good practice to close the plot to free resources
# print("Plot saved successfully.")

# path = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/result/VariableFriction.pth'
# data = torch.load(path, map_location=torch.device('cpu'))
# print(data.keys())
# print(data['state_normalizer_mean'], data['state_normalizer_std'])

# # pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_slide'
# pickle_file_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10k_demos_random_slide'  # Training
# with open(pickle_file_path, 'rb') as file:
#     data = pickle.load(file)
# # for idx, item in enumerate(data['actions']):
# #     data['actions'][idx][1] = discretize_action_to_control_mode_E2E(item[1])[1]
# # plot_dim1(data['actions'][:,1])
