import gymnasium as gym
import time
import mujoco
import os

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
#     env = gym.make("VariableFriction-v0", render_mode="human")
#     print(env.observation_space)
#     observation, info = env.reset(seed=42)
#     for _ in range(1000):
#         action = env.action_space.sample()  # this is where you would insert your policy
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         if terminated or truncated:
#             observation, info = env.reset()
#     env.close()

    # print(env.observation_space)

    # env = gym.make("HandManipulateBlock-v1")
    # print(env.observation_space)
    # print(env.observation_space["observation"])

    # The following always has to hold:
    # assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    # assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
    # assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

MANIPULATE_BLOCK_XML = os.path.join("/Users/qiyangyan/Desktop/IHM_finger", "IHM.xml")
fullpath = MANIPULATE_BLOCK_XML
_mujoco = mujoco
model = _mujoco.MjModel.from_xml_path(fullpath)
data = _mujoco.MjData(model)

print(model.actuator_gainprm)
print(model.actuator_biastype)
print(model.actuator_biasprm)
print(model.actuator_ctrlrange)

# pos_param = []
# pos_param.append(model.actuator_gainprm[0][0])
# pos_param.append(model.actuator_biastype[0])
# pos_param.append(model.actuator_biasprm[0][1])
# pos_param.append(model.actuator_ctrlrange[0])
# print(pos_param[3])

# for i in range(2):
#     print(i)

'''Plot success rate'''
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load data from .npy file
# data = np.load('/Users/qiyangyan/Desktop/TD3+HER/success_list.npy')
#
# # Create an array for the x-axis (index values)
# x = np.arange(len(data))
#
# # Plot the data
# plt.plot(x, data)
#
# # Label the axes
# plt.xlabel('Episode')
# plt.ylabel('Success Rate')
#
# # Display the plot
# plt.show()

''''''

# env = gym.make("VariableFriction-v0",render_mode="human")
# env.reset()
# action = [0,0,0]
#
#
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [0,0,1]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [1,0,1]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [0,0,0]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)

# start_time = time.time()
# for _ in range(20):
#     next_env_dict, _, _, _, _ = env.step(action)
#     print("time:", time.time() - start_time, "L:", friction_state_L, "R", friction_state_R)
#
# print("next action")
# action = [0,0,1]
# count = 0
# while friction_state_L != 0 and friction_state_R != 0.34:
#     print("time:", time.time() - start_time, "count:", count, "L:", friction_state_L, "R", friction_state_R)
#     next_env_dict, _, _, _, _ = env.step(action)
#     next_state = next_env_dict["observation"]
#     friction_state_L = next_state[1]
#     friction_state_R = next_state[3]
#     count += 1

print("finish")

# 35 reaches approaximate right