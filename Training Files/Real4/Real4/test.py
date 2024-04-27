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

import numpy as np
import matplotlib.pyplot as plt

# Number of points to generate for plotting
num_points = 1000

# Generate x values uniformly
x_values = np.random.uniform(-0.04, 0.04, num_points)

# Calculate corresponding y values
y_upper = 1.625 * np.abs(x_values) - 0.315
y_values = np.random.uniform(-0.24, y_upper)

# Adjust y using np.clip (though in the original code it seems to have a mistake in order)
y_values = np.clip(y_values - 0.02, -0.24, -0.30)  # Note: This clip range seems incorrect as it won't change any values based on your original code logic

# Plotting the points
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sampling Area for Coordinate Generation')
plt.grid(True)

# Show x and y boundaries
plt.axhline(y=-0.24, color='r', linestyle='--')
plt.fill_between(np.linspace(-0.04, 0.04, 500), -0.24, 1.625 * np.abs(np.linspace(-0.04, 0.04, 500)) - 0.315, color='gray', alpha=0.5)
plt.show()

