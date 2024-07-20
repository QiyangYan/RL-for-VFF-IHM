import numpy as np
import matplotlib.pyplot as plt

def convert_corner_2_world_coord(euler, object_size, centre):
    # if euler[2] < 0:
    #     euler[2] = 360 - euler[2]
    print("EULER: ", euler)
    # euler[2] = euler[2] + 90
    cross_length = np.sqrt(2) * (object_size / 2)
    corners = np.array([
        [centre[1] + cross_length * np.cos(np.deg2rad(45 - euler[2])),
         centre[0] - cross_length * np.sin(np.deg2rad(45 - euler[2])),
         centre[2]],
        [centre[1] + cross_length * np.cos(np.deg2rad(45 + euler[2])),
         centre[0] + cross_length * np.sin(np.deg2rad(45 + euler[2])),
         centre[2]],
        [centre[1] - cross_length * np.cos(np.deg2rad(45 - euler[2])),
         centre[0] + cross_length * np.sin(np.deg2rad(45 - euler[2])),
         centre[2]],
        [centre[1] - cross_length * np.cos(np.deg2rad(45 + euler[2])),
         centre[0] - cross_length * np.sin(np.deg2rad(45 + euler[2])),
         centre[2]],
    ])
    return corners

angle = 3.54937376
corners = convert_corner_2_world_coord(np.array([180,0,angle]), 1, np.array([0,0, 0]))

color_list = ['black', 'green', 'red', 'yellow']
for i, corner in enumerate(corners):
    plt.scatter(corner[0], corner[1], color=color_list[i])
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.axis('equal')
plt.grid(True)
plt.show()

