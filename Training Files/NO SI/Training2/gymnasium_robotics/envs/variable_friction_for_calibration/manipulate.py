# Information regarding get_base_manipulate_env
'''

'''

# Action Space and Observation Space Definitions
'''Action Space and Observation Space Definitions.

Action Space:
l stands for rotate to left
r stands for rotate to right
0 stands for low_fric
1 stands for high_fric
1 unit angle for 1 time step
        L   R   L_s  R_s
1       \   l   0   1
2       \   l   1   0
3       \   l   1   1
4       r   \   0   1
5       r   \   1   0
6       r   \   1   1

Observation Space:
State: Instead of using predefined feature vector, I will use image as input
# 1   x-coordinate of the object
# 2   y-coordinate of the object
# 3   z-coordinate of the object
# 4   angular velocity of the left finger
# 5   angular velocity of the right finger
# 6   angle of the left finger
# 7   angle of the right finger
# 8

Reward:
1 Next state +5
2 object has no z-axis angle/distance-shift +10
3 object has small z-axis angle/distance-shift +1
4 Reach goal_pos +100
5 Below Velocity max_limit +1
6 Below (x,y,z) Velocity desired_limit +10
7 Same Action +10
8 Different Action +5
9 Below z-axis Angular velocity limit + 10

Done: Object fall off finger/angle>certain degree, exceed max_limit

CONSIDER FOLLOWING QUESTIONS
1. How to set the reward? cause the dense reward in the robot hand example also only considers the position
and angular difference between goal and current position, without considering healthy.

'''


from typing import Union

import numpy as np
from gymnasium import error

from gymnasium_robotics.envs.variable_friction_for_calibration import MujocoHandEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation

import xml.etree.ElementTree as ET
import os

# from gymnasium_robotics.envs.plot_array import plot_numbers, plot_numbers_two_value

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def get_base_manipulate_env(HandEnvClass: MujocoHandEnv):
    """Factory function that returns a BaseManipulateEnv class that inherits from MujocoPyHandEnv or MujocoHandEnv depending on the mujoco python bindings."""

    '''基于这个BaseManipulateEnv, 按需求进行添加'''
    class BaseManipulateEnv(HandEnvClass):
        def __init__(
            self,
            target_position,  # with a randomized target position (x,y,z), where z is fixed but with tolerance
            target_rotation,  # with a randomized target rotation around z, no rotation around x and y
            reward_type,  # more likely dense
            initial_qpos=None,
            randomize_initial_position=False,
            randomize_initial_rotation=False,
            distance_threshold=0.005,
            rotation_threshold=0.1,
            slip_pos_threshold = 0.005,  # this is the tolerance for displacement along z-axis due to slipping
            slip_rot_threshold = 0.2,  # this is the tolerance for rotation around z-axis due to slipping
            n_substeps=20,
            relative_control=False,
            # ignore_z_target_rotation=False,
            **kwargs,
        ):
            """Initializes a new Hand manipulation environment.

            Args:
                model_path (string): path to the environments XML file
                target_position (string): the type of target position:
                    - fixed: target position is set to the initial position of the object
                    - random: target position is fully randomized according to target_position_range
                target_rotation (string): the type of target rotation:
                    - z: fully randomized target rotation around the Z axis
                target_position_range (np.array of shape (3, 2)): range of the target_position randomization
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                randomize_initial_position (boolean): whether or not to randomize the initial position of the object
                randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
                distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
                rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
                n_substeps (int): number of substeps the simulation runs on every call to step
                relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state

                Removed:
                ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            """
            self.target_position = target_position
            self.target_rotation = target_rotation
            # 这段代码用于将一组 “欧拉角表示的旋转” (“平行旋转”) 转换为 “四元数表示”
            self.parallel_quats = [
                rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
            ]
            self.randomize_initial_rotation = randomize_initial_rotation
            self.randomize_initial_position = randomize_initial_position
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.r_threshold = 0.005 # 0.001 is more accurate
            self.d_threshold = 0.01
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
            self.switchFriction_count = 0;
            self.terminate_r_limit = [0.05,0.14]
            self.L = 0.015
            self.success = False
            self.left_contact_idx = None;
            self.right_contact_idx = None;
            self.start_radi = 0
            self.slip_terminate = False
            self.friction_change_error = 0
            self.slip_error = 0
            self.last_r_diff = 0
            self.last_height = None
            self.last_angle = None
            self.slip_error_angle = None
            self.reward_history = []
            # self.pick_up_height = 0
            self.pick_up_height = 3
            self.reset_everything = True
            self.goal_radi = np.zeros(2)

            # self.successSlide = False
            # self.ignore_z_target_rotation = ignore_z_target_rotation

            assert self.target_position in ["fixed", "random"]
            assert self.target_rotation in ["fixed", "z"]
            initial_qpos = initial_qpos or {}

            super().__init__(
                n_substeps=n_substeps,
                initial_qpos=initial_qpos,
                relative_control=relative_control,
                **kwargs,
            )

        def compute_reward(self, achieved_goal, goal, info):
                return 1

        def _is_success(self, achieved_goal, desired_goal):
            """Indicates whether or not the achieved goal successfully achieved the desired goal."""
            return 1

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        # print("reset")
        '''self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        '''

        # print("\033[92m reset everything \033[0m")
        # self.reset_ctrl_type()

        self.data.ctrl[:] = 0

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)

        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        if self.firstEpisode:
            for _ in range(10):
                self._set_action(0)
                try:
                    self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
                except Exception:
                    return False
            self.firstEpisode = False

        return True


    def _sample_goal(self):
        goal = np.zeros(9)
        assert goal.shape == (9,)
        return goal


    def _get_obs(self):

        env_qpos = self.data.qpos
        env_qvel = self.data.qvel

        # achieved_goal = robot_qpos[4:11]
        robot_qpos = env_qpos[1:5]
        robot_qvel = env_qvel[1:5]

        '''two finger, two finger insert'''
        assert robot_qpos.shape == (4,)

        # for observation: 4 + 4 + 2 + 1
        observation = np.concatenate(
            [
                robot_qpos, # 4 element
                robot_qvel, # 4 element
            ]
        )

        complete_obs = {
            "observation": observation.copy(),
            "achieved_goal": np.zeros(2),
            "desired_goal": np.zeros(2),
        }
        return complete_obs

