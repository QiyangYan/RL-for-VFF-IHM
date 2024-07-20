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
import pickle

from gymnasium_robotics.envs.diffusion3_cross_object import MujocoHandEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation
from gymnasium_robotics.envs.training5.DomainRandomisation.randomisation import RandomisationModule

import xml.etree.ElementTree as ET
import os


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat

def get_base_manipulate_env(HandEnvClass: MujocoHandEnv):
    """Factory function that returns a BaseManipulateEnv class that inherits from MujocoPyHandEnv or MujocoHandEnv depending on the mujoco python bindings."""
    # class domain_randomisation():
    #     @staticmethod
    #     def randomise_obs(obs):
    #         return obs
    #
    #     def randomise_physics_params(self):
    #         self.get_pos_ctrl_params()
    #         self.get_torque_ctrl_params()
    #         # self.check_physics_parameter()

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
            self.angle_threshold = 0.17
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
            self.switchFriction_count = 0;
            self.terminate_r_limit = [0.06,0.145]
            # self.terminate_r_limit = [0.06, 0.13]
            self.L = 0.015
            self.success = False
            self.left_contact_idx = None
            self.right_contact_idx = None
            self.left_contact_idx_real = None
            self.right_contact_idx_real = None
            self.start_radi = 0
            self.slip_terminate = False
            self.friction_change_error = 0
            self.slip_error = 0
            self.last_r_diff = 0
            self.last_height = None
            self.last_angle = None
            self.slip_error_angle = None
            self.reward_history = []
            self.pick_up_height = 0
            self.reset_everything = True
            self.goal_radi = np.zeros(2)
            self.domain_randomise = RandomisationModule()
            self.friction_change_penalty = False
            self.sliding = False
            self.gradients = []
            self.idx = 0

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

        """ ---------------------------------------------------------------------------  """
        """ ---------------------------------------------------------------------------  """
        """ RANDOMISATION """
        def randomise_physics_params(self):
            randomise = True
            print(" ---------------------------------------------------------- ")
            print("| Randomise physics param: ", randomise)
            self.get_pos_ctrl_params(randomise=randomise)
            # self.get_pos_ctrl_params(randomise=True)
            self.get_torque_ctrl_params()
            if randomise:
                self.check_physics_parameter()
            # self.check_physics_parameter()
            print(" ---------------------------------------------------------- ")

        def randomise_object(self, left_contact_point, right_contact_point, object_qpos):
            # print("| Randomise object")
            object_noise = self.domain_randomise.generate_gaussian_noise("object_position", 2, correlated=False)
            object_qpos[:2] += object_noise
            left_contact_point[:2] += object_noise
            right_contact_point[:2] += object_noise
            return left_contact_point, right_contact_point, object_qpos

        def randomise_joint(self, robot_qpos, robot_qvel):
            # print("| Not Randomise joint ")
            # joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            # robot_qpos[0] = robot_qpos[0] + joint_noise[0]
            # robot_qpos[2] = robot_qpos[2] + joint_noise[1]
            # print(self.data.ctrl)
            return robot_qpos, robot_qvel

        def get_correlated_obs_noise(self):
            self.correlated_noise_joint = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=True)
            self.correlated_noise_object = self.domain_randomise.generate_gaussian_noise("object_position", 2, correlated=True)
        """ ---------------------------------------------------------------------------  """
        """ ---------------------------------------------------------------------------  """

        def _is_success_old(self, achieved_goal, desired_goal):
            d_radi = np.array(self._get_achieved_goal_contact_point()) - np.array(self.goal_radi)
            success_radi = (np.mean(abs(d_radi)) < self.r_threshold).astype(np.float32)
            self.success = success_radi
            return success_radi

        def _is_success(self, achieved_goal, desired_goal):
            """
            Return True/False indicate success of the episode
            """
            if len(achieved_goal.shape) == 1:
                d_pos, d_rot = self._goal_distance(achieved_goal[:7], desired_goal[:7])
                d_rot = abs(d_rot)
                d_pos = abs(d_pos)
                # print("Pose success: ", d_pos < self.d_threshold)
                # print("Orientation success: ", d_rot < self.angle_threshold)
                # if d_pos < self.d_threshold:
                #     print("| d_pos")
                # if d_rot < self.angle_threshold:
                #     print("| d_rot")
                success = d_pos < self.d_threshold and d_rot < self.angle_threshold
                # if success:
                #     print("| Success ")
            elif len(achieved_goal.shape) > 1:
                d_pos, d_rot = self._goal_distance(achieved_goal[:, :7], desired_goal[:, :7])
                d_rot = abs(d_rot)
                d_pos = abs(d_pos)
                success = np.where(np.logical_and(d_pos < self.d_threshold, d_rot < self.angle_threshold), True, False)
            else:
                raise ValueError("Unsupported array shape.")
            return success

        def _is_success_radi(self, achieved_goal, desired_goal):
            """
            Return:
                success: bool
                d_radi: [left, right]
                r_pos: int
                d_rot: int (rad)
            """
            assert achieved_goal.shape == desired_goal.shape, \
                f"Achieved goal and desired goal might have different shape of {achieved_goal.shape, desired_goal.shape}"
            if len(achieved_goal.shape) == 1:
                '''check if radi is success for single step'''
                assert len(achieved_goal) == 11, f"Wrong size, check: {np.shape(achieved_goal)}"  # 11 element
                d_radi = achieved_goal[7:9] - desired_goal[7:9]
                d_pos, d_rot = self._goal_distance(achieved_goal[:7],desired_goal[:7])
                d_rot = abs(d_rot)
                d_pos = abs(d_pos)
                success = (d_pos < self.d_threshold and d_rot < self.d_threshold).astype(np.float32)
            elif len(achieved_goal.shape) > 1:
                # print("Used by replay buffer")
                # ValueError("More than one achieved goal. Might be needed by replay buffer")
                '''train'''
                assert achieved_goal.shape[1] == 11, f"Wrong size, check: {np.shape(achieved_goal)}"  # 11 element
                d_radi = achieved_goal[:, 7:9] - desired_goal[:, 7:9]
                d_pos, d_rot = self._goal_distance(achieved_goal[:, :7], desired_goal[:, :7])
                d_rot = abs(d_rot)
                d_pos = abs(d_pos)
                success = np.where(np.logical_and(d_pos < self.d_threshold, d_rot < self.angle_threshold), 1, 0)
            else:
                raise ValueError("Unsupported array shape.")
            return success, d_radi, d_rot, d_pos

        def _goal_distance(self, goal_a, goal_b):
            ''' get pos difference and rotation difference
            left motor pos: 0.037012 -0.1845 0.002
            right motor pos: -0.037488 -0.1845 0.002
            '''
            assert goal_a.shape == goal_b.shape, f"Check: {goal_a.shape}, {goal_b.shape}"
            assert goal_a.shape[-1] == 7
            goal_a[2] = goal_b[2]

            d_pos = np.zeros_like(goal_a[..., 0])

            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            euler_a = rotations.quat2euler(quat_a)
            euler_b = rotations.quat2euler(quat_b)
            if euler_a.ndim == 1:
                euler_a = euler_a[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
            if euler_b.ndim == 1:
                euler_b = euler_b[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
            euler_a[:,:2] = euler_b[:,:2]  # make the second and third term of euler angle the same
            quat_a = rotations.euler2quat(euler_a)
            quat_a = quat_a.reshape(quat_b.shape)

            # print(quat_a, quat_b)
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
            d_rot = angle_diff
            # assert d_pos.shape == d_rot.shape
            return d_pos, d_rot
            # return d_pos

        def _compute_radi(self, a):
            # left motor pos: 0.037012 -0.1845 0.002
            # right motor pos: -0.037488 -0.1845 0.002
            a[2] = 0.002

            left_motor = [0.037012,-0.1845,0.002]
            right_motor = [-0.037488,-0.1845,0.002]

            assert a.shape[-1] == 7

            radius_al = np.zeros_like(a[..., 0])
            radius_ar = np.zeros_like(a[..., 0])

            delta_r_a_left_motor = a[..., :3] - left_motor # pos of motor left
            delta_r_a_right_motor = a[..., :3] - right_motor # pos of motor right
            radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)
            radius_ar = np.linalg.norm(delta_r_a_right_motor, axis=-1)

            return radius_al,radius_ar

        def add_terminate_penalty(self, achieved_goal):
            """
            Out of range penalty
            """
            # print("shape: ", achieved_goal.shape)
            if len(achieved_goal.shape) == 1:
                assert len(achieved_goal) == 11, f"Wrong achieved goal shape, check: {achieved_goal}"
                penalty = 0
                radius_l = achieved_goal[7]
                radius_r = achieved_goal[8]
                if radius_l == 0 or radius_r == 0:
                    pass
                elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (
                not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):
                    penalty = -5
            elif len(achieved_goal.shape) > 1:
                assert achieved_goal.shape[1] == 11, f"Wrong achieved goal shape, check: {achieved_goal}"
                penalty = np.zeros(achieved_goal.shape[0])
                for idx in range(achieved_goal.shape[0]):
                    radius_l = achieved_goal[idx, 7]
                    radius_r = achieved_goal[idx, 8]
                    if radius_l == 0 or radius_r == 0:
                        pass
                    elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (
                            not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):
                        penalty[idx] = -5
            else:
                raise ValueError("Unsupported achieved_goal shape.")

            return penalty

        def compute_reward(self, achieved_goal, goal, info):
            '''
            0: default
            1: smaller success reward
            2: without success reward
            '''
            reward_option = 0
            if reward_option == 0:
                success_reward_scale = 1
            elif reward_option == 1:
                success_reward_scale = 0.5
            else:
                assert reward_option == 2, f"Wrong reward option, check: {reward_option}"
                success_reward_scale = 0

            self.reward_type = "dense"
            # print("reward_type:",self.reward_type)
            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success_radi(achieved_goal, goal)
                print("sparse")
                return success.astype(np.float32) - 1.0
            else:
                if len(goal) == 11:
                    success, d_radi, d_rot, d_pos = self._is_success_radi(achieved_goal, goal)
                else:
                    assert goal.shape == (2048, 11) or goal.shape == (256, 11), f"The shape of goal is wrong, check: {goal.shape}"
                    success, d_radi, d_rot, d_pos = self._is_success_radi(achieved_goal, goal)
                    penalty = self.add_terminate_penalty(achieved_goal)
                    reward_dict = {
                        # "RL_IHM": - abs(d_radi) * 20 + penalty + success_radi * success_reward_scale,
                        "RL_inspired_IHM_with_RL_Friction": None,
                        "d_radi_seperate": d_radi * 20,
                        "action_complete": self.check_action_complete(),
                        "d_radi": d_radi * 20,
                        "d_pos": d_pos * 20,
                        "pos_control_position": self.data.qpos[self.pos_idx * 2 + 1],
                        "torque_control_position": self.data.qpos[(1-self.pos_idx) * 2 + 1],
                        'E2E_IHM': self.compute_reward_E2E_buffer(achieved_goal, goal),
                    }
                    return reward_dict

                ''' This part produce reward for friction change '''
                if self.last_angle is None:
                    self.last_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
                    self.last_angle = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4]
                    self.start_height = self.last_height
                    self.start_radi = np.mean(abs(d_radi))
                    self.last_r_diff = 0
                    self.friction_change_error = 0
                    self.slip_error = 0
                    self.slip_error_angle = 0

                else:
                    r_diff = abs(self.start_radi - np.mean(abs(d_radi)))
                    z = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
                    current_angle = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4]

                    self.friction_change_error = abs(self.last_r_diff - r_diff)
                    self.slip_error = abs(z - self.last_height)
                    self.slip_error_angle = abs(current_angle - self.last_angle)

                    self.last_r_diff = r_diff
                    self.last_height = z
                    self.last_angle = current_angle

                assert np.all(self.goal_radi) != 0, f"Wrong goal radi: {self.goal_radi}"
                reward_dict = {
                    "RL_IHM": None,
                    "RL_inspired_IHM_with_RL_Friction": None,
                    "d_radi_seperate": d_radi * 20,
                    "action_complete": self.check_action_complete(),
                    "d_radi": np.mean(abs(d_radi)) * 20,
                    "d_pos": d_pos * 20,
                    "pos_control_position": self.data.qpos[self.pos_idx*2+1],
                    "torque_control_position": self.data.qpos[(1 - self.pos_idx) * 2 + 1],
                    'current_goal_centre_distance': None,
                    'goal_pose': None,
                    'E2E_IHM': None,
                    'desired_goal_contact_point_radi': self.goal_radi,
                    'achieved_goal_contact_point_radi': self._get_achieved_goal_contact_point()
                }

                success = 1 if self.success else -1
                penalty = self.add_terminate_penalty(achieved_goal=achieved_goal)
                reward_dict['goal_pose'] = achieved_goal[:7]
                reward_dict['E2E_IHM'] = self.compute_reward_E2E(achieved_goal, goal)
                distance = self._goal_distance(achieved_goal[:7], goal[:7])[0]
                reward_dict['current_goal_centre_distance'] = distance

                'RL-based IHM'
                'without support'
                if self.pick_up_height == 3:
                    print("pick up height is 3")
                    reward_dict["RL_IHM"] = - np.mean(abs(d_radi)) * 20 - self.slip_error * 20000 - self.slip_error_angle * 20000 + penalty
                else:
                    'with support'
                    assert self.pick_up_height == 0
                    reward_dict["RL_IHM"] = - np.mean(abs(d_radi)) * 20 + success * success_reward_scale + penalty

                'RL_based_Friction'
                'without support'
                if self.pick_up_height == 3:
                    reward_dict["RL_inspired_IHM_with_RL_Friction"] = - self.friction_change_error * 20000 - self.slip_error * 20000 - self.slip_error_angle * 20000 + penalty
                else:
                    'with support'
                    assert self.pick_up_height == 0
                    reward_dict["RL_inspired_IHM_with_RL_Friction"] = - self.friction_change_error * 20000 + penalty

                return reward_dict

        def normalize(self, data, mean, std):
            std = np.where(std == 0, 1, std)
            return (data-mean)/std

        def compute_reward_E2E_buffer(self, achieved_goals, goals):
            reward = []
            for i, achieved_goal in enumerate(achieved_goals):
                reward.append(self.compute_reward_E2E(achieved_goal, goals[i]))

            assert np.array(reward).shape[0] == 2048 or np.array(reward).shape[0] == 256, f"Wrong shape, check: {np.shape(reward)}"
            return np.array(reward)

        def compute_reward_E2E(self, achieved_goal, goal):
            """
            Range:
            d_rot_norm: 0 ~ -1
            penalty: 0, -5
            success_term: -0.5, 5
            d_radi_rms: 0 ~ 0.9

            weighting term:
            w1: 0.7
            w2: 0.7

            sliding: radi diff + w1 * orientation diff
            rotation: w1 * radi diff + orientation diff
            same action: w2 * sliding, w2 * rotation
            """

            success, d_radi, d_rot, d_pos = self._is_success_radi(achieved_goal, goal)
            # print("Angle: ", d_rot)
            d_rot_norm = self.normalize(d_rot, 0, np.pi)  # 0 ~ -1
            penalty = self.add_terminate_penalty(achieved_goal=achieved_goal)  # 0, -5
            success_term = 5 if success else -0.5  # -1, 5

            d_radi_rms = np.sqrt(np.mean([d_radi[0] ** 2, d_radi[1] ** 2])) * 35  # 0 ~ -0.9
            # print(d_radi_rms, d_rot, d_rot_norm)
            w1 = 0.7  # care more about sliding
            w2 = 1  # weight
            w3 = 0.7  # rotation
            if achieved_goal[-2] == 1 and achieved_goal[-1] == 1:
                # friction change to sliding
                reward = - ( d_radi_rms + w1 * d_rot_norm)  # when d_radi_rms gets very large, this is the best option
            elif achieved_goal[-2] == 1 and achieved_goal[-1] == 0:
                # friction change to rotation
                reward = - (w3 * d_radi_rms + d_rot_norm)  # when d_rot_norm gets very large, this is the best option
            elif achieved_goal[-2] == 0 and achieved_goal[-1] == 1:
                # continue sliding
                reward = - w2 * (d_radi_rms + w1 * d_rot_norm)
            elif achieved_goal[-2] == 0 and achieved_goal[-1] == 0:
                # continue rotation
                reward = - w2 * (w3 * d_radi_rms + d_rot_norm)
            # print("Reward: ", reward, d_radi_rms, d_rot_norm)
            reward += success_term + penalty
            # print(d_radi_rms, d_radi[0], d_radi[1])
            # print(w1 * d_radi_rms + d_rot_norm, d_radi_rms, d_rot_norm)
            # print("Chekc: ", d_radi_rms/d_rot_norm, reward)

            return reward

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        '''7 position element of object + 2 radius'''
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "joint:object")
        assert object_qpos.shape == (7,)

        ''' RANDOMISATION '''
        achieved_goal_radi = self.compute_goal_radi(object_qpos[:3], object_qpos[:3])

        # print(object_qpos)
        achieved_goal = np.concatenate((object_qpos, achieved_goal_radi))
        assert achieved_goal.shape == (9,)
        return achieved_goal

    def _get_achieved_goal_contact_point(self):
        '''7 position element of object + 2 radius'''
        if self.left_contact_idx != None and self.right_contact_idx != None:
            left_contact_point = self.data.site_xpos[self.left_contact_idx]
            right_contact_point = self.data.site_xpos[self.right_contact_idx]
            ''' RANDOMISATION '''
            achieved_goal_radi = self.compute_goal_radi(left_contact_point, right_contact_point)
        else:
            achieved_goal_radi = [0,0]
        return achieved_goal_radi

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        # self.check_physics_parameter()
        # print("reset")
        '''self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        '''
        # self.action_count = 0;
        self.goal_radi = np.zeros(2)
        self.switchFriction_count = 0;
        # self.data.ctrl = 0
        self.count = 0
        self.last_motor_diff = 0
        self.torque_high_indicator = 0
        self.terminate_count = 0
        self.stuck_terminate = False
        self.successSlide = False
        self.success = False
        self.left_contact_idx = None;
        self.right_contact_idx = None;
        self.IHM_start = False
        self.friction_changing = False
        self.friction_state = 0
        self.gradients = []

        # if self.last_height is not None:
        #     print("Error of the episode(height & radi): ", abs(self.last_height - self.start_height), abs(self.last_r_diff))

        self.friction_change_error = 0
        self.slip_error = 0
        self.slip_error_angle = None
        self.last_r_diff = 0
        self.last_height = None
        self.last_angle = None
        self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
        self.start_radi = None
        self.switch_ctrl_type_pos_idx(0)

        ' Randomisation '
        # self.get_pos_ctrl_params()
        # self.get_torque_ctrl_params()
        # self.domain_randomise.randomise_physics_params()
        self.randomise_physics_params()

        if self.slip_terminate == True or self.reset_everything is True:
            # print("\033[92m reset everything \033[0m")
            self.reset_ctrl_type()
            self.pick_up = False
            self.closing = False
            self.slip_terminate = False

            self.data.ctrl[:] = 0

            self.data.time = self.initial_time
            self.data.qpos[:] = np.copy(self.initial_qpos)
            self.data.qvel[:] = np.copy(self.initial_qvel)

            self.torque_ctrl = 1

            if self.model.na != 0:
                self.data.act[:] = None

            self._mujoco.mj_forward(self.model, self.data)
            initial_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "joint:object"
            ).copy()
            initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
            assert initial_qpos.shape == (7,)
            assert initial_pos.shape == (3,)
            assert initial_quat.shape == (4,)
            initial_qpos = None

            if self.randomize_initial_rotation:
                if self.target_rotation == "z":
                    angle = self.np_random.uniform(-np.pi, np.pi)
                    axis = np.array([0.0, 0.0, 1.0])
                    offset_quat = quat_from_angle_and_axis(angle, axis)
                    initial_quat = rotations.quat_mul(initial_quat, offset_quat)
                elif self.target_rotation == "fixed":
                    pass
                else:
                    raise error.Error(
                        f'Unknown target_rotation option "{self.target_rotation}".'
                    )

            # Randomize initial position.
            if self.randomize_initial_position:
                if self.target_position != "fixed":
                    # initial_pos += self.np_random.normal(size=3, scale=0.005)
                    initial_pos = self._sample_coord(0)

            # finalise initial pose
            initial_quat /= np.linalg.norm(initial_quat)
            initial_qpos = np.concatenate([initial_pos, initial_quat])

            self._utils.set_joint_qpos(self.model, self.data, "joint:object", initial_qpos)


        '''
        self.torque_ctrl = 1
        
        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        initial_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "joint:object"
        ).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        # 注意: 每次都需要升到固定高度,需要给z加一个offset
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f'Unknown target_rotation option "{self.target_rotation}".'
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                # initial_pos += self.np_random.normal(size=3, scale=0.005)
                initial_pos = self._sample_coord(0)

        # finalise initial pose
        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "joint:object", initial_qpos)

        # def is_on_palm():
        #     self._mujoco.mj_forward(self.model, self.data)
        #     cube_middle_idx = self._model_names._site_name2id["object:center"]
        #     cube_middle_pos = self.data.site_xpos[cube_middle_idx]
        #     is_on_palm = cube_middle_pos[2] > 0.04
        #     return is_on_palm
        
        '''

        # Run the simulation for a bunch of timesteps to let everything settle in.
        if self.firstEpisode:
            for _ in range(10):
                self._set_action(np.array([0,0,False]))
                self.pos_idx = 0
                try:
                    self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
                except Exception:
                    return False
            self.firstEpisode = False

        # print("----------------")
        # self.check_physics_parameter()
        return True

    def reset_ctrl_type(self):
        self.switch_ctrl_type_pos_idx(0)


    def _sample_coord(self, z):
        # TODO: Check this, the y keeps being -0.3
        x_range = [-0.04, 0.04]
        x = self.np_random.uniform(x_range[0], x_range[1])
        y_range = [-0.24, 1.625 * abs(x) - 0.315]
        y = self.np_random.uniform(y_range[1], y_range[0])
        coord = [x, np.clip(y - 0.02, -0.24, -0.30), z]
        return coord

    def _sample_goal(self):
        # Select a goal for the object position.
        ''' this random was set to add offset to x,y,z, but now it will only add offset to x,y '''
        target_pos = None
        if self.target_position == "random":
            # z = 0.10
            # z = 0
            if self.pick_up_height == 3:
                z = 0.10
            else:
                z = 0
            target_pos = self._sample_coord(z)
            target_pos = np.array(target_pos, dtype=np.float32)
        elif self.target_position in "fixed":
            target_pos = [0.02, -0.26, 0]
            target_pos = np.array(target_pos, dtype=np.float32)
        else:
            raise error.Error(
                f'Unknown target_position option "{self.target_position}".'
            )
        assert target_pos is not None
        assert target_pos.shape == (3,)
        # print("target_pos",target_pos)

        '''Select a goal for the object rotation.'''
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi / 4, np.pi / 4)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in "fixed":
            angle = 0
            # target_quat = self.data.get_joint_qpos("object:joint")
            target_quat = self._utils.get_joint_qpos(
                self.model, self.data, "joint:target"
            ).copy()[3:]
        else:
            raise error.Error(f'Unknown target_rotation option "{self.target_rotation}".')

        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        # goal_radi = self.find_goal_radi(target_pos, angle)

        # convert goal coordinate to goal radisu
        '''3+4+2
        4 is needed to set the target position as joint
        '''
        # TODO: Uncomment below to use pre-sampled slide-end goals
        # print("target pos: ", target_pos)
        # print("Use slide end goal for E2E policy")
        # mid_goals = load_pickle("/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/demonstration/bigSteps_10000demos_middle_goal")
        # # mid_goals = load_pickle("/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/demonstration/bigSteps_10000demos_end_goal")
        # target_pos = mid_goals[self.idx][:3]
        # target_quat = mid_goals[self.idx][3:]
        # self.idx += 1

        goal_radi = self.compute_goal_radi(target_pos, target_pos)

        goal = np.concatenate((target_pos, target_quat, goal_radi))
        assert goal.shape == (9,)

 #        goal = np.array([-3.67611821e-02, -2.99649019e-01,  2.00000000e-03,  9.57776145e-01,
 # -8.36114126e-05,  1.23010103e-04 , 2.87514930e-01,  1.36754448e-01,
 #  1.15151313e-01])

        return goal

    def _sample_goal_fixed(self):
        pose = np.array([ 2.1969790e-02,-2.9955113e-01, 0, 0.9063078, 0, 0, 0.4226183])
        goal_radi = self.compute_goal_radi(pose[:3], pose[:3])
        goal = np.concatenate([pose, goal_radi])
        assert len(goal) == 9, f"Wrong goal size, check: {goal}"
        return goal

    def _corner_gradient(self):
        """
        Gradient contains the gradient of
        corners1 with corner2
        corners2 with corner3
        ...
        cornern with corner1
        """
        contact_coord = []
        for num in range(4):
            contact_idx = self._model_names._site_name2id[f"target:corner{num + 1}"]
            contact_coord.append(self.data.site_xpos[contact_idx])
        contact_coord = np.array(contact_coord)
        for i in range(len(contact_coord)):
            next_idx = (i + 1) % len(contact_coord)  # Ensure the last and first coordinates are considered adjacent
            gradient = contact_coord[next_idx][:2] - contact_coord[i][:2]  # Only x, y coordinates
            self.gradients.append(gradient)

    def _get_contact_point(self):
        contact_coord = []
        for num in range(self.number_of_corners):
            contact_idx = self._model_names._site_name2id[f"target:corner{num + 1}"]
            contact_coord.append(self.data.site_xpos[contact_idx])
        # print("contact point: ", contact_coord)
            # print("contact:", contact_idx, self.data.site_xpos[contact_idx])
        left_index, left_contact = max(enumerate(contact_coord), key=lambda coord: coord[1][0])
        right_index, right_contact = min(enumerate(contact_coord), key=lambda coord: coord[1][0])
        # print("left contact: ", left_contact, right_contact)

        # print("target position: ", self._utils.get_joint_qpos(self.model, self.data, "joint:target"))

        '''sim '''
        self.left_contact_idx = self._model_names._site_name2id[f"object:corner{left_index + 1}"]
        self.right_contact_idx = self._model_names._site_name2id[f"object:corner{right_index + 1}"]

        '''real '''
        self.left_contact_idx_real = left_index
        self.right_contact_idx_real = right_index
        #
        # self.left_contact_idx = left_index
        # self.right_contact_idx = right_index


        # print("left index of target and object:", left_index, self.left_contact_idx, right_index,
        #       self.right_contact_idx)
        # print("left index of target and object:", left_index, self.left_contact_idx)

        left_contact[2] = 0.025
        left_contact_coord = np.concatenate((left_contact, [0, 0, 0, 0]))
        right_contact[2] = 0.025
        right_contact_coord = np.concatenate((right_contact, [0, 0, 0, 0]))

        self.goal_radi = self.compute_goal_radi(left_contact_coord[:3], right_contact_coord[:3])

        self._utils.set_joint_qpos(self.model, self.data, "site-checker", right_contact_coord)
        self._utils.set_joint_qvel(self.model, self.data, "site-checker", np.zeros(6))

        # print("goal_radi: ", self.goal_radi)

    def compute_goal_radi(self,a, b, left_motor=[0.037012, -0.1845, 0.002], right_motor=[-0.037488, -0.1845, 0.002]):
        '''
        a is the left contact point,
        b is the right contact point
        '''
        # print("compute: ", a, b)
        a[2] = 0.002
        b[2] = 0.002

        # left_motor = [0.037012, -0.1845, 0.002]
        # right_motor = [-0.037488, -0.1845, 0.002]

        # print("Check radius: ", a, b, left_motor, right_motor)
        assert left_motor[2] == 0.002 and right_motor[2] == 0.002, f"Wrong motor height, check: {left_motor[2], right_motor[2]}"

        assert a.shape == b.shape
        assert a.shape[-1] == 3

        radius_al = np.zeros_like(a[..., 0])
        radius_br = np.zeros_like(b[..., 0])

        delta_r_a_left_motor = a[..., :3] - left_motor  # pos of motor left
        radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)

        delta_r_b_right_motor = b[..., :3] - right_motor  # pos of motor right
        radius_br = np.linalg.norm(delta_r_b_right_motor, axis=-1)

        goal_radi = [radius_al,radius_br]

        return goal_radi

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        # assert goal.shape == (11,)

        self._utils.set_joint_qpos(self.model, self.data, "joint:target", goal[:7])
        self._utils.set_joint_qvel(self.model, self.data, "joint:target", np.zeros(6))

        # print("test complete")

        self._mujoco.mj_forward(self.model, self.data)

        self._get_contact_point()
        # self.goal[7:] = self.goal_radi

    def _get_obs(self):

        # print("Check")

        env_qpos = self.data.qpos
        env_qvel = self.data.qvel

        robot_qpos = env_qpos[1:5]
        robot_qvel = env_qvel[1:5]
        object_qpos = env_qpos[5:12]  # pos + quat (3+4)
        object_qvel = env_qvel[5:11]  # linear velocity + angular velocity (3+3)

        self._corner_gradient()

        # simplify observation space
        ''' RANDOMISATION '''
        robot_qpos, robot_qvel = self.randomise_joint(robot_qpos, robot_qvel)

        '''object information: current pose + object centre to two motor radius (7 + 2) '''
        achieved_goal = self._get_achieved_goal().ravel()
        desired_goal = self.goal.ravel().copy()

        assert len(achieved_goal)==9, f"Wrong length: {achieved_goal}"
        achieved_goal = np.append(achieved_goal, int(self.friction_change_penalty))
        achieved_goal = np.append(achieved_goal, int(self.sliding))
        # print("achieved goal: ", achieved_goal)

        '''two finger, two finger insert'''
        assert robot_qpos.shape == (4,)

        '''Get friction state'''
        if len(self.goal.ravel()) == 9:
            # print("Check: ", self.friction_change_penalty, achieved_goal)
            desired_goal = np.append(desired_goal, int(self.friction_change_penalty))
            desired_goal = np.append(desired_goal, int(self.sliding))  # indicate doing sliding
            # assert len(self.goal.ravel()) == 11, f"Wrong lenght, check: {self.goal.ravel()}"
            self.goal = desired_goal
        elif len(self.goal.ravel()) == 0:
            radi_diff = np.zeros(2)
            angle_diff = 0
        else:
            assert len(self.goal.ravel()) == 11 and len(achieved_goal) == 11, f"Wrong goal length, check: {self.goal.ravel()}"

        '''Calculate goal and achieved goal's radi and angle difference'''
        if len(self.goal.ravel()) == 11:
            radi_diff = self.goal.ravel()[7:9] - achieved_goal[7:9]
            m_distance_diff, angle_diff = self._goal_distance(self.goal.ravel()[:7], achieved_goal[:7])
        else:
            assert len(self.goal.ravel()) == 0, f"Wrong length, check: {self.goal.ravel()}"

        '''Get obs'''
        observation = np.concatenate(
            [
                robot_qpos, # (4, )
                robot_qvel, # (4, )
                object_qpos, # (7, )
                object_qvel, # (6, )
                # the following 2 terms are the difference between desired goal and achieved goal
                radi_diff, # (2, ) radius difference (radi = object centre to actuators)
                np.array([angle_diff]), # (1, ) orientation difference
            ]
        )

        complete_obs = {
            "observation": observation.copy(),  # (24, )
            # (11, ) pose (7,) + achieved radi (2, ) + change friction (1, ) + action is sliding (1, )
            "achieved_goal": achieved_goal.copy(),
            # (11, ) pose (7,) + desired radi (2, ) + change friction (1, ) + action is sliding (1, )
            "desired_goal": desired_goal.copy(),
        }

        self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
        new_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
        if new_height > self.start_height:
            self.start_height = new_height.copy()
        if self.pick_up_height == 3:
            obs = {
                "observation": complete_obs["observation"].copy(),
                "achieved_goal": np.concatenate([complete_obs["achieved_goal"], [current_height]]),
                "desired_goal": np.concatenate([complete_obs["desired_goal"], [self.start_height]])
            }
        else:
            assert self.pick_up_height == 0
            obs = complete_obs

        assert len(obs['observation']) == 24, f"Check: {obs['observation']}"  # 4 + 4 + 7 + 6 + 2 + 1 = 24
        assert len(obs['achieved_goal']) == 11, f"Check: {obs['achieved_goal']}"  # 7 + 2 + 2 = 11
        assert len(obs['desired_goal']) == 11 or len(obs['desired_goal']) == 0, f"Check: {obs['desired_goal']}"  # 7 + 2 + 2 = 11

        return obs

    def compute_terminated(self, achieved_goal, desired_goal, info):
        # exceed range
        assert len(achieved_goal) == 11, f"achieved goal should have length of 11, check: {achieved_goal}"
        radius_l = abs(achieved_goal[7])
        radius_r = abs(achieved_goal[8])
        # print(radius_r, radius_l)

        if np.any(achieved_goal) == 0:
            print("\033[91m| Empty Achieved Goal \033[0m")
            return False

            '''Stuck indicator'''
        # elif self.stuck_terminate == True:
        #     self.slip_terminate = True
        #     print("\033[91m| Terminate: stuck \033[0m")
        #     # print("------------------------------------")
        #     # print("------------------------------------")
        #     return True

            '''out of operation range'''
        elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):
            self.slip_terminate = True
            print("terminate: out of range", radius_l, radius_r)
            # print("------------------------------------")
            # print("------------------------------------")
            return True

            '''Success terminate'''
        # elif self.success:
        #     print("success")
        #     self.data.ctrl[2] = self.data.qpos[2]
        #     self.data.ctrl[3] = self.data.qpos[4]
        #     print("------------------------------------")
        #     print("------------------------------------")
        #     return True

            '''Pos slip'''
        # elif self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2] < 0.092 and self.data.qpos[0] >= 0.1 and self.pick_up_height == 3:
        #     self.slip_terminate = True
        #     print("terminate: pos slip with error of (limit - 0.092): ", abs(self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]))
        #     # print(self.start_height - achieved_goal[2], self.start_height, z)
        #     # print("------------------------------------")
        #     # print("------------------------------------")
        #     # print("------------------------------------")
        #     # print("------------------------------------")
        #     # print("------------------------------------")
        #     return True

            '''Angle slip'''
        # elif self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4] > 0.1:
        #     self.slip_terminate = True
        #     print("terminate: angle slip with error of (limit - 0.1): ", self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4])
        #     return True

        else:
            """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
            return False

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None