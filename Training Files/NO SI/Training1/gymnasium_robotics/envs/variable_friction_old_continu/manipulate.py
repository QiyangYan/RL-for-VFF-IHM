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

from gymnasium_robotics.envs.variable_friction_friction import MujocoHandEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation

import xml.etree.ElementTree as ET
import os

from gymnasium_robotics.envs.plot_array import plot_numbers, plot_numbers_two_value

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
            self.r_threshold = 0.002
            self.d_threshold = 0.01
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
            self.switchFriction_count = 0;
            self.terminate_r_limit = [0.055,0.14]
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


        def _is_success_radi(self,achieved_goal,desired_goal):
            '''find the actual goal, 3 = 2+1'''
            # print("object:target_corner1",self.model.site("object:target_corner1").pos)  # before get new goal
            # print("_is_success_radi")
            # d_radi, r_left, r_right = self._radi_error(achieved_goal[:7], desired_goal[:7])
            if len(achieved_goal.shape) == 1:
                '''each step'''
                # print("each step")
                # desired_goal = desired_goal[:8]
                if self.friction_changing == True:
                    d_radi = abs(achieved_goal[7:] - desired_goal[7:])
                    success_pos = 1
                    d_pos = 0
                else:
                    d_radi = abs(achieved_goal[7:] - desired_goal[7:])
                    # print("check achieved:", achieved_goal[-2:])
                    d_pos = self._goal_distance(achieved_goal[:7],desired_goal[:7])
                    success_pos = (d_pos < self.d_threshold).astype(np.float32)

                d_radi_mean = np.mean(d_radi)
                success_radi = (np.mean(abs(d_radi_mean)) < self.r_threshold).astype(np.float32)
                # print("success_radi: ", d_radi_mean, d_radi, self.r_threshold)

            elif len(achieved_goal.shape) > 1:
                '''train'''
                # print("train")
                # desired_goal = desired_goal[:,:8]
                # print("achieved:",achieved_goal[:,7:])
                # print("desired:",desired_goal[:,7:])
                # print("self.friction_changing: ", self.friction_changing)
                if self.friction_changing == True:
                    d_radi = abs(achieved_goal[:, :2] - desired_goal[:, :2])
                    d_pos = 0
                    success_pos = 1
                else:
                    d_radi = abs(achieved_goal[:,7:] - desired_goal[:,7:])
                    # print("d_radi:", d_radi)
                    # print(achieved_goal[:,:7].shape,desired_goal[:,:7].shape)
                    d_pos = self._goal_distance(achieved_goal[:,:7],desired_goal[:,:7])
                    success_pos = np.where(d_pos < self.d_threshold, 1, 0)

                d_radi_mean = np.mean(d_radi, axis=1).reshape(-1, 1)
                # print("train:",d_radi_mean)
                d_radi_mean = np.where(d_radi_mean == 0, 1, d_radi_mean)
                success_radi = np.where(d_radi_mean < self.r_threshold, 1, 0)
                # print("success_radi: ", d_radi_mean, self.r_threshold)

            else:
                raise ValueError("Unsupported array shape.")

            return success_radi, success_pos, d_radi, d_pos

        def _goal_distance(self, goal_a, goal_b):
            ''' get pos difference and rotation difference
            left motor pos: 0.037012 -0.1845 0.002
            right motor pos: -0.037488 -0.1845 0.002
            '''
            assert goal_a.shape == goal_b.shape
            assert goal_a.shape[-1] == 7
            goal_a[2] = goal_b[2]

            d_pos = np.zeros_like(goal_a[..., 0])

            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*

            # assert d_pos.shape == d_rot.shape
            # return d_pos, d_rot
            return d_pos

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

        def compute_reward_discrete(self, achieved_goal, goal, info):
            success_radi, success_pos, d_radi, d_pos = self._is_success_radi(achieved_goal, goal)


        def compute_reward(self, achieved_goal, goal, info):
            self.reward_type = "dense"
            # print("reward_type:",self.reward_type)
            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success_radi(achieved_goal, goal)
                print("sparse")
                return success.astype(np.float32) - 1.0
            else:

                # print("The goals for the training: ", achieved_goal, goal)
                success_radi, success_pos, d_radi, d_pos = self._is_success_radi(achieved_goal, goal)

                ''' with reference to last time step '''
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


                if self.step_count_for_friction_change > 60:
                    stuck = -1
                else:
                    stuck = 0

                ''' friction_change_reward '''
                # print("error: ", self.friction_change_error, self.slip_error, self.slip_error_angle)
                # print("error reward: ", - self.friction_change_error * 20000 - self.slip_error * 20000 + stuck - self.slip_error_angle * 20000)

                # print("check: ", d_radi)

                'without support'
                if self.IHM_RL:
                    if self.pick_up_height == 3:
                        reward = - np.mean(abs(d_radi)) * 20 - self.slip_error * 20000 + stuck - self.slip_error_angle * 20000
                        return reward
                    else:
                        # print("check", d_radi, achieved_goal)
                        reward = - np.mean(abs(d_radi)) * 20 + success_radi * 2
                        return reward
                else:
                    if self.pick_up_height == 3:
                        reward = - self.friction_change_error * 20000 - self.slip_error * 20000 + stuck - self.slip_error_angle * 20000
                        self.reward_history.append(reward)
                        return reward
                    else:
                        'with support'
                        assert self.pick_up_height == 0
                        reward = - self.friction_change_error * 20000
                        if reward >= -0.5:
                            reward = 0
                        self.reward_history.append(reward)
                        return reward

                '''Add more reward term'''
                # slip penalty
                # d_slip, drop = self._slip_indicator(achieved_goal)  # d_slip is negative value

                # same action reward
                # if not self.same_friction and not self.same_motor_direction:
                #     d_action = -1
                #     self.switchFriction_count += 1
                # elif not self.same_friction or not self.same_motor_direction:
                #     self.switchFriction_count += 1
                #     d_action = -0.5
                # else:
                #     d_action = 0
                #
                # if self.switchFriction_count < 7:
                #     d_action = 0


        def compute_reward_slide(self, achieved_goal, goal, info):
            self.friction_steps = 0
            self.reward_type = "dense"
            # print("reward_type:",self.reward_type)

            if self.reward_history != []:
                if self.friction_change_count % 200 == 0 or self.friction_change_count == 1:
                    print("plot---------------------------------------------------------------------------------------")
                    # plot_numbers(self.reward_history,
                    #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                    #              'reward')
                self.reward_history = []

            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success_radi(achieved_goal, goal)
                print("sparse")
                return success.astype(np.float32) - 1.0
            else:
                if self.last_angle is None:
                    self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]

                # print("The goals for the training: ", achieved_goal, goal)
                # success_radi, success_pos, d_radi, d_pos = self._is_success_radi(achieved_goal, goal)
                # # print("sliding")
                # return {"d_radi": np.mean(abs(d_radi)) * 20, # this is the average of two radi
                #         "d_pos": d_pos, # distance from goal pose
                #         "d_radi_seperate": d_radi * 20, # this is two radi from two actuator
                #        }

                # print("----- achieved goal: ", achieved_goal[-2:])
                success_radi, success_pos, d_radi, d_pos = self._is_success_radi(achieved_goal, goal)
                # print("check d_radi: ", d_radi)
                # print("check 1", d_radi, achieved_goal)
                # print("sliding")
                # return {"d_radi": np.mean(abs(d_radi)) * 20, # this is the average of two radi
                #         "d_pos": d_pos, # distance from goal pose
                #         "d_radi_seperate": d_radi * 20, # this is two radi from two actuator
                #        }
                return - np.mean(abs(d_radi)) * 20 + success_radi * 2

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        '''7 position element of object + 2 radius'''
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "joint:object")
        assert object_qpos.shape == (7,)

        if self.left_contact_idx != None and self.right_contact_idx != None:
            left_contact_point = self.data.site_xpos[self.left_contact_idx]
            right_contact_point = self.data.site_xpos[self.right_contact_idx]
            achieved_goal_radi = self.compute_goal_radi(left_contact_point, right_contact_point)
            # print("contact points:",self.left_contact_idx,self.right_contact_idx, achieved_goal_radi)
        else:
            achieved_goal_radi = [0,0]
            # print("initialising")

        achieved_goal = np.concatenate((object_qpos, achieved_goal_radi))
        # print(achieved_goal)
        assert achieved_goal.shape == (9,)
        return achieved_goal

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self, reset_everything):
        # print("reset")
        '''self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        '''
        # self.action_count = 0;
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

        # if self.last_height is not None:
        #     print("Error of the episode(height & radi): ", abs(self.last_height - self.start_height), abs(self.last_r_diff))

        self.friction_change_error = 0
        self.slip_error = 0
        self.slip_error_angle = None
        self.last_r_diff = 0
        self.last_height = None
        self.last_angle = None
        self.start_height = None
        self.start_radi = None

        if self.slip_terminate == True or reset_everything is True:
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
                self._set_action(np.zeros(2))
                try:
                    self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
                except Exception:
                    return False
            self.firstEpisode = False

        return True

    # def _reset_sim_terminate(self):
    #     # print("reset")
    #     self.data.time = self.initial_time
    #     self.data.qpos[:] = np.copy(self.initial_qpos)
    #     self.data.qvel[:] = np.copy(self.initial_qvel)
    #
    #     # self.action_count = 0;
    #     self.switchFriction_count = 0;
    #     self.pick_up = False
    #     self.closing = False
    #     # self.data.ctrl = 0
    #     self.count = 0
    #     self.last_motor_diff = 0
    #     self.torque_high_indicator = 0
    #     self.terminate_count = 0
    #     self.stuck_terminate = False
    #     self.successSlide = False
    #     self.success = False
    #     self.left_contact_idx = None;
    #     self.right_contact_idx = None;
    #     self.IHM_start = False
    #     self.friction_changing = False
    #
    #     if self.slip_terminate == True:
    #         self.slip_terminate = False
    #
    #         self.data.ctrl[:] = 0
    #
    #         self.data.time = self.initial_time
    #         self.data.qpos[:] = np.copy(self.initial_qpos)
    #         self.data.qvel[:] = np.copy(self.initial_qvel)
    #
    #         self.torque_ctrl = 1
    #
    #         if self.model.na != 0:
    #             self.data.act[:] = None
    #
    #         self._mujoco.mj_forward(self.model, self.data)
    #         initial_qpos = self._utils.get_joint_qpos(
    #             self.model, self.data, "joint:object"
    #         ).copy()
    #         initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
    #         assert initial_qpos.shape == (7,)
    #         assert initial_pos.shape == (3,)
    #         assert initial_quat.shape == (4,)
    #         initial_qpos = None
    #
    #         if self.randomize_initial_rotation:
    #             if self.target_rotation == "z":
    #                 angle = self.np_random.uniform(-np.pi, np.pi)
    #                 axis = np.array([0.0, 0.0, 1.0])
    #                 offset_quat = quat_from_angle_and_axis(angle, axis)
    #                 initial_quat = rotations.quat_mul(initial_quat, offset_quat)
    #             elif self.target_rotation == "fixed":
    #                 pass
    #             else:
    #                 raise error.Error(
    #                     f'Unknown target_rotation option "{self.target_rotation}".'
    #                 )
    #
    #         # Randomize initial position.
    #         if self.randomize_initial_position:
    #             if self.target_position != "fixed":
    #                 # initial_pos += self.np_random.normal(size=3, scale=0.005)
    #                 initial_pos = self._sample_coord(0)
    #
    #         # finalise initial pose
    #         initial_quat /= np.linalg.norm(initial_quat)
    #         initial_qpos = np.concatenate([initial_pos, initial_quat])
    #
    #         self._utils.set_joint_qpos(self.model, self.data, "joint:object", initial_qpos)
    #
    #     '''
    #     self.torque_ctrl = 1
    #
    #     if self.model.na != 0:
    #         self.data.act[:] = None
    #
    #     self._mujoco.mj_forward(self.model, self.data)
    #     initial_qpos = self._utils.get_joint_qpos(
    #         self.model, self.data, "joint:object"
    #     ).copy()
    #     initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
    #     assert initial_qpos.shape == (7,)
    #     assert initial_pos.shape == (3,)
    #     assert initial_quat.shape == (4,)
    #     initial_qpos = None
    #
    #     # Randomization initial rotation.
    #     # 注意: 每次都需要升到固定高度,需要给z加一个offset
    #     if self.randomize_initial_rotation:
    #         if self.target_rotation == "z":
    #             angle = self.np_random.uniform(-np.pi, np.pi)
    #             axis = np.array([0.0, 0.0, 1.0])
    #             offset_quat = quat_from_angle_and_axis(angle, axis)
    #             initial_quat = rotations.quat_mul(initial_quat, offset_quat)
    #         elif self.target_rotation == "fixed":
    #             pass
    #         else:
    #             raise error.Error(
    #                 f'Unknown target_rotation option "{self.target_rotation}".'
    #             )
    #
    #     # Randomize initial position.
    #     if self.randomize_initial_position:
    #         if self.target_position != "fixed":
    #             # initial_pos += self.np_random.normal(size=3, scale=0.005)
    #             initial_pos = self._sample_coord(0)
    #
    #     # finalise initial pose
    #     initial_quat /= np.linalg.norm(initial_quat)
    #     initial_qpos = np.concatenate([initial_pos, initial_quat])
    #
    #     self._utils.set_joint_qpos(self.model, self.data, "joint:object", initial_qpos)
    #
    #     # def is_on_palm():
    #     #     self._mujoco.mj_forward(self.model, self.data)
    #     #     cube_middle_idx = self._model_names._site_name2id["object:center"]
    #     #     cube_middle_pos = self.data.site_xpos[cube_middle_idx]
    #     #     is_on_palm = cube_middle_pos[2] > 0.04
    #     #     return is_on_palm
    #
    #     '''
    #
    #     # Run the simulation for a bunch of timesteps to let everything settle in.
    #     if self.firstEpisode:
    #         for _ in range(10):
    #             self._set_action(np.zeros(2))
    #             try:
    #                 self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
    #             except Exception:
    #                 return False
    #         self.firstEpisode = False
    #
    #     return True


    def _sample_coord(self, z):
        # sample region: a triangle
        # y_range = [-0.25, -0.315]
        # finger_length = [0.0505,0.1305] # this is the length of pad from motor
        # target_r_l = self.np_random.uniform(finger_length[0], finger_length[1])
        # target_r_r = self.np_random.uniform(finger_length[0], finger_length[1])

        # finger_length = np.linalg.norm(delta_pos, axis=-1)
        x_range = [-0.04, 0.04]
        x = self.np_random.uniform(x_range[0], x_range[1])
        y_range = [-0.24, 1.625*abs(x)-0.315]
        y = self.np_random.uniform(y_range[1],y_range[0])
        coord = [x, y-0.02, z]
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
            angle = self.np_random.uniform(-np.pi/4, np.pi/4)
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
        goal_radi = np.zeros(2)

        goal = np.concatenate((target_pos,target_quat,goal_radi))
        assert goal.shape == (9,)

        return goal

    def _get_contact_point(self, goal):
        contact_coord = []
        for num in range(self.number_of_corners):
            contact_idx = self._model_names._site_name2id[f"target:corner{num + 1}"]
            contact_coord.append(self.data.site_xpos[contact_idx])
            # print("contact:", contact_idx, self.data.site_xpos[contact_idx])
        left_index, left_contact = max(enumerate(contact_coord), key=lambda coord: coord[1][0])
        right_index, right_contact = min(enumerate(contact_coord), key=lambda coord: coord[1][0])

        # print("target position: ", self._utils.get_joint_qpos(self.model, self.data, "joint:target"))

        self.left_contact_idx = self._model_names._site_name2id[f"object:corner{left_index + 1}"]
        self.right_contact_idx = self._model_names._site_name2id[f"object:corner{right_index + 1}"]
        # print("left index of target and object:", left_index, self.left_contact_idx)

        left_contact[2] = 0.025
        left_contact_coord = np.concatenate((left_contact, [0, 0, 0, 0]))
        right_contact[2] = 0.025
        right_contact_coord = np.concatenate((right_contact, [0, 0, 0, 0]))

        goal_radi = self.compute_goal_radi(left_contact_coord[:3],right_contact_coord[:3])
        goal[7:] = goal_radi
        assert goal.shape == (9,)

        self._utils.set_joint_qpos(self.model, self.data, "site-checker", right_contact_coord)
        self._utils.set_joint_qvel(self.model, self.data, "site-checker", np.zeros(6))

        return goal

    def compute_goal_radi(self,a, b):
        '''
        a is the left contact point,
        b is the right contact point
        '''
        a[2] = 0.002
        b[2] = 0.002

        left_motor = [0.037012, -0.1845, 0.002]
        right_motor = [-0.037488, -0.1845, 0.002]

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
        assert goal.shape == (9,)

        self._utils.set_joint_qpos(self.model, self.data, "joint:target", goal[:7])
        self._utils.set_joint_qvel(self.model, self.data, "joint:target", np.zeros(6))

        # print("test complete")

        self._mujoco.mj_forward(self.model, self.data)


    def _get_obs_for_multiple_policies(self):
        # what's expect with single joint: (array([], dtype=float64), array([], dtype=float64))
        # what's expected:
        # Position: 4 + 1 slide joints, 2 6DOF free joints, 1 + 4 + 2*7 = 19 element
        # each slide joint has position with 1 element, each free joints has position with 7 elements
        # for free joint, 3 elements are (x,y,z), 4 elements are (x,y,z,w) quaternion
        # Velocity: 4 + 1 slide joints, 2 DOF free joints, 1 + 4 + 2*6 = 17 element
        # Result:
        # pos = [0. 0.0.0.0. 0.-0.251365 0.1.0.0.0. 0.0.0.1.0.0.0.]
        # vel = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        robot_qpos = self.data.qpos
        robot_qvel = self.data.qvel

        # achieved_goal = robot_qpos[4:11]
        object_qvel = robot_qvel[5:11]
        robot_qpos = robot_qpos[1:5]
        robot_qvel = robot_qvel[1:5]

        # simplify observation space

        '''object information: radius to two motor + current coordinate'''
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the current radius to two motor + current coordinate

        '''new observation
            position: 3 slide joints, 2 radius, pos of 6DOF free joint (3 elements)
            velocity: 3 slide joints, vel of 6 DOF free joint (6 elements)
        '''

        '''useful infor:
        robot_joint_pos: 3 element
        achieved-goal: 3 + 2
        '''
        # print(np.array(robot_joint_pos))
        # achieved_goal = [achieved_goal[:3],achieved_goal[7:]]
        # assert achieved_goal.shape == (5,)
        assert robot_qpos.shape == (4,)

        # for observation: 3 + 2 + 3
        observation = np.concatenate(
            [
                robot_qpos,
                # robot_qvel,
                # object_qvel,
                achieved_goal
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def _get_obs(self):

        env_qpos = self.data.qpos
        env_qvel = self.data.qvel

        # achieved_goal = robot_qpos[4:11]
        robot_qpos = env_qpos[1:5]
        robot_qvel = env_qvel[1:5]
        object_qvel = env_qvel[5:11]

        # simplify observation space

        '''object information: radius to two motor + current coordinate'''
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the current radius to two motor + current coordinate

        '''two finger, two finger insert'''
        assert robot_qpos.shape == (4,)

        # for observation: 4 + 4 + 2
        observation = np.concatenate(
            [
                robot_qpos, # 4 element
                robot_qvel, # 4 element
                achieved_goal[-2:] # last two element, radius
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def compute_terminated(self, achieved_goal, desired_goal, info):
        # exceed range
        radius_l, radius_r = self._compute_radi(achieved_goal[:7])
        # print("compute terminated", z)
        # radius_l = achieved_goal[7]
        # radius_r = achieved_goal[8]
        # print(radius_l,radius_r)
        # print("check termination")
        # print("check terminate: ", self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2], self.IHM_start)

        if self.stuck_terminate == True:
            self.slip_terminate = True
            print("\033[91m| Terminate: stuck \033[0m")
            # print("------------------------------------")
            # print("------------------------------------")
            return True
        elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):
            self.slip_terminate = True
            print("terminate: out of range", radius_l, radius_r)
            # print("------------------------------------")
            # print("------------------------------------")
            return True
        elif self.success:
            print("success")
            self.data.ctrl[2] = self.data.qpos[2]
            self.data.ctrl[3] = self.data.qpos[4]
            print("------------------------------------")
            print("------------------------------------")
            return True
        elif self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2] < 0.092 and self.data.qpos[0] >= 0.1 and self.pick_up_height == 3:
            self.slip_terminate = True
            print("terminate: pos slip with error of (limit - 0.092): ", abs(self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]))
            # print(self.start_height - achieved_goal[2], self.start_height, z)
            # print("------------------------------------")
            # print("------------------------------------")
            # print("------------------------------------")
            # print("------------------------------------")
            # print("------------------------------------")
            return True
        elif self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4] > 0.1:
            self.slip_terminate = True
            print("terminate: angle slip with error of (limit - 0.1): ", self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4])
            return True
        else:
            """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
            return False

