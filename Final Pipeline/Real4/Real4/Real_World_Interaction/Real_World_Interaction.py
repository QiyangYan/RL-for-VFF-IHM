import time
import numpy as np
from gymnasium_robotics.dynamixel_driver.bulk_read_write import BULK
from gymnasium_robotics.dynamixel_driver.angle_conversion import AngleConversion
from gymnasium_robotics.envs.real4.manipulate import MujocoManipulateEnv
from gymnasium_robotics.vision.aruco_pose_estimation import ARUCO
from typing import Tuple, List, Dict, Union
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

class RealEnv(BULK, ARUCO):
    def __init__(self, env, display=False):
        BULK.__init__(self)
        ARUCO.__init__(self, quat=True)

        self.end_pos_rotation = None
        self.start_pos_rotation = None
        self.aruco_size = 0.0275
        self.object_size = 0.03
        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]
        self.real_env = env
        self.observation = []
        self.gripper_pos_range = self.MAX_POS[0] - self.MIN_POS[0]
        # self.get_actuator_position()
        self.last_friction_state = 0
        self.AngleConvert = AngleConversion()
        self.pos_idx = 0
        self.gripper_pos_ctrl = 0
        self.start = 0
        self.goal_pos = 0
        self.firstEpisode = True
        self.friction_state = 0
        self.reset_robot_without_obs()
        self.init_camera_for_obs(self.object_size, self.aruco_size, display=display)

    def manual_trajectory(self):
        # Pick up, 1 is torque
        self.xm_posControl(0, 2048, reset=False)
        time.sleep(2)
        self.xm_torque_control(1, 10, reset=True)
        time.sleep(1)

        # Change friction and move to side
        self.xl_posControl(0, 0)
        time.sleep(1)
        self.xm_posControl(0, self.MIN_POS[0])
        time.sleep(1)

        # change mode and move to middle
        self.xl_posControl(0, 1)
        time.sleep(1)
        self.xm_posControl(1, self.packetHandler.read4ByteTxRx(
            self.portHandler, self.DXL_ID_aray[1], self.XM['ADDR_PRO_PRESENT_POSITION'])[0])
        time.sleep(1)
        self.xm_torque_control(0, reset=True)
        self.xl_posControl(1, 0)
        time.sleep(1)
        self.xm_posControl(1, 2048)
        time.sleep(1)
        self.xl_posControl(1, 1)
        time.sleep(1)

    def pick_up_real(self, inAir):
        self.xl_posControl(0, 1)
        self.xl_posControl(1, 1)

        # pick
        ID = 0
        pick = int(self.MIN_POS[ID] - self.AngleConvert.rad_2_xm(0.9556) * (ID * 2 - 1))
        self.xm_posControl(ID, pick)
        time.sleep(2)
        self.xm_torque_control(1 - ID, reset=True, goal_torque=-5)
        time.sleep(2)

        obs = self.get_obs_real()
        return obs

    def switch_control_mode(self, pos_idx):
        """
        This change the pos_idx corresponding finger to position control, and another one to torque control

        Todo:
        1. Check later to see the necessity of change both friction to high before change control mode

        :param pos_idx: right finger = 0, left finger = 1
        :return: None
        """
        # Pick up, 1 is torque
        robot_obs = self.get_obs_dynamixel()
        # print("check: ", robot_obs)
        self.xm_posControl(pos_idx, robot_obs[pos_idx*2], reset=True)
        self.xm_torque_control(1 - pos_idx, reset=True)

    def change_friction_to_low(self, action) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Friction change doesn't change control mode, only change control mode for the subsequent slide

        :param action: [2, friction_state, True]
        :return: full observation

        TODO:
        1. Require switch control mode based on the next action type, not friction state
        This one is based on the friction state, align with simulation, but not good
        2. Complete this using env.step

        """
        friction_state = action[2]
        assert friction_state == 1 or friction_state == -1, f"Friction state is wrong, check: {friction_state}"

        # switch control type
        if friction_state != self.friction_state:
            self.friction_state = friction_state
            pos = self.friction_state_mapping(friction_state)

            # change friction
            self.xl_posControl(0, pos[0])  # left finger
            self.xl_posControl(1, pos[1])  # right finger
            time.sleep(1)

        robot_obs = self.get_obs_dynamixel()
        object_obs = []

        return robot_obs

    def change_friction_to_high(self, action):
        """
        Friction change doesn't change control mode, only change control mode for the subsequent slide

        :param action: [2, friction_state, True]
        :return: full observation
        """
        # change friction
        pos, _ = self.friction_state_mapping(action[2])
        self.xl_posControl(0, pos[0])  # left finger
        self.xl_posControl(1, pos[1])  # right finger
        time.sleep(1)

    def change_friction_real(self, action):
        friction_state = action[1]
        assert friction_state == 1 or friction_state == -1 or friction_state == 0, f"Friction state is wrong, check: {friction_state}"

        # switch control type
        to_high_pos = self.friction_state_mapping(0)
        self.xl_posControl(0, to_high_pos[0])  # left finger
        self.xl_posControl(1, to_high_pos[1])  # right finger
        time.sleep(1)

        if friction_state != 0:
            pos = self.friction_state_mapping(friction_state)

            # change friction
            self.xl_posControl(0, pos[0])  # left finger
            self.xl_posControl(1, pos[1])  # right finger
            time.sleep(1)

        robot_obs = self.get_obs_dynamixel()

        return robot_obs

    def step_real(self, action):
        """
        This is for: IHM with continuous action space in actual robot, replace the elif len(action) == 2
        :param action: [-1, 1]
        :return:
        """
        assert len(action) == 2, f"check action length: {len(action)}, action is: {action}"

        """
        Get obs
        """
        robot_obs = self.get_obs_dynamixel()

        """ 
        Get Ready for the slide
        """
        _, control_mode = self.real_env.discretize_action_to_control_mode(action[1])
        # if step_friction_state != self.real_env.last_friction_state:
        #     self.real_env.friction_change_penalty = True
        # else:
        #     self.real_env.friction_change_penalty = False

        _, pos_idx = self.real_env.action_to_control(control_mode)
        if pos_idx != self.pos_idx:
            self.pos_idx = pos_idx
            self.switch_control_mode(pos_idx)

        """ 
        If control signal is different as the previous control signal, then it means new slide starts, need to reassign 
        the start position
        """
        range = np.sort(np.array([self.MIN_POS[pos_idx], self.MAX_POS[pos_idx]]))

        gripper_action = self.map_policy_to_real_action(action[0])
        if gripper_action != self.gripper_pos_ctrl:
            self.gripper_pos_ctrl = gripper_action  # action is the movement related to the current position
            # self.start = self.real_env.observation[self.pos_idx * 2]
            self.start = robot_obs[self.pos_idx*2]
            self.goal_pos = np.clip(self.start + self.gripper_pos_ctrl*(self.pos_idx*2-1)/7, range[0], range[1])
            # print("Start: ", self.start, self.goal_pos, self.pos_idx, self.gripper_pos_ctrl)
            self.xm_posControl(self.pos_idx, int(self.goal_pos), reset=True)
            self.xm_torque_control(1 - self.pos_idx, reset=True)
        elif self.real_env.firstEpisode:
            pass
        else:
            # print(self.goal_pos)
            self.xm_posControl(self.pos_idx, int(self.goal_pos), reset=False)
            self.xm_torque_control(1 - self.pos_idx, reset=False)

        return self.pos_idx

    def reset_robot(self):
        self.xm_posControl(0, self.MIN_POS[0])
        self.xm_posControl(1, self.MIN_POS[1])
        self.xl_posControl(0, 0)
        self.xl_posControl(1, 0)
        time.sleep(2)
        return self.get_obs_real()

    def reset_robot_without_obs(self):
        self.xm_posControl(0, self.MIN_POS[0])
        self.xm_posControl(1, self.MIN_POS[1])
        self.xl_posControl(0, 0)
        self.xl_posControl(1, 0)
        time.sleep(2)

    def map_policy_to_real_action(self, action_):
        """
        This is: get the amount of movement for relative control
        """
        standarised_action = (action_ + 1) / 2
        gripper_action = standarised_action * self.gripper_pos_range
        return gripper_action

    def get_obs_real(self, display=False):
        """
        This gets the observation based on AruCo and Dynamixel reading.
        Observation should contain as much obtainable information as I can for now.
        :return: complete observation
        """
        gripper_obs = np.array(self.get_obs_dynamixel(), dtype=np.float64)
        # print("Size: ", self.object_size, self.aruco_size)
        aruco_obs_dict = self.get_obs_aruco(object_size=self.object_size, aruco_size=self.aruco_size, display=display)

        '''Gripper'''
        for i, pos in enumerate(gripper_obs):
            if i == 0 or i == 2:
                gripper_obs[i] = self.AngleConvert.xm_2_sim(pos, int(i/2))
            elif i == 1 or i == 3:
                gripper_obs[i] = self.AngleConvert.xl_2_sim(pos, int((i-1)/2))
            elif i == 4 or i == 6:
                gripper_obs[i] = self.AngleConvert.xm_2_sim_vel(pos)
            elif i == 5 or i == 7:
                gripper_obs[i] = 0

        '''Aruco'''
        achieved_goal = aruco_obs_dict['object_centre']
        corners = aruco_obs_dict['object_corner']
        left_xm = aruco_obs_dict['left_xm']
        right_xm = aruco_obs_dict['right_xm']
        # print("Check: ", aruco_obs_dict)

        achieved_goal_with_radi = self._get_achieved_goal_real(achieved_goal, corners, left_xm, right_xm)
        # print("Radi: ", achieved_goal_with_radi)

        observation = np.concatenate(
            [
                gripper_obs,
                achieved_goal_with_radi[-2:]
            ]
        )

        complete_obs = {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal_with_radi.copy(),
            "desired_goal": self.real_env.goal.ravel().copy(),
        }

        obs = {
            "observation": complete_obs["observation"].copy(),
            "achieved_goal": complete_obs["achieved_goal"][-2:],
            "desired_goal": complete_obs["desired_goal"][-2:],
            "object_pose": achieved_goal,
            "corners": corners
        }
        # print("desired_goal: ", obs['desired_goal'])
        return obs

    def _get_achieved_goal_real(self, object_qpos, object_corners, left_xm, right_xm, display=True):
        """
        7 position element of object + 2 radius, without randomisation

        :param object_qpos:
        :param object_corners:
        :return:
        """
        assert object_qpos.shape == (7,)

        if self.real_env.left_contact_idx_real != None and self.real_env.right_contact_idx_real != None:
            left_contact_point = object_corners[self.real_env.left_contact_idx_real]
            right_contact_point = object_corners[self.real_env.right_contact_idx_real]
            # print("contact index", self.real_env.left_contact_idx_real, self.real_env.right_contact_idx_real)
            # print("Corners: ", object_corners)
            achieved_goal_radi = self.real_env.compute_goal_radi(
                                                        b=right_contact_point[:3],
                                                        a=left_contact_point[:3],
                                                        left_motor=left_xm[:3],
                                                        right_motor=right_xm[:3])
            # print("contact point idx: ", self.real_env.left_contact_idx_real, self.real_env.right_contact_idx_real)
            # print("contact point: ", left_contact_point, right_contact_point)
            # print("motor: ", left_xm[:3], right_xm[:3])
        else:
            achieved_goal_radi = [0, 0]

        achieved_goal = np.concatenate((object_qpos, achieved_goal_radi))
        assert achieved_goal.shape == (9,)
        return achieved_goal

    # def start_rotation_real(self, env_goal):
    #     goal = env_goal + [0, 0.10225, 0, 0, 0, 0, 0, 0, 0]
    #     print("Goal: ", goal)
    #     i = 0
    #     self.pos_idx = 1 - self.pos_idx
    #     next_env_dict = self.get_obs_real()
    #     print("pos control finger: ", self.pos_idx, next_env_dict["observation"][self.pos_idx * 2])
    #     self.goal_pos_rotation = next_env_dict["observation"][self.pos_idx * 2]
    #     while True:
    #         next_env_dict = self.get_obs_real()
    #         # self.goal_pos_rotation = next_env_dict["observation"][self.pos_idx * 2]
    #         print("Achieve: ", next_env_dict['object_pose'], self.goal_pos_rotation)
    #         if i == 0:
    #             '''First iteration start with big move'''
    #             last_distance = np.linalg.norm(np.array(goal[:2]) - np.array(next_env_dict['object_pose'][:2]))
    #             start_distance = last_distance
    #             self.start_pos_rotation = self.goal_pos_rotation
    #             self.goal_pos_rotation = np.clip((self.goal_pos_rotation - 0.1), 0, 1.65)
    #         else:
    #             ''' Regular steps '''
    #             distance = np.linalg.norm(np.array(goal[:2]) - np.array(next_env_dict['object_pose'][:2]))
    #             self.goal_pos_rotation = np.clip((self.goal_pos_rotation - 0.01), 0, 1.65)
    #
    #             if distance < last_distance:
    #                 print("1")
    #                 break
    #
    #         goal_pos_dynamixel = (self.goal_pos_rotation/1.8807) * self.gripper_pos_range
    #         print(self.MIN_POS[self.pos_idx] - goal_pos_dynamixel*(self.pos_idx*2-1), goal_pos_dynamixel, self.pos_idx)
    #
    #         self.xm_posControl(self.pos_idx, int(self.MIN_POS[self.pos_idx] - goal_pos_dynamixel*(self.pos_idx*2-1)), reset=True)
    #         self.xm_torque_control(1 - self.pos_idx, reset=True)
    #
    #         i += 1
    #         distance = last_distance
    #     print("Rotation complete")

    def start_rotation_real(self, env_goal):
        goal = env_goal + [0, 0.12212, 0, 0, 0, 0, 0, 0, 0]
        # print("Goal: ", goal)
        next_env_dyn = self.get_obs_dynamixel()
        next_env_dict = self.get_obs_real()
        # print("pos control finger: ", self.pos_idx, next_env_dyn[self.pos_idx * 2])

        if goal[0] < next_env_dict['object_pose'][0]:
            self.pos_idx = 1
        else:
            self.pos_idx = 0
        i = 0
        success_count = 0
        self.start_pos_rotation = [next_env_dyn[0], next_env_dyn[2]]
        self.goal_pos_rotation = next_env_dyn[self.pos_idx * 2]
        last_distance = np.linalg.norm(np.array(goal[:2]) - np.array(next_env_dict['object_pose'][:2]))
        if last_distance < 0.003:
            achieved_goal_euler = self.convert_quat_to_euler(next_env_dict['object_pose'])
            goal_euler = self.convert_quat_to_euler(goal[:7])
            angle_diff = achieved_goal_euler[5] + 90 - goal_euler[3] % 360
            pose_diff = [last_distance, angle_diff]
            print("Achieve: ", achieved_goal_euler, goal_euler)
            print(f"Pose difference: {pose_diff}")
            print("No need for rotation")
        else:
            while True:
                # next_env_dict = self.get_obs_real()
                # print("Achieve: ", next_env_dict['object_pose'], self.goal_pos_rotation)
                # print(self.goal_pos_rotation)
                if i == 0:
                    '''First iteration start with big move'''
                    self.goal_pos_rotation = self.goal_pos_rotation + 100 * (self.pos_idx*2-1)
                else:
                    ''' Regular steps '''
                    self.goal_pos_rotation = self.goal_pos_rotation + 30 * (self.pos_idx*2-1)

                self.xm_posControl(self.pos_idx, int(self.goal_pos_rotation), reset=True)
                self.xm_torque_control(1 - self.pos_idx, reset=True)

                time.sleep(0.5)
                for _ in range(20):
                    next_env_dict = self.get_obs_real()
                # next_env_dict = self.get_obs_real()
                distance = np.linalg.norm(np.array(goal[:2]) - np.array(next_env_dict['object_pose'][:2]))
                print(f'Step {i} with distance: {distance}')
                # print(distance, next_env_dict['object_pose'][:2], goal)
                if distance > last_distance or distance < 0.003:
                    success_count += 1
                    # print("more: ", distance, last_distance)
                    # if success_count > 1:
                    # print("1")
                    for _ in range(50):
                        next_env_dict = self.get_obs_real()
                    achieved_goal_euler = self.convert_quat_to_euler(next_env_dict['object_pose'])
                    goal_euler = self.convert_quat_to_euler(goal[:7])
                    self.end_pos_rotation = [next_env_dict['observation'][0], next_env_dict['observation'][2]]
                    # angle_diff = achieved_goal_euler[5]+90 - goal_euler[3]
                    angle_diff = achieved_goal_euler[5] + 90 - goal_euler[3] % 360
                    pose_diff = [distance, angle_diff]
                    print("Achieve: ", achieved_goal_euler, goal_euler)
                    print(f"Pose difference: {pose_diff}")
                    print("Rotation complete")
                    break
                i += 1
                last_distance = distance
        return achieved_goal_euler, goal_euler, pose_diff

    def reverse_rotation_real(self):
        self.pos_idx = 1 - self.pos_idx
        self.xm_posControl(self.pos_idx, int(self.start_pos_rotation[self.pos_idx]), reset=True)
        self.xm_torque_control(1 - self.pos_idx, reset=True)
        time.sleep(0.5)

    @staticmethod
    def friction_state_mapping(action):
        """
            Friction control follows this formate: [left right]
            Third element decide, LF&HF, or HF&LF, or HF&HF
            Friction State          -1          0           1
            Left friction servo     L(-90)      H(-90)        H(0)
            Right friction servo    H(0)        H(90)        L(90)

            -- On Dynamixel
            For left finger: Low friction is 60 degree (204.8), High friction is 150 degree (512)
            For right finger: Low friction is 240 degree (818.4), High friction is 150 degree (512)

            -- On Mujoco (Define the 0 degree is the centre, clockwise is negative)
            For left finger: Low friction is -90 degree (-1.571 rad), High friction is 0 degree (0 rad)
            For right finger: Low friction is 90 degree (1.571 rad), High friction is 0 degree (0 rad)
            Note: + 150 degree for Dynamixel Control
        """
        if -1 <= action < 0:
            friction_ctrl = [0, 1]
        elif 0 < action <= 1:
            friction_ctrl = [1, 0]
        else:
            assert action == 0, f"Wrong friction state, check: {action}"
            friction_ctrl = [1, 1]
        friction_ctrl = np.array(friction_ctrl)
        # print(friction_ctrl)
        if friction_ctrl is None:
            raise ValueError("Invalid Action with Invalid Friction State")
        assert friction_ctrl.shape == (2,)
        return friction_ctrl

    def convert_quat_to_euler(self, pose_quat):
        quat = pose_quat[3:]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=True)
        pose_euler = np.concatenate([pose_quat[:3], euler])
        return pose_euler



if __name__ == "__main__":
    env_name = "VariableFriction-v5"
    env_ = gym.make(env_name, render_mode="human")
    env_.reset(seed=1)
    Real_HandEnv = RealEnv(env_)
    # print(Real_HandEnv.get_obs_real())



