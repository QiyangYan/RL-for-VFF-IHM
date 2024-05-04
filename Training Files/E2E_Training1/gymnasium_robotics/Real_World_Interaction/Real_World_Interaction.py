import time
import numpy as np
from gymnasium_robotics.dynamixel_driver.bulk_read_write import BULK
from gymnasium_robotics.dynamixel_driver.angle_conversion import AngleConversion
from gymnasium_robotics.envs.real4.manipulate import MujocoManipulateEnv
from gymnasium_robotics.vision.aruco_pose_estimation import ARUCO
from typing import Tuple, List, Dict, Union
import gymnasium as gym


class RealEnv(BULK, ARUCO):
    def __init__(self, env):
        BULK.__init__(self)
        ARUCO.__init__(self, quat=True)

        self.object_size = 0.0355
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
        self.init_camera_for_obs(self.object_size)

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

        robot_obs = self.get_obs_dynamixel()
        return robot_obs

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
        assert friction_state == 1 or friction_state == -1, f"Friction state is wrong, check: {friction_state}"

        # switch control type
        to_high_pos = self.friction_state_mapping(0)
        self.xl_posControl(0, to_high_pos[0])  # left finger
        self.xl_posControl(1, to_high_pos[1])  # right finger
        time.sleep(1)

        pos = self.friction_state_mapping(friction_state)

        # change friction
        self.xl_posControl(0, pos[0])  # left finger
        self.xl_posControl(1, pos[1])  # right finger
        time.sleep(1)

        robot_obs = self.get_obs_dynamixel()
        object_obs = []

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
        gripper_action = self.map_policy_to_real_action(action[0], self.pos_idx)
        if gripper_action != self.gripper_pos_ctrl:
            self.gripper_pos_ctrl = gripper_action  # action is the movement related to the current position
            # self.start = self.real_env.observation[self.pos_idx * 2]
            self.start = robot_obs[self.pos_idx*2]
            self.goal_pos = self.start + self.gripper_pos_ctrl
            # print(self.goal_pos)
            self.xm_posControl(self.pos_idx, int(self.start + self.gripper_pos_ctrl), reset=True)
            self.xm_torque_control(1 - self.pos_idx, reset=True)
        elif self.real_env.firstEpisode:
            pass
        else:
            # print(self.goal_pos)
            self.xm_posControl(self.pos_idx, int(self.goal_pos), reset=False)
            self.xm_torque_control(1 - self.pos_idx, reset=False)

    def reset_robot(self):
        self.xm_posControl(0, self.MIN_POS[0])
        self.xm_posControl(1, self.MIN_POS[1])
        self.xl_posControl(0, 0)
        self.xl_posControl(1, 0)
        time.sleep(2)
        print("gripper joint pose: ", self.get_obs_dynamixel())

    def map_policy_to_real_action(self, action_, pos_idx):
        """
        This is: get the amount of movement for relative control
        """
        standarised_action = (action_ + 1) / 2
        if pos_idx == 0:  # left finger
            gripper_action = - standarised_action * self.gripper_pos_range
        else:  # right finger
            assert pos_idx == 1, f"Invalid pos index, check: {pos_idx}"
            gripper_action = standarised_action * self.gripper_pos_range
        return gripper_action

    def get_obs_real(self):
        """
        This gets the observation based on AruCo and Dynamixel reading.
        Observation should contain as much obtainable information as I can for now.
        :return: complete observation
        """
        gripper_obs = np.array(self.get_obs_dynamixel(), dtype=np.float64)
        aruco_obs_dict = self.get_obs_aruco(self.object_size)

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
        achieved_goal_with_radi = self._get_achieved_goal_real(achieved_goal, corners, left_xm, right_xm)

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
        return complete_obs

    def _get_achieved_goal_real(self, object_qpos, object_corners, left_xm, right_xm):
        """
        7 position element of object + 2 radius, without randomisation

        :param object_qpos:
        :param object_corners:
        :return:
        """
        assert object_qpos.shape == (7,)

        if self.real_env.left_contact_idx != None and self.real_env.right_contact_idx != None:
            left_contact_point = object_corners[self.real_env.left_contact_idx]
            right_contact_point = object_corners[self.real_env.right_contact_idx]
            achieved_goal_radi = self.real_env.compute_goal_radi(left_contact_point[:3],
                                                        right_contact_point[:3],
                                                        left_motor=left_xm[:3],
                                                        right_motor=right_xm[:3])
        else:
            achieved_goal_radi = [0, 0]

        achieved_goal = np.concatenate((object_qpos, achieved_goal_radi))
        assert achieved_goal.shape == (9,)
        return achieved_goal

    def _get_contact_point_real(self):
        contact_coord = []
        for num in range(self.number_of_corners):
            contact_idx = self._model_names._site_name2id[f"target:corner{num + 1}"]
            contact_coord.append(self.data.site_xpos[contact_idx])
        print("contact point: ", contact_coord)
            # print("contact:", contact_idx, self.data.site_xpos[contact_idx])
        left_index, left_contact = max(enumerate(contact_coord), key=lambda coord: coord[1][0])
        right_index, right_contact = min(enumerate(contact_coord), key=lambda coord: coord[1][0])
        # print("left contact: ", left_contact, right_contact)

        # print("target position: ", self._utils.get_joint_qpos(self.model, self.data, "joint:target"))

        self.left_contact_idx = left_index
        self.right_contact_idx = right_index
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


if __name__ == "__main__":
    env_name = "VariableFriction-v5"
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=1)
    Real_HandEnv = RealEnv(env)
    # print(Real_HandEnv.get_obs_real())



