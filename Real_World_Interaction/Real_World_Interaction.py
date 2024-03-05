import time
import numpy as np
from dynamixel_driver.bulk_read_write import BULK
from gymnasium_robotics.envs.variable_friction_continuous.manipulate import MujocoManipulateEnv

import sys
sys.path.append('/Users/qiyangyan/Desktop/FYP/Vision')
from aruco_obs import VisionObs


class RealHandEnv(BULK, VisionObs, MujocoManipulateEnv):
    def __init__(self):
        super().__init__()
        self.observation = []
        self.gripper_pos_range = self.MAX_POS[0] - self.MIN_POS[0]
        self.get_actuator_position()

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

    def switch_control_mode(self, pos_idx):
        """
        This change the pos_idx corresponding finger to position control, and another one to torque control

        Todo:
        1. Check later to see the necessity of change both friction to high before change control mode

        :param pos_idx: right finger = 0, left finger = 1
        :return: None
        """
        # Pick up, 1 is torque
        self.xm_posControl(pos_idx, self.packetHandler.read4ByteTxRx(
            self.portHandler, self.DXL_ID_aray[1], self.XM['ADDR_PRO_PRESENT_POSITION'])[0], reset=True)
        self.xm_torque_control(1 - pos_idx, reset=True)

    def change_friction_to_low(self, action, last_pos_idx):
        """
        Friction change doesn't change control mode, only change control mode for the subsequent slide

        :param action: [2, friction_state, True]
        :return: full observation

        todo:
        1. Require switch control mode based on the next action type, not friction state
        This one is based on the friction state, align with simulation, but not good
        """
        friction_state = action[2]
        assert friction_state == 1 or friction_state == -1, f"Friction state is wrong, check: {friction_state}"

        # switch control type
        pos, pos_idx = self.friction_state_mapping(friction_state)
        if pos_idx != last_pos_idx:
            self.switch_control_mode(pos_idx)

        # change friction
        self.xl_posControl(0, pos[0])  # left finger
        self.xl_posControl(1, pos[1])  # right finger
        time.sleep(1)

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

    def step_continuous_real(self, action):
        """
        This is for: IHM with continuous action space in actual robot, replace the elif len(action) == 2
        :param action: [-1, 1]
        :return:
        """
        assert len(action) == 2, f"check action length: {len(action)}, action is: {action}"

        """ 
        Get Ready for the slide
        """
        step_friction_state, control_mode = self.discretize_action_to_control_mode(action[1])
        if step_friction_state != self.last_friction_state:
            self.friction_change_penalty = True
        else:
            self.friction_change_penalty = False

        _, self.pos_idx = self.action_to_control(control_mode)
        self.switch_control_mode(self.pos_idx)

        """ 
        If control signal is different as the previous control signal, then it means new slide starts, need to reassign 
        the start position
        """
        gripper_action = self.map_policy_to_real_action(action, self.pos_idx)
        if gripper_action != self.gripper_pos_ctrl:
            self.gripper_pos_ctrl = gripper_action  # action is the movement related to the current position
            self.start = self.observation[self.pos_idx * 2]

        """
        Start slide: relative control
        """
        if self.firstEpisode:
            self.data.ctrl[1] = 0  # change friction if stuck
            self.data.ctrl[0] = 0
        else:
            self.xm_posControl(self.pos_idx, self.start + gripper_action, reset=False)

    def map_policy_to_real_action(self, action, pos_idx):
        """
        This is: get the amount of movement for relative control
        """
        standarised_action = (action + 1) / 2
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
        gripper_obs = self.get_obs_dynamixel()
        achieved_goal, achieved_goal_radi = self._get_achieved_goal_real()
        observation = np.concatenate([gripper_obs, achieved_goal])
        achieved_goal_with_radi = np.concatenate([achieved_goal, achieved_goal_radi])

        complete_obs = {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal_with_radi.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }
        return complete_obs

    def _get_achieved_goal_real(self):
        achieved_goal_pose = self.get_obs_aruco()
        assert achieved_goal_pose.shape == (6,), f"Achieved goal has wrong shape, check: {achieved_goal_pose}"

        if self.left_contact_idx is not None and self.right_contact_idx is not None:
            left_contact_point, right_contact_point = self.get_contact_point(self.left_contact_idx, self.right_contact_idx)
            achieved_goal_radi = self.compute_goal_radi(left_contact_point, right_contact_point)
        else:
            achieved_goal_radi = [0, 0]
            # print("initialising")
        # achieved_goal = np.concatenate((achieved_goal_pose, achieved_goal_radi))
        # assert achieved_goal.shape == (9,)
        return achieved_goal_pose, achieved_goal_radi




