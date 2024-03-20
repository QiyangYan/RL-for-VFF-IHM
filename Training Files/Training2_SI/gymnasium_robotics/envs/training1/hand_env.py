# This is the summary of the structure of hand_env.py and fetch_env.py
'''
This is the summary of the structure of hand_env.py and fetch_env.py
"fetch_env" includes
1. _step_callback
2. _set_action
3. generate_mujoco_observations
4. _get_gripper_xpos
5. _render_callback
6. _reset_sim
7. _env_setup

"hand_env" includes
1. _set_action

"MujocoManipulateEnv" includes
1. _get_achieved_goal
2. _env_setup
3. _reset_sim
4. _sample_goal
5. _render_callback
6. _get_obs
'''
import time

# Information regarding _set_action
'''
Information regarding _set_action
Overview: This function gets the action input from high-layer and set the joint based on these

fetch_env: 
1. Get action input, check shape is valid
2. Extract specific info from action, since last element corresponds to two joint
3. Apply modifications to those info_s
4. Return modified action
Note:
    1. Always do a check to the array before use it
    2. Include an end_effector direction quaternion
    3. When multiple joints are set based on single input infor, we can modify based on that
    , for example concatenate, to control those joints.
    
hand_env:
1. Get action input, check shape is valid
2. Get actuator control range from model
3. If relative control mode: create actuation_center, fill it with data read from joint pos
4. If absolute control mode: actuation_center = the half of control range
5. Output is obtained by applying input action to actuation center
6. Add clip to ensure output is within range
Note: 
    1. Actuation center: means the reference point for control, mid position (absolute) or current position (relative)

'''

# This is the action space
'''
Box(-1.0, 1.0, (2,), float32), haven't confirm other parameters

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
# | 1   | Angular position of the right finger                | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
| 1   | Friction States                                     | -1          | 1           | -1.571 (rad) | 1.571 (rad) | robot0:A_FFJ3 & robot0:A_FFJ4    | hinge | angle (rad) |

Third element decide, LF&HF, or HF&LF, or HF&HF
Friction State          -1          0           1
Left friction servo     L(-90)      H           H
Right friction servo    H           H           L (90)

-- On Dynamixel
For left finger: Low friction is 60 degree (204.8), High friction is 150 degree (512)
For right finger: Low friction is 240 degree (818.4), High friction is 150 degree (512)

-- On Mujoco (Define the 0 degree is the centre, clockwise is negative)
For left finger: Low friction is -90 degree (-1.571 rad), High friction is 0 degree (0 rad)
For right finger: Low friction is 90 degree (1.571 rad), High friction is 0 degree (0 rad)
Note: + 150 degree for Dynamixel Control


CONSIDER FOLLOWING PROBLEMS
1. How you want to control the gripper? Both position control or switching between pos and torque control ✅
2. Is it possible to include both continuous and discrete actions? ✅
3. What parameters should I set for the table? ✅
4. What to do with the mid-layer I implemented before? 
'''

from typing import Union
import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
import math
# from gymnasium_robotics.envs.plot_array import plot_numbers, plot_numbers_two_value

DEFAULT_CAMERA_CONFIG = {
    "distance": -0.5,
    "azimuth": 90,
    "elevation": -90,
    "lookat": np.array([0, -0.25, 0.3]),
}


def get_base_hand_env(
    RobotEnvClass: MujocoRobotEnv
    # 它表示 RobotEnvClass 参数的类型是 Union[MujocoPyRobotEnv, MujocoRobotEnv]
    # 并且它是一个必需的参数，调用函数时需要传递这个参数的值
) -> MujocoRobotEnv:
    """Factory function that returns a BaseHandEnv class that inherits from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings."""

    class BaseHandEnv(RobotEnvClass):
        """Base class for all robotic hand environments."""

        def __init__(self, relative_control, **kwargs):
            self.relative_control = relative_control
            super().__init__(n_actions=2, **kwargs)

        # RobotEnv methods
        # ----------------------------

    return BaseHandEnv

'''Read me'''
'''
Both fingers follow position control, try this version first.

If this doesn't work, modify this so that 
1. the high friction finger follows position control
2. the low friction finger follows torque control

CONSIDER:
1. What to do with the mid-layer I implemented before? 
'''
class MujocoHandEnv(get_base_hand_env(MujocoRobotEnv)):
    def __init__(
        self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs
    ) -> None:
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        # self.torque_ctrl = 1
        self.pick_up = False
        self.closing = False
        self.count = 0
        self.last_motor_diff = 0
        self.torque_high_indicator = 0
        self.stuck_terminate = False
        self.firstEpisode = True
        self.step_count_for_friction_change = 0
        self.IHM_start = False
        self.start_height = 0
        self.friction_changing = False
        self.friction_change_count = 0
        self.friction_withRL = False
        self.pos_idx = 0

        self.torque_history = []
        self.error_history = []
        self.angle_history = []
        self.vel_history_1 = []
        self.vel_history_2 = []
        self.joint_left_history = []
        self.joint_right_history = []
        self.last_friction_state = 0
        self.friction_change_penalty = False
        self.gripper_pos_ctrl = 0
        self.friction_state = 0

        # no overshoot
        self.pos_ctrl_params = {
            'damping': 4.99979599,
            'armature': 0.39951026,
            'frictionloss': 0.03,
            'ctrlrange': [0, 1.8807],
            'kp': 12.87708353
        }

        # self.pos_ctrl_params = {
        #     'damping': 4.99979599,
        #     'armature': 0.92183252,
        #     'frictionloss': 0.03,
        #     'ctrlrange': [0, 1.8807],
        #     'kp': 12.02150098
        # }

        # [ 5.          0.9198448  12.02074596  0.        ]

        self.torque_ctrl_params = {
            'damping': 5.67185526e-02,
            'armature': 3.55878361e-03,
            'frictionloss': 4.86429385e-07,
            'ctrlrange': [0, 1]
        }


    def _set_action(self, action=None):

        ctrlrange = self.model.actuator_ctrlrange

        if isinstance(action, np.int64):
            '''slide fixed magnitude with discrete action space, discrete(4)'''
            step_friction_state, pos_idx = self.action_to_control(action)
            if step_friction_state != self.last_friction_state:
                self.friction_change_penalty = True
            else:
                self.friction_change_penalty = False
            self.last_friction_state = step_friction_state
            if pos_idx != self.pos_idx:
                self.switch_ctrl_type_pos_idx(self.pos_idx)
                self.pos_idx = pos_idx
            if self.firstEpisode:
                self.data.ctrl[1] = 0  # change friction if stuck
                self.data.ctrl[0] = 0
            else:
                # left finger low/torque, right finger high/pos
                self.data.ctrl[1-self.pos_idx] = 0.3
                self.data.ctrl[self.pos_idx] = self.data.qpos[self.pos_idx * 2 + 1] - 0.01
                # print("toque is: ", self.data.ctrl[self.torque_idx], "action is: ", self.data.ctrl[self.pos_idx])
                # print("move: ", self.data.ctrl[self.pos_idx], self.data.qpos[self.pos_idx * 2 + 1], step_friction_state)

        elif self.friction_withRL == True and action[2] == True:
            '''use RL for friction change, (3, )'''
            print("use RL for friction change")
            assert action.shape == (3,), f"Require an extra friction change indicator at action[2]: {len(action)}"
            assert isinstance(action[2], bool), "friction_changing must be a boolean"
            self._set_action_friction(action)

        elif len(action) == 2:
            '''IHM with continuous action space
            gripper_pos_ctrl = (0, 1.68)
            self.friction_state = (1, -1)
            if control mode has different friction state as the current position
            '''
            step_friction_state, control_mode = self.discretize_action_to_control_mode(action[1])
            if step_friction_state != self.last_friction_state:
                self.friction_change_penalty = True
            else:
                self.friction_change_penalty = False

            _, pos_idx = self.action_to_control(control_mode)
            if pos_idx != self.pos_idx:
                self.pos_idx = pos_idx
                self.switch_ctrl_type_pos_idx(self.pos_idx)
            if (action[0] + 1) / 2  != self.gripper_pos_ctrl:
                # print("---------------------", self.pos_idx, control_mode)
                # input("press enter to continue")
                self.gripper_pos_ctrl = (action[0] + 1) / 2 # action is the movement related to the current position
                # self.move = 0
                self.start = self.data.qpos[self.pos_idx*2+1]

            if self.firstEpisode:
                self.data.ctrl[1] = 0  # change friction if stuck
                self.data.ctrl[0] = 0
            else:
                self.data.ctrl[1 - self.pos_idx] = 0.3
                self.data.ctrl[self.pos_idx] = self.start - self.gripper_pos_ctrl * 1.8807 / 7
                ''' Make sure the qpos changed by given amount '''
                # if self.data.qpos[self.pos_idx * 2 + 1] - self.start < self.gripper_pos_ctrl:
                #     self.data.ctrl[self.pos_idx] = self.data.ctrl[self.pos_idx] - 0.01
                    # print("check: ", self.data.ctrl[self.pos_idx])
                # print("check: ", self.data.qpos[self.pos_idx * 2 + 1] - self.start, self.data.qpos[self.pos_idx * 2 + 1], self.data.ctrl[self.pos_idx], self.gripper_pos_ctrl)

                ''' Make sure the ctrl changed by given amount '''
                # if self.moved != self.gripper_pos_ctrl:
                #     self.moved += 0.01  # amount of movement
                # if self.moved < self.gripper_pos_ctrl:
                #     self.data.ctrl[self.pos_idx] = self.data.ctrl[self.pos_idx] - 0.01
                # elif self.moved > self.gripper_pos_ctrl:
                #     self.data.ctrl[self.pos_idx] = self.data.ctrl[self.pos_idx] - (self.gripper_pos_ctrl - (self.moved - 0.01))
                #     self.moved = self.gripper_pos_ctrl
                # else:
                #     assert self.moved == self.gripper_pos_ctrl, "moved amount should be equal as the action required"
                #     pass

        else:
            '''RL-inspired method and pick-up process, (3, )
            This friction change is used if not RL for friction change
            '''
            # print("here")
            assert action.shape == (3,), f"Action should have size 3: {action.shape}"
            assert action[2] == 0 or action[2] == 1, f"friction_changing must be a int boolean 0 or 1: {action[2]}"
            assert self.friction_withRL == False
            self._set_action_continuous(action)

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        # print(self.data.ctrl)

    def discretize_action_to_control_mode(self, action):
        # print(self.model.actuator_ctrlrange[:, 1][self.pos_idx])
        action_norm = (action + 1) / 2

        if 1 / 4 > action_norm >= 0:
            control_mode = 0
            friction_state = 1
        elif 2 / 4 > action_norm > 1 / 4:
            control_mode = 1
            friction_state = 1
        elif 3 / 4 > action_norm > 2 / 4:
            control_mode = 2
            friction_state = -1
        else:
            assert 4 >= action_norm > 3 / 4, f"wrong action: {action}"
            control_mode = 3
            friction_state = -1

        return friction_state, control_mode

    def _set_action_friction(self, action):

        if action[1] == 1 or action[1] == -1 or action[1] == 0:
            # torque
            gripper_torque_ctrl, self.friction_state = action[0], action[1]
            ctrlrange = self.model.actuator_ctrlrange

            # gripper_torque_ctrl = np.clip((gripper_torque_ctrl + 1) / 8 + 0.25, 0.25, 0.55)
            # gripper_torque_ctrl = np.clip((gripper_torque_ctrl + 1) / 5.71 + 0.25, 0.25, 0.6)

            assert self.friction_state == -1 or 0 or 1

            self.data.ctrl[self.torque_idx] = gripper_torque_ctrl
            self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

            self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

            z = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
            current_angle = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4]
            current_vel_1 = self._utils.get_joint_qvel(self.model, self.data, "joint:leftInsert")
            current_vel_2 = self._utils.get_joint_qvel(self.model, self.data, "joint:rightInsert")

            # self.torque_history.append(gripper_torque_ctrl)
            # self.error_history.append(z)
            # self.angle_history.append(current_angle)
            # self.vel_history_1.append(current_vel_1)
            # self.vel_history_2.append(current_vel_2)

        else:
            # torque and position
            gripper_torque_ctrl = action[0]
            pos_deviation = action[1]
            self.friction_state = 0
            ctrlrange = self.model.actuator_ctrlrange

            self.data.ctrl[self.torque_idx] = gripper_torque_ctrl
            self.data.ctrl[self.pos_idx] += pos_deviation
            self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)
            self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

            # print("check position deviation", self.data.ctrl[self.pos_idx], pos_deviation)

            z = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
            current_angle = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4]
            current_vel_1 = self._utils.get_joint_qvel(self.model, self.data, "joint:leftInsert")
            current_vel_2 = self._utils.get_joint_qvel(self.model, self.data, "joint:rightInsert")
            joint_left = self._utils.get_joint_qpos(self.model, self.data, "joint:left")
            joint_right = self._utils.get_joint_qpos(self.model, self.data, "joint:right")

            self.torque_history.append(gripper_torque_ctrl)
            self.error_history.append(z)
            self.angle_history.append(current_angle)
            self.vel_history_1.append(current_vel_1)
            self.vel_history_2.append(current_vel_2)
            self.joint_left_history.append(joint_left)
            self.joint_right_history.append(joint_right)


    def _set_action_continuous(self, action):

        # print("action in slide: ", action)
        gripper_pos_ctrl, friction_state, friction_change = action[0], action[1], action[2]
        ctrlrange = self.model.actuator_ctrlrange

        if friction_state == 2:
            # print("Picking", self.pick_up, self.closing, self.count, action[0])
            '''grasp object'''
            self.pos_idx = 0
            assert friction_change == False, f"friction change should be False: {friction_change}"
            if self.pick_up == False:
                self.data.ctrl[0] = action[0]
                self.data.ctrl[1] = 0
                self.data.ctrl[4] = 0
                if abs(self.data.qpos[1] - 1.05) < 0.003: # when hh, 0.001
                    self.pick_up = True
            elif self.closing == True and self.pick_up == True:
                pass
            else:
                assert self.closing == False and self.pick_up == True
                self.data.ctrl[0] = action[0]
                self.data.ctrl[1] = 0.3
                self.count += 0.5
                if self.count == 49:
                    self.closing = True

        elif friction_state == 3:
            '''lift the object'''
            self.pos_idx = 0
            assert friction_change == False, f"friction change should be False: {friction_change}"
            self.data.ctrl[0] = action[0]
            self.data.ctrl[1] = 0.3
            self.data.ctrl[4] = self.pick_up_height
            self.count += 1
            if self.count > 165:
                # this the reference height for slip
                self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
                self.IHM_start = True
                # print("start_height: ", self.start_height)

        elif gripper_pos_ctrl == 2 or friction_change == True:
            '''friction change'''
            # print("Friction change")
            assert gripper_pos_ctrl == 2, f"action[0] is not 2, not indicating friction change, instead it's {gripper_pos_ctrl}"
            assert friction_change == True, f"action[1] is not True, not indicating friction change, instead it's {friction_change}"
            assert friction_state == -1 or 0 or 1
            # print("friction: ", friction_state, self.friction_state)
            if friction_state != self.friction_state:
                # print(self.data.ctrl, self.data.qpos[[self.pos_idx * 2 + 1]],
                #       self.data.qpos[[(1 - self.pos_idx) * 2 + 1]])
                self.switch_ctrl_type_friction(friction_state)
                self.friction_state = friction_state
            # time.sleep(2)

            '''Smaller the torque, smaller the introduced error, but less stable the friction change is'''
            torque_ctrl = 0.3
            # print("Index: ", self.pos_idx, self.data.ctrl)

            self.data.ctrl[1-self.pos_idx] = torque_ctrl
            self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

            if self.check_action_complete():
                self.data.ctrl[1-self.pos_idx] = torque_ctrl
                self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

            # print(self.data.ctrl)

        else:
            '''manipulation'''
            print("Manipulation")
            self.switch_ctrl_type_pos_idx(self.pos_idx)
            if self.firstEpisode:
                self.data.ctrl[1] = 0  # change friction if stuck
                self.data.ctrl[0] = 0
            elif action[1] == 0:
                # rotation
                self.data.ctrl[1-self.pos_idx] = 0.3
                self.data.ctrl[self.pos_idx] = action[0]
            else:
                # left finger low/torque, right finger high/pos
                self.data.ctrl[1-self.pos_idx] = 0.3
                self.data.ctrl[self.pos_idx] = action[0] # change friction if stuck
                print("toque is: ", self.data.ctrl[1-self.pos_idx], "action is: ", self.data.ctrl[self.pos_idx])

            # ctrlrange[0, 1] is because only single motor is controlled by action

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])



    def check_action_complete(self):
        self.motor_diff = self.data.ctrl[self.pos_idx] - self.data.qpos[self.pos_idx*2+1]

        friction_check_1 = self.friction_complete_check(self.data.ctrl[2], self.data.qpos[2])
        friction_check_2 = self.friction_complete_check(self.data.ctrl[3], self.data.qpos[4])

        if self.data.ctrl[2] != self.data.ctrl[3]: # one high one low
            motor_limit = 0.045  # 0.04, the torque is relatively small, so high accuracy is desired
        else: # both to high
            motor_limit = 0.1  # the torque is very large to avoid slip, so pos might deviate from the position it wants to remain, higher tolerance

        # print("friction diff: ", friction_check_1, friction_check_2, abs(self.motor_diff), motor_limit)

        if friction_check_1 and friction_check_2 and abs(self.motor_diff) < motor_limit:
            # print("friction diff: ", self.data.qpos[2], self.data.qpos[4], abs(self.motor_diff))
            # print(self.data.qpos[2], self.data.ctrl[3], self.data.qpos[2], self.data.qpos[4])
            return True
        else:
            return False

    def friction_complete_check(self, ctrl_val, qpos):
        if ctrl_val == 0:  # low friction
            if abs(qpos) < 0.0004:  # this is the boundary the friction turns from low to high, it was 0.00165
                return True
        else:
            if abs(ctrl_val / 100 - qpos) < 0.0004:  # this is the value we used to ensure low friction is completed
                return True
        return False

    def friction_state_mapping(self,action):
        '''
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
                '''
        if -1 <= action < 0:
            friction_ctrl = [0.34, 0]
        elif 0 < action <= 1:
            friction_ctrl = [0, 0.34]
        elif action == 0:
            friction_ctrl = [0, 0]
        friction_ctrl = np.array(friction_ctrl)
        # print(friction_ctrl)
        if friction_ctrl is None:
            raise ValueError("Invalid Action with Invalid Friction State")
        assert friction_ctrl.shape == (2,)
        return friction_ctrl

    def switch_ctrl_type_pos_idx(self, pos_idx):
        '''
        Friction State          -1          0           1
        Left friction servo     L(-90)      H(-90)      H(0)
        Right friction servo    H(0)        H(90)       L(90)

        Action Type	Control Mode       |    (Left, Right)
        Slide up on right finger	        P, T
        Slide down on right finger	        T, P
        Slide up on left finger	            T, P
        Slide down on left finger	        P, T
        # Rotate clockwise	                T, P
        # Rotate anticlockwise	            P, T
        '''
        self.pos_idx = pos_idx
        self.torque_idx = 1 - pos_idx

        self.model.actuator_gainprm[self.pos_idx][0] = self.pos_param[0]
        self.model.actuator_biastype[self.pos_idx] = self.pos_param[1]
        self.model.actuator_biasprm[self.pos_idx][1] = self.pos_param[2]
        self.model.actuator_ctrlrange[self.pos_idx] = self.pos_ctrl_params['ctrlrange']
        self.model.dof_damping[self.pos_idx * 2 + 1] = self.pos_ctrl_params['damping']
        self.model.dof_armature[self.pos_idx * 2 + 1] = self.pos_ctrl_params['armature']
        self.model.dof_frictionloss[self.pos_idx * 2 + 1] = self.pos_ctrl_params['frictionloss']

        self.model.actuator_gainprm[self.torque_idx][0] = 1
        self.model.actuator_biastype[self.torque_idx] = 0
        self.model.actuator_biasprm[self.torque_idx][1] = 0
        self.model.actuator_ctrlrange[self.torque_idx] = self.torque_ctrl_params['ctrlrange']
        self.model.dof_damping[self.torque_idx * 2 + 1] = self.torque_ctrl_params['damping']
        self.model.dof_armature[self.torque_idx * 2 + 1] = self.torque_ctrl_params['armature']
        self.model.dof_frictionloss[self.torque_idx * 2 + 1] = self.torque_ctrl_params['frictionloss']

        self.data.ctrl[self.torque_idx] = 1
        self.data.ctrl[self.pos_idx] = self.data.qpos[self.pos_idx * 2 + 1]  # 0 -> 1, 1 -> 3
        # print("switch: ", self.data.qpos[self.pos_idx * 2 + 1])

        return self.data.qpos[self.pos_idx * 2 + 1], self.pos_idx

    def switch_ctrl_type_friction(self,next_friction_state):
        '''
        Friction State          -1          0           1
        Left friction servo     L(-90)      H(-90)        H(0)
        Right friction servo    H(0)        H(90)        L(90)

        High friction is position control,
        Low friction is torque control,
        For Rotation: Check the position of target to determine the control mode
        '''
        # 0 is left finger, 1 is right finger
        if next_friction_state == -1:
            pos_idx = 1
            # print("L is torque, R is position")
        elif next_friction_state == 1:
            pos_idx = 0
            # print("L is position, R is torque")
        else:
            # pos_idx = 1 - self.pos_idx  # following finger is position control

            # based on object's goal position, for rotation
            if self.goal[0] > self._utils.get_joint_qpos(self.model, self.data, "joint:object")[0]:
                # the goal is on the left side of the object
                pos_idx = 1
                # print("L is torque, R is position")
            else:
                # the goal is on the right side of the object
                pos_idx = 0
                # print("L is position, R is torque")

        self.pos_idx = pos_idx
        self.torque_idx = 1 - self.pos_idx

        self.model.actuator_gainprm[1-self.pos_idx][0] = 1
        self.model.actuator_biastype[1-self.pos_idx] = 0
        self.model.actuator_biasprm[1-self.pos_idx][1] = 0
        self.model.actuator_ctrlrange[self.torque_idx] = self.torque_ctrl_params['ctrlrange']
        self.model.dof_damping[self.torque_idx * 2 + 1] = self.torque_ctrl_params['damping']
        self.model.dof_armature[self.torque_idx * 2 + 1] = self.torque_ctrl_params['armature']
        self.model.dof_frictionloss[self.torque_idx * 2 + 1] = self.torque_ctrl_params['frictionloss']

        self.model.actuator_gainprm[self.pos_idx][0] = self.pos_param[0]
        self.model.actuator_biastype[self.pos_idx] = self.pos_param[1]
        self.model.actuator_biasprm[self.pos_idx][1] = self.pos_param[2]
        self.model.actuator_ctrlrange[self.pos_idx] = self.pos_ctrl_params['ctrlrange']
        self.model.dof_damping[self.pos_idx * 2 + 1] = self.pos_ctrl_params['damping']
        self.model.dof_armature[self.pos_idx * 2 + 1] = self.pos_ctrl_params['armature']
        self.model.dof_frictionloss[self.pos_idx * 2 + 1] = self.pos_ctrl_params['frictionloss']

        self.data.ctrl[1-self.pos_idx] = 0.3
        self.data.ctrl[self.pos_idx] = self.data.qpos[self.pos_idx * 2 + 1]  # 0 -> 1, 1 -> 3

        return self.data.qpos[self.pos_idx * 2 + 1], self.pos_idx

    def action_to_control(self, action):
        """
        Action Space:
        Action  |   Movement                        |   friction state      |   pos index
        0           Slide up on right finger                1                       0
        1           Slide down on right finger              1                       1
        2           Slide up on left finger                -1                       1
        3           Slide down on left finger              -1                       0
        """

        if action == 0:
            friction_state = 1
            pos_idx = 0
        elif action == 1:
            friction_state = 1
            pos_idx = 1
        elif action == 2:
            friction_state = -1
            pos_idx = 1
        else:
            assert action == 3, "action value outputted by policy is incorrect"
            friction_state = -1
            pos_idx = 0

        return friction_state, pos_idx

