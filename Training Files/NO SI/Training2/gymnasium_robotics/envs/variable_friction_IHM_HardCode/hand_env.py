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
from gymnasium_robotics.envs.plot_array import plot_numbers, plot_numbers_two_value

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

        def _set_action(self, action):
            assert action.shape == (2,)


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
        self.terminate_count = 0
        self.stuck_terminate = False
        self.firstEpisode = True
        self.pick_up_height = 3
        self.step_count_for_friction_change = 0
        self.IHM_start = False
        self.start_height = 0
        self.friction_changing = False
        self.friction_change_count = 0

        self.torque_history = []
        self.error_history = []
        self.angle_history = []
        self.vel_history_1 = []
        self.vel_history_2 = []
        self.joint_left_history = []
        self.joint_right_history = []

        # self.last_friction = 0
        # self.current_friction = 0
        # self.action_count = 0
        # self.last_motor_pos = 0
        # self.current_motor_pos = 0
        # self.motor_direction = 0 # 0 for clockwise, 1 for anticlockwise
        # self.last_motor_direction = 0
        # self.same_friction = True
        # self.same_motor_direction = True

    def _set_action(self, action):
        super()._set_action(action)  # check if action has the right shape: 3 dimension

        gripper_pos_ctrl, self.friction_state = action[0], action[1]
        ctrlrange = self.model.actuator_ctrlrange
        # self.pick_up_height = 3

        if self.friction_state == 2:
            '''grasp object'''
            self.step_count_for_friction_change = 0
            if self.pick_up == False:
                self.data.ctrl[0] = action[0]
                self.data.ctrl[1] = 0
                self.data.ctrl[4] = 0
                if self.action_complete() and abs(action[0]-1.05)<0.005: # when hh, 0.001
                    # print("action complete")
                    self.pick_up = True
            elif self.closing == True and self.pick_up == True:
                # print("check", self.pick_up, self.closing, self.data.ctrl[:2])
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = 1
            else:
                # print(self.pick_up, self.closing, self.data.ctrl[:2])
                assert self.closing == False and self.pick_up == True
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = 1
                self.count += 1
                # print("closing", self.count)
                if self.count == 50:
                    self.closing = True

        elif self.friction_state == 3:
            '''lift the object'''
            self.step_count_for_friction_change = 0
            self.data.ctrl[0] = 1.05
            self.data.ctrl[1] = 1
            self.data.ctrl[4] = self.pick_up_height
            self.count += 1
            if self.count > 165:
                # this the reference height for slip
                self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
                self.IHM_start = True
                # print("start_height: ", self.start_height)

        elif gripper_pos_ctrl == 2 or self.friction_changing == True:
            ''' change friction
            this is to ensure the action is completed before change the frition
            '''
            self.step_count_for_friction_change += 1
            assert self.friction_state == -1 or 0 or 1
            if self.friction_withRL:
                self.data.ctrl[self.torque_idx] = action[0]

                if self.action_complete():
                    self.data.ctrl[self.torque_idx] = action[0]
                    self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)
            else:
                torque_ctrl = 0.3
                self.data.ctrl[self.torque_idx] = torque_ctrl

                if self.action_complete():
                    self.data.ctrl[self.torque_idx] = torque_ctrl
                    self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

        # else:
        #     '''manipulation'''
        #     self.step_count_for_friction_change = 0
        #     # print("manipulation")
        #     # print("torque magnitude: ", self.data.ctrl[self.torque_idx])
        #     # print(self.goal)
        #     print("manipulation idx in hand_env",self.torque_idx, self.pos_idx)
        #     self.terminate_count = 0
        #     if self.firstEpisode:
        #         self.data.ctrl[1] = 0  # change friction if stuck
        #         self.data.ctrl[0] = 0
        #     elif action[1] == 0:
        #         # rotation
        #         self.data.ctrl[self.torque_idx] = 0.5
        #         self.data.ctrl[self.pos_idx] = action[0]
        #     else:
        #         # left finger low/torque, right finger high/pos
        #         self.data.ctrl[self.torque_idx] = 1
        #         self.data.ctrl[self.pos_idx] = action[0] # change friction if stuck

            # ctrlrange[0, 1] is because only single motor is controlled by action

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _set_action_friction(self, action):
        super()._set_action(action)  # check if action has the right shape: 3 dimension
        # print(action[1])

        if action[1] == 1 or action[1] == -1 or action[1] == 0:
            # torque
            gripper_torque_ctrl, self.friction_state = action[0], action[1]
            ctrlrange = self.model.actuator_ctrlrange

            # gripper_torque_ctrl = np.clip((gripper_torque_ctrl + 1) / 8 + 0.25, 0.25, 0.55)
            # gripper_torque_ctrl = np.clip((gripper_torque_ctrl + 1) / 5.71 + 0.25, 0.25, 0.6)
            # print("action: ", action)

            self.step_count_for_friction_change += 1

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

            print("check position deviation", self.data.ctrl[self.pos_idx], pos_deviation)

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


    def _set_action_slide(self, action):
        super()._set_action(action)  # check if action has the right shape: 3 dimension

        gripper_pos_ctrl, self.friction_state = action[0], action[1]
        ctrlrange = self.model.actuator_ctrlrange

        if self.friction_state == 2:
            '''grasp object'''
            self.step_count_for_friction_change = 0
            if self.pick_up == False:
                self.data.ctrl[0] = action[0]
                self.data.ctrl[1] = 0
                self.data.ctrl[4] = 0
                if abs(self.data.qpos[1] - 1.05) < 0.04: # when hh, 0.001
                    # print("action complete")
                    self.pick_up = True
            elif self.closing == True and self.pick_up == True:
                # print("check", self.pick_up, self.closing, self.data.ctrl[:2])
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = 1
            else:
                # print(self.pick_up, self.closing, self.data.ctrl[:2])
                assert self.closing == False and self.pick_up == True
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = 1
                self.count += 1
                # print("closing", self.count)
                if self.count == 50:
                    self.closing = True

        elif self.friction_state == 3:
            '''lift the object'''
            self.step_count_for_friction_change = 0
            self.data.ctrl[0] = 1.05
            self.data.ctrl[1] = 1
            self.data.ctrl[4] = self.pick_up_height
            self.count += 1
            if self.count > 165:
                # this the reference height for slip
                self.start_height = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
                self.IHM_start = True
                # print("start_height: ", self.start_height)

        elif gripper_pos_ctrl == 2 or self.friction_changing == True:
            ''' change friction
            this is to ensure the action is completed before change the frition
            '''
            self.step_count_for_friction_change += 1
            assert self.friction_state == -1 or 0 or 1

            '''Smaller the torque, smaller the introduced error, but less stable the friction change is'''
            if self.friction_state == 0:
                torque_ctrl = 0.2
            else:
                torque_ctrl = 0.4
            # torque_ctrl = 1 # In-Air

            # print("torque: ", torque_ctrl)

            # if self.friction_state == 0 \
            #     and abs(self.data.ctrl[2] / 100 - self.data.qpos[2]) < 0.005 \
            #     and abs(self.data.ctrl[3] / 100 - self.data.qpos[4]) < 0.005:
            #     torque_ctrl = 0.4

            self.data.ctrl[self.torque_idx] = torque_ctrl
            self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

            if self.action_complete():
                self.data.ctrl[self.torque_idx] = torque_ctrl
                self.data.ctrl[2:4] = self.friction_state_mapping(self.friction_state)

            # z = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[2]
            # current_angle = self._utils.get_joint_qpos(self.model, self.data, "joint:object")[4]
            # current_vel_1 = self._utils.get_joint_qvel(self.model, self.data, "joint:leftInsert")
            # current_vel_2 = self._utils.get_joint_qvel(self.model, self.data, "joint:rightInsert")
            #
            # self.torque_history.append(torque_ctrl)
            # self.error_history.append(z)
            # self.angle_history.append(current_angle)
            # self.vel_history_1.append(current_vel_1)
            # self.vel_history_2.append(current_vel_2)


        else:
            '''manipulation'''
            if self.torque_history != []:  # is not empty
                self.friction_change_count += 1
                # print("friction change count: ", self.friction_change_count)

                # plot
                if self.friction_change_count % 200 == 0 or self.friction_change_count == 1:
                    print("plot--------------------------------------------------------------------------------------")
                    plot_numbers(self.torque_history,
                                 '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                                 'torque')
                    # plot_numbers(self.error_history,
                    #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                    #              'height error')
                    # plot_numbers(self.angle_history,
                    #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                    #              'angle')
                    # plot_numbers_two_value(self.vel_history_1, self.vel_history_2,
                    #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                    #              'velocity')
                    plot_numbers_two_value(self.joint_left_history, self.joint_right_history,
                                 '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                                 'two joints angle')

                # empty the list
                self.torque_history = []
                self.error_history = []
                self.angle_history = []
                self.vel_history_1 = []
                self.vel_history_2 = []
                self.joint_left_history = []
                self.joint_right_history = []

            self.step_count_for_friction_change = 0
            # print("manipulation")
            # print("torque magnitude: ", self.data.ctrl[self.torque_idx])
            # print(self.goal)
            # print("manipulation idx in hand_env",self.torque_idx, self.pos_idx)
            self.terminate_count = 0
            if self.firstEpisode:
                self.data.ctrl[1] = 0  # change friction if stuck
                self.data.ctrl[0] = 0
            elif action[1] == 0:
                # rotation
                self.data.ctrl[self.torque_idx] = 0.5
                self.data.ctrl[self.pos_idx] = action[0]
            else:
                # left finger low/torque, right finger high/pos
                self.data.ctrl[self.torque_idx] = 1
                # self.data.ctrl[self.pos_idx] = action[0] # change friction if stuck
                self.data.ctrl[self.pos_idx] = self.data.qpos[self.pos_idx * 2 + 1] + (action[0] + 1) / 2 * 1
                print("toque is: ", self.data.ctrl[self.torque_idx], "action is: ", self.data.ctrl[self.pos_idx])

            # ctrlrange[0, 1] is because only single motor is controlled by action

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


    def _set_action_fixed_magnitude(self):

        ctrlrange = self.model.actuator_ctrlrange

        # if self.torque_history != []:  # is not empty
        #     self.friction_change_count += 1

            # plot
            # if self.friction_change_count % 200 == 0 or self.friction_change_count == 1:
            #     print("plot--------------------------------------------------------------------------------------")
                # plot_numbers(self.torque_history,
                #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                #              'torque')
                # plot_numbers(self.error_history,
                #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                #              'height error')
                # plot_numbers(self.angle_history,
                #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                #              'angle')
                # plot_numbers_two_value(self.vel_history_1, self.vel_history_2,
                #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                #              'velocity')
                # plot_numbers_two_value(self.joint_left_history, self.joint_right_history,
                #              '/Users/qiyangyan/Desktop/LearnFrictionChange/plot/torque_history',
                #              'two joints angle')

            # empty the list
            # self.torque_history = []
            # self.error_history = []
            # self.angle_history = []
            # self.vel_history_1 = []
            # self.vel_history_2 = []
            # self.joint_left_history = []
            # self.joint_right_history = []

        '''manipulation'''
        self.step_count_for_friction_change = 0
        self.terminate_count = 0
        if self.firstEpisode:
            self.data.ctrl[1] = 0  # change friction if stuck
            self.data.ctrl[0] = 0
        else:
            # left finger low/torque, right finger high/pos
            self.data.ctrl[self.torque_idx] = 1
            self.data.ctrl[self.pos_idx] = self.data.qpos[self.pos_idx * 2 + 1] - 0.01
            # print("toque is: ", self.data.ctrl[self.torque_idx], "action is: ", self.data.ctrl[self.pos_idx])

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


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
