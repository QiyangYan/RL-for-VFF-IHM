"""
TODO:
1. PT test

SI RESULT:
RIGHT

LEFT
Torque: [0.18494267, 0.02372372, 0.03176467]
Position: [4.99979599, 0.39951026, 12.87708353, 0.03]

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import re

from scipy.optimize import basinhopping
from scipy.optimize import Bounds

from dynamixel_driver.angle_conversion import AngleConversion
# from Real_executor import Dynamixel_Driver
from Sim_executor import MuJoCo_Simulation_executor
from DomainRandomisation.actuator_dr import ActuatorDR


# 定义要最小化的目标函数

def value_censorship(new_solution):
    # Positive value
    new_solution[0] = abs(new_solution[0])
    new_solution[1] = abs(new_solution[1])
    return new_solution


class Parameter_Tuning:
    def __init__(self):
        # initialize
        self.is_render_Mujoco = False
        self.is_render_Window = False
        self.is_render_Result = True

        self.niter = 100
        self.real_data_samples_num = 100

        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]

        # left position:
        self.trajectory_list = [
            [1, 10, 140, 13, 15, 11, 245, 225, 204, 234, 16, 42, 57, 141, 245, 248, 10, 10, 10],
            [1, 10, 100, 10, 10, 10, 200, 245, 10, 40, 50, 140, 245, 245, 10, 10, 10],  # calibration trajectory
            [265, 210, 150, 42, 15, 48, 105, 230, 243, 201, 246, 205],
            [215, 210, 245, 204, 245, 189, 213, 43, 123, 224],
            [245, 2, 12, 45, 294, 43, 231, 34, 124, 142, 23, 64, 95, 21],
            [3, 43, 45, 58, 123, 59, 22, 124, 50, 23, 163, 145, 94, 48, 194, 205]
        ]
        self.ID = 1
        self.torque = False
        self.withObject = False
        self.initial_prms = [ 5.     ,     0.92183252, 12.02150098,  0.03        ]
        # self.initial_prms = [ 8.24341447 , 1.54178521, 30. ,         0.19671346]
        self.lb = [0, 0, 0, 0]
        self.ub = [5, 10, 30, 1]
        self.env = gym.make("VariableFriction-v3")
        # self.env = gym.make("VariableFriction-v3", render_mode="human")
        # MULTIPLE TRAJECTORY
        self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Multiple IHM-like Trajectory/Set2'
        # IHM-like trajectory
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration IHM-like Trajectory/Model_Dynamics_20240305_181840.csv'
        # Step Response & Sine
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/L_Model_Dynamics_20240304_122832.csv'
        # self.real_record_path_sin = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/L_Model_Dynamics_sinusoidal_20240304_115947.csv'

        # right:
        # remember to: change the hand_env in variable_friction_for_calibration when change the hand for calibration
        # self.trajectory_list = [
        #     [1, 10, 100, 10, 10, 10, 200, 245, 10, 40, 50, 140, 245, 245, 10, 10, 10]
        # ]
        # self.ID = 1
        # self.initial_prms = [4.99979599, 0.39951026, 12.87708353, 0.03]
        # self.lb = [0, 0, -30, 0]
        # self.ub = [10, 10, 30, 2]
        # self.torque = False
        # self.withObject = False
        # self.env = gym.make("VariableFriction-v3")
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration Trajectories/XM430 Calibration 8 - 0.08s/R_Model_Dynamics_20240229_210133.csv'
        # # self.real_record_path_sin = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/R_Model_Dynamics_sinusoidal_20240304_104604.csv'
        # # self.real_record_path_sin = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_20240304_223628.csv'

        # left torque
        # remember to: change the swap the shared.xml with shared_torque.xml
        # self.ID = 0
        # self.torque = True
        # self.withObject = False
        # self.initial_prms = [0, 0, 0]
        # self.lb = [0, 0, 0]
        # self.ub = [5, 10, 1]
        # self.env = gym.make("VariableFriction-v3")
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration IHM-like Trajectory/Model_Dynamics_torque_20_20240305_221627.csv'

        # friction with object
        # self.ID = 0
        # self.torque = False
        # self.withObject = True
        # self.initial_prms = 0
        # self.lb = 0
        # self.ub = 1
        # self.env = gym.make("VariableFriction-v2")
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_20240307_204451.csv'

        self.saving_path = './Sim2Real'
        self.optimizer = ''  #basinhopping  if optimizer = '', skip optimization

        self.bounds = Bounds(self.lb, self.ub)
        # add executors
        self.AngleConvert = AngleConversion()
        self.dynamixel_driver = None
        self.calibration = MuJoCo_Simulation_executor(env_=self.env)
        self.calibration_withObject = ActuatorDR(env_=self.env)

        self.minimizer_kwargs = {
            "bounds": self.bounds,
        }

        self.result = None

        self.real_time_list = []
        self.real_pos_list = []
        self.real_vel_list = []
        self.real_ctrl_list = []

        if self.real_record_path != '':
            # FOR OTHERS
            if len(self.trajectory_list)>1:
                # MULTIPLE TRAJECTORIES
                trajectories_data = self.read_real_data_list(self.real_record_path)
                for i in trajectories_data:
                    self.real_time_list.append(trajectories_data[i]['time'])
                    self.real_pos_list.append(trajectories_data[i]['position'])
                    self.real_vel_list.append(trajectories_data[i]['velocity'])
                    self.real_ctrl_list.append(trajectories_data[i]['control'])

            else:
                # FOR OTHERS
                if not self.torque:
                    full_time, full_pos, full_vel, full_ctrl = self.read_real_data(self.real_record_path)
                    self.real_ctrl = full_ctrl
                else:
                    full_time, full_pos, full_vel = self.read_real_data(self.real_record_path)
                self.real_time = full_time
                self.real_pos = full_pos
                self.real_vel = full_vel

            # FOR SIN
            # full_time_sin, full_pos_sin, full_vel_sin = self.read_real_data(self.real_record_path_sin)
            # self.real_time_sin = full_time_sin[:100]
            # self.real_pos_sin = full_pos_sin[:100]
            # self.real_vel_sin = full_vel_sin[:100]
        else:
            pass

        self.sim_time = None
        self.sim_pos = None
        self.sim_vel = None
        self.sim_control = None

        self.sim_time_sin = None
        self.sim_pos_sin = None
        self.sim_vel_sin = None

        self.itp_sim_pos = None
        self.itp_sim_pos_sin = None

        self.number_of_iter = 0
        self.best_parameter = None
        self.best_loss = np.inf

        self.t_list = []
        self.sim_present_position_list = []
        self.sim_present_vel_list = []
        self.sim_goal_position_list = []

        self.paras_tuning()

    def paras_tuning(self):

        if self.optimizer == 'basinhopping':
            print("running")
            self.result = basinhopping(
                self.objective_function,
                self.initial_prms,
                minimizer_kwargs=self.minimizer_kwargs,
                callback=self.callback_function
            )
            if self.result.success:
                if self.result.fun > self.best_loss:
                    self.result.x = self.best_parameter
                    self.result.fun = self.best_loss
                print("X values:\t", self.result.x)
                print("minimum loss:\t", self.result.fun)
                if self.is_render_Result:
                    self.objective_function(self.result.x)
                    self.plt_result()
            else:
                print(self.result.message)
        elif self.optimizer == '':
            print("Using Initial parameters without optimization")
            loss = self.objective_function(self.initial_prms)
            print(loss)
            if len(self.trajectory_list)>1:
                self.plt_combined_results_2x2_list()
            else:
                self.plt_combined_results_2x2()
        else:
            print("Wrong_Name")

    def read_real_data(self, real_record_path):
        df = pd.read_csv(real_record_path)
        position = df['Actual Position']
        real_time = df['t']
        velocity = df['Actual Velocity']
        if not self.torque:
            control = df['Goal Position']
            return np.array(real_time), np.array(position), np.array(velocity), np.array(control)
        return np.array(real_time), np.array(position), np.array(velocity)

    @staticmethod
    def read_real_data_list(folder_path):
        # List all CSV files in the specified folder
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        # Sort files by the rank index (last number in the filename before .csv)
        csv_files.sort(key=lambda x: int(re.search(r'(\d+).csv$', x).group(1)))

        # Dictionary to store data based on rank index
        data_store = {}

        # Read each file and store the data
        for csv_file in csv_files:
            rank_index = int(re.search(r'(\d+).csv$', csv_file).group(1))
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)

            position = df['Actual Position'].to_numpy()
            real_time = df['t'].to_numpy()
            velocity = df['Actual Velocity'].to_numpy()
            control = df['Goal Position'].to_numpy()

            data_store[rank_index] = {
                'time': real_time,
                'position': position,
                'velocity': velocity,
                'control': control
            }

        return data_store

    def objective_function(self, prms):
        loss_sin = 0
        if self.ID == 1:  # RIGHT
            if self.torque:  # TORQUE CALIBRATION
                self.calibration.adjust_parameter_right(damping=prms[0],
                                                        armature=prms[1],
                                                        frictionloss=prms[2],
                                                        torque=True)
            elif self.withObject:  # OBJECT SLIDES ON RIGHT FINGER
                self.calibration_withObject.adjust_parameter_right_object(friction=prms)
            else:  # POSITION CALIBRATION OF RIGHT FINGER WITHOUT OBJECT
                assert len(prms) == 4, f"wrong prms length, check: {prms}"
                self.calibration.adjust_parameter_right(damping=prms[0],
                                                        armature=prms[1],
                                                        gainprm=prms[2],
                                                        frictionloss=prms[3])
        else:  # LEFT
            assert self.ID == 0, f"wrong ID, check: {self.ID}"
            if self.torque:  # TORQUE CALIBRATION
                self.calibration.adjust_parameter_left(damping=prms[0],
                                                       armature=prms[1],
                                                       frictionloss=prms[2],
                                                       torque=True)
            elif self.withObject:  # OBJECT SLIDES ON LEFT FINGER
                self.calibration_withObject.adjust_parameter_left_object(friction=prms)
            else:  # POSITION CALIBRATION OF LEFT FINGER WITHOUT OBJECT
                assert len(prms) == 4, f"wrong prms length, check: {prms}"
                self.calibration.adjust_parameter_left(damping=prms[0],
                                                       armature=prms[1],
                                                       gainprm=prms[2],
                                                       frictionloss=prms[3])

        if self.torque:  # torque control
            self.sim_time, self.sim_pos, self.sim_vel = self.calibration.torque(self.real_time[-1], ID=self.ID)
        elif self.withObject:  # OBJECT SLIDES ON LEFT FINGER
            self.sim_time, self.sim_pos, self.sim_vel, self.sim_control, _, _ = self.calibration_withObject.slide_trajectory(self.env)
        elif len(self.trajectory_list) == 1:  # position control
            self.sim_time, self.sim_pos, self.sim_vel, self.sim_control = \
                self.calibration.manual_policy(ID=self.ID,
                                               goal_pos_list_dyn=self.trajectory_list[0],
                                               real_time_=self.real_time[-1])
        else:  # MULTIPLE TRAJECTORIES
            loss_list = []
            for i, goal_pos_list in enumerate(self.trajectory_list):
                # print(self.real_time_list[i][-1])
                self.sim_time, self.sim_pos, self.sim_vel, self.sim_control = \
                    self.calibration.manual_policy(ID=self.ID,
                                                   goal_pos_list_dyn=goal_pos_list,
                                                   real_time_=self.real_time_list[i][-1])

                self.t_list.append(self.sim_time)
                self.sim_present_position_list.append(self.sim_pos)
                self.sim_present_vel_list.append(self.sim_vel)
                self.sim_goal_position_list.append(self.sim_control)

                self.itp_sim_pos = np.interp(self.real_time_list[i], self.sim_time, self.sim_pos)
                loss_list.append(np.sum(np.abs(np.array(self.real_pos_list[i]) - np.array(self.itp_sim_pos))))

                self.plt_combined_results_2x2_list(i)

            return np.sum(loss_list)

        '''This is the old version that produce unsatisfactory SI result'''
        # self.sim_time, self.sim_pos, self.sim_vel = self.calibration.step_response(self.real_time[-1], ID=self.ID)
        # self.sim_time_sin, self.sim_pos_sin, self.sim_vel_sin, _ = self.calibration.sin_response(self.real_time_sin[-1], ID=self.ID)
        # self.itp_sim_pos_sin = np.interp(self.real_time_sin, self.sim_time_sin, self.sim_pos_sin)
        # loss_sin = np.sum(np.abs(np.array(self.real_pos_sin) - np.array(self.itp_sim_pos_sin)))

        self.itp_sim_pos = np.interp(self.real_time, self.sim_time, self.sim_pos)
        loss = np.sum(np.abs(np.array(self.real_pos) - np.array(self.itp_sim_pos)))

        return loss + loss_sin

    def callback_function(self, x, f, accepted):
        self.number_of_iter += 1
        print(f"Iter:{self.number_of_iter}, accepted:{accepted}, prms:{x}, loss:{f}")
        if self.is_render_Window:
            pass
        if self.best_loss > f:
            self.best_loss = f
            self.best_parameter = x
        if self.number_of_iter >= self.niter:
            return True

    def plt_combined_results_2x2(self):
        # Create a 2x2 grid for plotting
        fig, axs = plt.subplots(2, 2, figsize=(8, 5))  # 2 rows, 2 columns

        # Plot for regular position on the first subplot (top left)
        axs[0, 0].plot(self.real_time, self.real_pos, label='Real Position', color='r')
        axs[0, 0].plot(self.sim_time, self.sim_pos, label='Simulated Position', color='g')
        axs[0, 0].hlines(y=self.MAX_POS[self.ID], xmin=0, xmax=self.real_time[-1], colors='b',
                         linestyles='--',
                         label='max_position',
                         linewidth=1)
        axs[0, 0].hlines(y=self.MIN_POS[self.ID], xmin=0, xmax=self.real_time[-1], colors='b',
                         linestyles='--',
                         label='min_position',
                         linewidth=1)
        if not self.torque:
            axs[0, 0].plot(self.sim_time, self.sim_control, label='Sim Control', color='k', linestyle="--")
            axs[0, 0].plot(self.real_time, self.real_ctrl, label='Real Control', color='b', linestyle='--')
        axs[0, 0].set_title('XM430 Regular Position Over Time')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Position')
        axs[0, 0].legend()

        # Plot for regular velocity on the second subplot (top right)
        axs[0, 1].plot(self.real_time, self.real_vel, label='Real Velocity', color='k')
        axs[0, 1].plot(self.sim_time, self.sim_vel, label='Simulated Velocity', color='b')
        axs[0, 1].set_title('XM430 Regular Velocity Over Time')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Velocity')
        axs[0, 1].legend()

        # Plot for sine position on the third subplot (bottom left)
        # axs[1, 0].plot(self.real_time_sin, self.real_pos_sin, label='Real Position', color='r')
        # axs[1, 0].plot(self.sim_time_sin, self.sim_pos_sin, label='Simulated Position', color='g')
        # axs[1, 0].set_title('XM430 Sin Position Over Time')
        # axs[1, 0].set_xlabel('Time (s)')
        # axs[1, 0].set_ylabel('Position')
        # axs[1, 0].legend()
        #
        # # Plot for sine velocity on the fourth subplot (bottom right)
        # axs[1, 1].plot(self.real_time_sin, self.real_vel_sin, label='Real Velocity', color='k')
        # axs[1, 1].plot(self.sim_time_sin, self.sim_vel_sin, label='Simulated Velocity', color='b')
        # axs[1, 1].set_title('XM430 Sin Velocity Over Time')
        # axs[1, 1].set_xlabel('Time (s)')
        # axs[1, 1].set_ylabel('Velocity')
        # axs[1, 1].legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    def plt_combined_results_2x2_list(self, idx):
        # Create a 2x2 grid for plotting
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 2 columns

        # Plot for regular position on the first subplot (top left)
        axs[0].plot(self.real_time_list[idx], self.real_pos_list[idx], label='Real Position', color='r')
        axs[0].hlines(y=self.MAX_POS[self.ID], xmin=0, xmax=self.real_time_list[idx][-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[self.ID], xmin=0, xmax=self.real_time_list[idx][-1], colors='b', linestyles='--', label='min_position',
                      linewidth=1)
        if not self.torque:
            axs[0].plot(self.sim_time, self.sim_control, label='Sim Control', color='k', linestyle="--")
            axs[0].plot(self.real_time_list[idx], self.real_ctrl_list[idx], label='Real Control', color='b', linestyle='--')
        axs[0].plot(self.sim_time, self.sim_pos, label='Simulated Position', color='g')
        axs[0].set_title('XM430 Regular Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        # Plot for regular velocity on the second subplot (top right)
        axs[1].plot(self.real_time_list[idx], self.real_vel_list[idx], label='Real Velocity', color='k')
        axs[1].plot(self.sim_time, self.sim_vel, label='Simulated Velocity', color='b')
        axs[1].set_title('XM430 Regular Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


# 调用 basinhopping 函数
if __name__ == "__main__":
    result = Parameter_Tuning()
