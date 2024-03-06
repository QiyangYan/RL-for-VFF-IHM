"""
Used for: Testing the parameters obtained in Parameter_Tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping
from scipy.optimize import Bounds

from dynamixel_driver.angle_conversion import AngleConversion
from Sim_executor import MuJoCo_Simulation_executor


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
        # self.initial_prms = [0.61543616, 0.58330355, 0.83544602]

        # RIGHT POSITION
        # self.ID = 1
        # self.initial_prms = [1.0865664, 0.36525921, 3.71816819]
        # self.initial_prms = [9.50167428, 3.9701131, 26.471617, 11.76470498, 0.]
        # self.initial_prms = [4.04170475, 1.77941277, 3.38524575, 30., 0.06679021]
        # self.initial_prms = [0.65246879, 0.6145034 , 0.87549634, 0.        ] # only step response
        # self.initial_prms = [1.22424895, 1.56275461, 1.75184656, 0.        ]

        # right
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/R_Model_Dynamics_20240229_210133.csv'
        # self.real_record_path_sin = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_20240304_223628.csv'

        'LEFT POSITION'
        # self.ID = 0
        # self.torque = False
        # self.lb = [0, 0, -30, 0]
        # self.ub = [10, 10, 30, 2]
        # self.initial_prms = [4.99979599, 0.39951026, 12.87708353, 0.03]  # good parameter + stiffer score: 4360
        # # self.initial_prms = [ 4.99979599,  0.39951026 ,12.87708353 , 0.01966778]  # good parameter score: 4150
        # # IHM-like trajectory
        # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration IHM-like Trajectory/Model_Dynamics_20240305_181840.csv'
        # # Step response and Sine
        # # self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/L_Model_Dynamics_20240304_122832.csv'
        # # self.real_record_path_sin = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 8 - 0.08s/L_Model_Dynamics_sinusoidal_20240304_115947.csv'

        'LEFT TORQUE'
        self.torque = True
        self.ID = 0
        self.initial_prms = [0.18494267, 0.02372372, 0.03176467]
        self.lb = [0, 0, -30]
        self.ub = [10, 10, 30]
        self.real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration IHM-like Trajectory/Model_Dynamics_torque_20_20240305_221627.csv'

        'Other Parameters'
        self.niter = 100
        self.real_data_samples_num = 400

        self.saving_path = './Sim2Real'
        # if optimizer = '', skip optimization
        self.optimizer = ''  #basinhopping

        self.bounds = Bounds(self.lb, self.ub)
        # add executors
        self.AngleConvert = AngleConversion()
        self.dynamixel_driver = None
        self.calibration = MuJoCo_Simulation_executor(self.is_render_Mujoco)

        self.minimizer_kwargs = {
            "bounds": self.bounds,
        }

        self.result = None

        if self.real_record_path != '':
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

        self.sim_time_sin = None
        self.sim_pos_sin = None
        self.sim_vel_sin = None
        self.sim_control_sin = None

        self.itp_sim_pos = None
        self.itp_sim_pos_sin = None

        self.number_of_iter = 0
        self.best_parameter = None
        self.best_loss = np.inf

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
                    self.plt_combined_results_2x2()
            else:
                print(self.result.message)
        elif self.optimizer == '':
            print("Using Initial parameters without optimization")
            loss = self.objective_function(self.initial_prms)
            print(loss)
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

    def objective_function(self, prms):
        loss_sin = 0
        if self.ID == 1:
            if self.torque:
                self.calibration.adjust_parameter_right(damping=prms[0], armature=prms[1], frictionloss=prms[2],
                                                        torque=True)
            else:
                assert len(prms) == 4, f"wrong prms length, check: {prms}"
                self.calibration.adjust_parameter_right(damping=prms[0], armature=prms[1], gainprm=prms[2],
                                                        frictionloss=prms[3])
        else:
            assert self.ID == 0, f"wrong ID, check: {self.ID}"
            if self.torque:
                self.calibration.adjust_parameter_left(damping=prms[0], armature=prms[1], frictionloss=prms[2],
                                                       torque=True)
            else:
                assert len(prms) == 4, f"wrong prms length, check: {prms}"
                self.calibration.adjust_parameter_left(damping=prms[0], armature=prms[1], gainprm=prms[2],
                                                       frictionloss=prms[3])

        if self.torque:  # torque control
            self.sim_time, self.sim_pos, self.sim_vel = self.calibration.torque(self.real_time[-1], ID=self.ID)
        else:  # position control
            self.sim_time, self.sim_pos, self.sim_vel, self.sim_control = self.calibration.manual_policy(
                self.real_time[-1], ID=self.ID)

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
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))  # 2 rows, 2 columns

        # Plot for regular position on the first subplot (top left)
        axs[0].plot(self.real_time, self.real_pos, label='Real Position', color='r')
        if not self.torque:
            axs[0].plot(self.sim_time, self.sim_control, label='Goal Position', color='k', linestyle="--")
            axs[0].plot(self.real_time, self.real_ctrl, label='Real Control', color='b', linestyle='--')
        axs[0].plot(self.sim_time, self.sim_pos, label='Simulated Position', color='g')
        axs[0].set_title('XM430 Regular Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        # Plot for regular velocity on the second subplot (top right)
        axs[1].plot(self.real_time, self.real_vel, label='Real Velocity', color='k')
        axs[1].plot(self.sim_time, self.sim_vel, label='Simulated Velocity', color='b')
        axs[1].set_title('XM430 Regular Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()

        # Plot for sine position on the third subplot (bottom left)
        # axs[1, 0].plot(self.real_time_sin, self.real_pos_sin, label='Real Position', color='r')
        # axs[1, 0].plot(self.sim_time_sin, self.sim_pos_sin, label='Simulated Position', color='g')
        # # axs[1, 0].plot(self.sim_time_sin, self.sim_control_sin, label='Simulated Control', color='k')
        # axs[1, 0].set_title('XM430 Sin Position Over Time')
        # axs[1, 0].set_xlabel('Time (s)')
        # axs[1, 0].set_ylabel('Position')
        # axs[1, 0].legend()

        # Plot for sine velocity on the fourth subplot (bottom right)
        # axs[1, 1].plot(self.real_time_sin, self.real_vel_sin, label='Real Velocity', color='k')
        # axs[1, 1].plot(self.sim_time_sin, self.sim_vel_sin, label='Simulated Velocity', color='b')
        # axs[1, 1].set_title('XM430 Sin Velocity Over Time')
        # axs[1, 1].set_xlabel('Time (s)')
        # axs[1, 1].set_ylabel('Velocity')
        # axs[1, 1].legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


# 调用 basinhopping 函数
if __name__ == "__main__":
    result = Parameter_Tuning()

