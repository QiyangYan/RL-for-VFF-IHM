import numpy as np
import gymnasium as gym
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from angle_conversion import AngleConversion

AngleConvert = AngleConversion()

class MuJoCo_Parameter_Optimisation():
    def __init__(self, render):
        if render is True:
            self.env = gym.make("VariableFriction-v3", render_mode="human")
        else:
            assert render is False, f"render is not bool: {render}"
            self.env = gym.make("VariableFriction-v3")
        self.render = render
        self.env.reset()
        self.MAX_POS = [1445, 2651]  # right, left [127.002, 232.998]
        self.MIN_POS = [2671, 1425]  # [234.756, 125.244]

    def env_test_right_finger(self):
        actions = [0, 1.8807, 0]
        start_pos_right = AngleConvert.xm_2_rad(self.MIN_POS[0])
        for action in actions:
            for _ in range(50):
                obs, _, _, _, _ = self.env.step(action)
                current_pos = obs["observation"][2]
                print(AngleConvert.rad_2_xm(start_pos_right - current_pos), obs["observation"][2])

    def get_trajectory_sinusoidal(self, ID=0):
        self.env.reset()

        time_duration = 5  # Total time of simulation in seconds
        sample_rate = 80  # How many samples per second
        min_freq = 0.1  # Minimum frequency in Hz
        max_freq = 1  # Maximum frequency in Hz

        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = np.linspace(min_freq, max_freq, len(t))
        sine_wave = np.array([np.sin(2 * np.pi * freq[i] * t[i]) for i in range(len(t))])
        normalized_sine_wave = self.MIN_POS[ID] + (sine_wave + 1) * (self.MAX_POS[ID] - self.MIN_POS[ID]) / 2

        'get ready for testing'
        position = 2048
        self.env.step(AngleConvert.xm_2_rad())
        time.sleep(3)

        'testing start'
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_t = time.time()
        for pos in normalized_sine_wave:
            actual_position.append(self.xm_posControl(0, int(pos), reset=False))
            t.append(time.time() - start_t)
            unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                    self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit_buf.append(self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                  self.XM['ADDR_PROFILE_VELOCITY'])[0])
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            actual_vel.append(signed_current_vel)
            # print(signed_current_vel, t[-1])

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': normalized_sine_wave,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            'Velocity Limit': vel_limit_buf
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_vary_f_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].plot(t, normalized_sine_wave, label='Goal Position (Control Signal)', color='b', linestyle='--',
                    linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='max_position', linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='min_position', linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        # axs[1].plot(t, vel_limit_buf*(-1), color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_vary_f_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

    def max_min_position_dynamics(self, ID):
        '''
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''

        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_pos_right = AngleConvert.xm_2_rad(self.MIN_POS[0])

        action = 1.8807
        start_time = time.time()
        count = 0
        while True:
            count += 1
            obs, _, _, _, _ = self.env.step(action)
            t.append(time.time() - start_time)
            current_pos = obs["observation"][2]
            current_vel = - obs["observation"][6]

            goal_position.append(self.MAX_POS[ID])
            actual_position.append(AngleConvert.rad_2_xm(start_pos_right - current_pos))
            actual_vel.append(AngleConvert.xm_rad_per_sec_to_rpm(current_vel))
            # vel_limit_buf.append(vel_limit)

            if abs(current_pos - action) < AngleConvert.xm_2_rad(2):
                break

        print("time taken for the simulation is: ", t[-1], count)

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': goal_position,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            # 'Velocity Limit': vel_limit_buf
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.csv"
        # df.to_csv(data_filename, index=False)

        if self.render is False:
            t = np.array(t)*358.15

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='min_position',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position', linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].hlines(y=-50, xmin=0, xmax=t[-1], colors='b', linestyles='--', label='min_position',
                      linewidth=1)
        # axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        # fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()


    def get_parameters_right_finger(self):
        damping = self.env.model.dof_damping[3]
        armature = self.env.model.dof_armature[3]
        friction_loss = self.env.model.dof_frictionloss[3]
        force_range = self.env.model.actuator_forcerange
        gear = self.env.model.actuator_gear[1][0]

        # print(self.env.model.dof_damping)
        # print(self.env.model.dof_armature)
        # print(self.env.model.dof_frictionloss)
        # print(self.env.model.jnt_stiffness)

        print("Damping: ", damping)
        print("Armature: ", armature)
        print("Frictionloss: ", friction_loss)
        print("Force Range: ", force_range)
        print("Gear: ", gear)

        print(self.env.model.actuator_gainprm[1][0])
        print(self.env.model.actuator_biastype[1])
        print(self.env.model.actuator_biasprm[1][1])

        # print(self.env.model.actuator_gainprm)
        # print(self.env.model.actuator_biastype)
        # print(self.env.model.actuator_biasprm)
        # print(self.env.model.actuator_ctrlrange)

    def adjust_parameter(self, damping=1.084, armature=0.045, frictionloss=0.03, gainprm=21.1, biastype=1, forcerange=1.3, gear=1):
        '''
        Actuator
            :param forcerange: -1.3 ~ 1.3
            :param kp:

        Joint
            :param damping:
            :param armature: increase
            :param frictionloss:
        '''
        self.env.model.actuator_gear[1][0] = gear
        self.env.model.dof_damping[3] = damping
        self.env.model.dof_armature[3] = armature
        self.env.model.dof_frictionloss[3] = frictionloss

        self.env.model.actuator_gainprm[1][0] = gainprm
        self.env.model.actuator_biastype[1] = biastype
        self.env.model.actuator_biasprm[1][1] = -gainprm
        self.env.model.actuator_forcerange = [-forcerange, forcerange]



calibration = MuJoCo_Parameter_Optimisation(False)



'TEST 1'
# calibration.env_test_right_finger()

'TEST 2'
calibration.adjust_parameter()
calibration.max_min_position_dynamics(0)

'TEST 3'
# calibration.get_parameters_right_finger()