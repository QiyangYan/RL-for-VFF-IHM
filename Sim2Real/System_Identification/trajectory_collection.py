import time

from dynamixel_sdk import *
import dynamixel_sdk as dxl
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from dynamixel_driver.bulk_read_write import BULK
from dynamixel_driver.dynamixel_control import Dynamixel_Driver


class TRAJECTORIES(BULK):
    def __init__(self):
        super().__init__()

    def max_min_position_dynamics(self, ID):
        '''
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        self.xm_posControl(ID, self.MAX_POS[ID])
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []

        self.xm_posControl(ID, self.MIN_POS[ID], profile_vel=50, acceleration_time=5, reset=True)
        start_time = time.time()
        while True:
            '''
            Condition: 
            ID1: Present Position(132, 0x0090, 2[byte]) = 119(0x0077)
            ID4: Present Velocity(128, )
            '''
            observation = self.get_obs_dynamixel()
            current_pos = observation[2]
            current_vel = observation[6]
            t.append(time.time() - start_time)

            vel_limit_buf.append(50)
            goal_position.append(self.MIN_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(current_vel)
            if abs(current_pos - self.MIN_POS[ID]) < 2:
                break

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
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].plot(t, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def max_min_position_dynamics_negative(self, ID):
        '''
        consider negative velocity
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        self.xm_posControl(ID, self.MIN_POS[ID])
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        commend_time = time.time()
        self.xm_posControl(ID, self.MAX_POS[ID], profile_vel=50, acceleration_time=5, reset=True)
        start_time = time.time()
        while True:
            observation = self.get_obs_dynamixel()
            current_pos = observation[ID*2]
            unsigned_current_vel = observation[ID*2+4]
            t.append(time.time() - start_time)

            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            goal_position.append(self.MAX_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(-50)
            if abs(current_pos - self.MAX_POS[ID]) < 2:
                break

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': goal_position,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            'Velocity Limit': vel_limit_buf,
            'commend_time': start_time - commend_time
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def model_calibration_sinusoidal(self, ID):
        time_duration = 2  # Total time of simulation in seconds
        sample_rate = 110  # How many samples per second
        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = 0.5
        sine_wave = np.array([-np.cos(2 * np.pi * freq * t[i]) for i in range(len(t))])
        normalized_sine_wave = self.MIN_POS[ID] + (sine_wave + 1) * (self.MAX_POS[ID] - self.MIN_POS[ID]) / 2
        # plt.plot(t, sine_wave)
        # plt.show()

        'get ready for testing'
        self.xm_posControl(ID, self.MIN_POS[ID], reset=True)
        time.sleep(3)

        'testing start'
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start = time.time()
        for i, pos in enumerate(normalized_sine_wave):
            # step for 0.04s in environment
            start_time = time.time()
            first = True
            while True:
                if first:
                    s = time.time()
                    self.xm_posControl(ID, int(pos), reset=False)  # 0.016
                    first = False

                if abs(time.time() - start_time - 0.44) > 0.016:
                    observation = self.get_obs_dynamixel()  # 0.016
                    current_pos = observation[ID*2]
                    unsigned_current_vel = observation[4+ID*2]
                    signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)

                if time.time() - start_time > 0.44:
                    vel_limit_buf.append(50)
                    actual_position.append(current_pos)
                    actual_vel.append(signed_current_vel)
                    t.append(time.time() - start)
                    break
                    # print(time.time() - start_time)
                    # t.append

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': normalized_sine_wave,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            'Velocity Limit': vel_limit_buf
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].plot(t, normalized_sine_wave, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
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
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_sinusoidal_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def model_calibration_sinusoidal_vary_f(self, ID):
        time_duration = 5  # Total time of simulation in seconds
        sample_rate = 80  # How many samples per second
        min_freq = 0.1  # Minimum frequency in Hz
        max_freq = 1  # Maximum frequency in Hz

        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = np.linspace(min_freq, max_freq, len(t))
        sine_wave = np.array([np.sin(2 * np.pi * freq[i] * t[i]) for i in range(len(t))])
        normalized_sine_wave = self.MIN_POS[ID] + (sine_wave + 1) * (self.MAX_POS[ID] - self.MIN_POS[ID]) / 2

        'get ready for testing'
        self.xm_posControl(0, 2048)
        time.sleep(3)

        'testing start'
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_t = time.time()
        for pos in normalized_sine_wave:
            actual_position.append(self.xm_posControl(0, int(pos), reset=False))
            t.append(time.time()-start_t)
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
        axs[0].plot(t, normalized_sine_wave, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
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

        # plt.show()

    def model_calibration_fixed_torque(self, ID):
        self.xm_posControl(ID, self.MIN_POS[ID])
        time.sleep(3)
        t = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        self.xm_torque_control(ID)
        start_t = time.time()
        while True:
            observation = self.get_obs_dynamixel()  # 0.016
            current_pos = observation[ID * 2]
            unsigned_current_vel = observation[4 + ID * 2]
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)

            t.append(time.time() - start_t)
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)

            if abs(current_pos - self.MAX_POS[ID]) < 50:
                self.xm_torque_control(ID, 0)
                break

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            # 'Velocity Limit': vel_limit_buf
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_torque_20_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time with Torque = -20')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        # axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time with Torque = -20')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        plt.show()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_torque_20_{timestamp}.png'
        fig.savefig(image_filename)

    def model_calibration_fixed_torque_revert(self, ID):
        self.xm_posControl(ID, self.MAX_POS[0])
        time.sleep(3)
        t = []
        actual_position = []
        actual_vel = []
        start_t = time.time()
        vel_limit_buf = []
        self.xm_torque_control(ID)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_t)
            unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                    self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                         self.XM['ADDR_PROFILE_VELOCITY'])[0]
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(vel_limit)

            print(current_pos - self.MIN_POS[0])
            if abs(current_pos - self.MIN_POS[0]) < 50:
                self.xm_torque_control(ID, 0)
                break

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time with Torque = -20')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time with Torque = -20')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        plt.show()
        # image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        # fig.savefig(image_filename)

    def max_min_position_dynamics_negative_current_pos(self, ID):
        '''
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        # self.xm_posControl(0, self.MAX_POS[ID])
        self.xm_current_posControl(0, self.MIN_POS[ID], 0)
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_t = time.time()
        self.xm_current_posControl(0, self.MAX_POS[ID], 0)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_t)
            unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                    self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                         self.XM['ADDR_PROFILE_VELOCITY'])[0]
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            goal_position.append(self.MAX_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(-vel_limit)
            if abs(current_pos - self.MAX_POS[ID]) < 2:
                break

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': goal_position,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            'Velocity Limit': vel_limit_buf
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.csv"
        # df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        # axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        fig.show()
        # fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def step_response_bulk_read(self, ID):
        '''
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        self.xm_posControl(0, self.MAX_POS[ID])
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []

        start_time = time.time()
        self.xm_posControl(0, self.MIN_POS[ID], profile_vel=50, acceleration_time=5, reset=True)

        while True:
            '''
            Condition: 
            ID1: Present Position(132, 0x0090, 2[byte]) = 119(0x0077)
            ID4: Present Velocity(128, )
            '''
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_time)
            current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                         self.XM['ADDR_PROFILE_VELOCITY'])[0]
            vel_limit_buf.append(vel_limit)
            goal_position.append(self.MIN_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(current_vel)
            if abs(current_pos - self.MIN_POS[ID]) < 2:
                break

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
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].plot(t, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def bulk_read(self):

        dxl_comm_result = self.packetHandler.groupBulkReadTxRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            Exception("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 1:  # XM
                if self.packetHandler.groupBulkReadIsAvailable(self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'], 4) \
                        and self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.XM['ADDR_PRESENT_VELOCITY'], 4):
                    # Read the data
                    dxl_present_position = self.packetHandler.groupBulkReadGetData(self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'], 4)
                    dxl_present_velocity = self.groupBulkRead.getData(self.DXL_ID_aray[ID], self.XM['ADDR_PRESENT_VELOCITY'], 4)
                    print(f"ID: {self.DXL_ID_aray[ID]} Position: {dxl_present_position} Velocity: {dxl_present_velocity}")
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")


            else:  # XL
                if self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.XL['ADDR_PRO_PRESENT_POSITION'], 4) \
                        and self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.XL['ADDR_PRESENT_VELOCITY'], 4):
                    # Read the data
                    dxl_present_position = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XL['ADDR_PRO_PRESENT_POSITION'], 4)
                    dxl_present_velocity = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XL['ADDR_PRESENT_VELOCITY'], 4)
                    print(
                        f"ID: {self.DXL_ID_aray[ID]} Position: {dxl_present_position} Velocity: {dxl_present_velocity}")
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")

    def max_min_position_dynamics_with_obj(self, ID):
        '''
        consider negative velocity
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        self.xl_posControl(0, 1)
        self.xl_posControl(1, 1)

        self.xm_posControl(1-ID, self.MIN_POS[1-ID])
        self.xm_torque_control(ID, reset=True, goal_torque=5)
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []

        obs = self.get_obs_dynamixel()
        self.xm_posControl(ID, obs[ID*2], reset=True)
        self.xm_torque_control(1-ID, reset=True)
        self.xl_posControl(1-ID, 0)
        time.sleep(1)

        commend_time = time.time()
        goal = self.MIN_POS[ID] - 200 * (ID*2-1)
        self.xm_posControl(ID, goal)
        start_time = time.time()
        while True:
            observation = self.get_obs_dynamixel()
            current_pos = observation[ID*2]
            unsigned_current_vel = observation[ID*2+4]
            t.append(time.time() - start_time)

            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            goal_position.append(goal)
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(-50)
            if abs(current_pos - goal) < 2:
                break

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': goal_position,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            'Velocity Limit': vel_limit_buf,
            'commend_time': start_time - commend_time
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Slide Trajectory/Model_Dynamics_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=goal, xmin=0, xmax=t[-1], colors='b', linestyles='--',
                      label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=obs[ID*2], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Slide Trajectory/Model_Dynamics_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def move_random(self, ID):
        """
            The adjusted parameters are listed below:
            * profile_vel = 50
            * acceleration_time = 5
        """
        self.xm_posControl(ID, self.MAX_POS[ID], reset=True)
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []

        goal_pos_list = [1, 10, 100, 10, 10, 10, 200, 245, 10, 40, 50, 140, 245, 245, 10, 10, 10]

        current_pos = self.MAX_POS[ID]
        start = time.time()
        for i, pos in enumerate(goal_pos_list):
            s = time.time()
            goal_pos = current_pos + pos * (ID * 2 - 1)
            goal_pos = np.clip(goal_pos, self.MIN_POS[ID], self.MAX_POS[ID])
            print(goal_pos)
            self.xm_posControl(ID, goal_pos, reset=False)  # 0.016
            while True:
                if abs(time.time() - s - 0.44) > 0.016:
                    observation = self.get_obs_dynamixel()  # 0.016
                    current_pos = observation[ID*2]
                    unsigned_current_vel = observation[4+ID*2]
                    signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)

                    print(goal_pos)
                    goal_position.append(goal_pos)
                    actual_position.append(current_pos)
                    actual_vel.append(signed_current_vel)
                    t.append(time.time() - start)

                if time.time() - s > 0.44:
                    # goal_position.append(goal_pos)
                    # actual_position.append(current_pos)
                    # actual_vel.append(signed_current_vel)
                    # t.append(time.time() - start)
                    break

        'Save data'
        df = pd.DataFrame({
            't': t,
            'Goal Position': goal_position,
            'Actual Position': actual_position,
            'Actual Velocity': actual_vel,
            # 'Velocity Limit': vel_limit_buf
        })
        print(goal_position)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.csv"
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].plot(t, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='min_position',
                      linewidth=1)
        axs[0].set_title('XM430 Position Over Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        # Plot for velocity on the second subplot
        axs[1].plot(t, actual_vel, label='Velocity', color='k')
        # axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def torque(self, ID):
        pass


if __name__ == '__main__':

    dynamixel_driver = TRAJECTORIES()

    'Move to init position'
    print("--------------------------------------------------------")
    dynamixel_driver.xm_posControl(0, dynamixel_driver.MIN_POS[0])
    dynamixel_driver.xm_posControl(1, dynamixel_driver.MIN_POS[1])
    # dynamixel_driver.xl_posControl(0, 0)
    dynamixel_driver.xl_posControl(1, 0)
    time.sleep(2)
    print("--------------------------------------------------------")

    'TEST 1: Measure the time-taken for passing control signal or read signal'
    # dynamixel_driver.time_taken_for_control_read()

    'TEST 2: Measures the dynamic response of the finger to sinusoidal signla'
    # dynamixel_driver.model_calibration_sinusoidal(1)
    # dynamixel_driver.model_calibration_sinusoidal(0)
    # dynamixel_driver.model_calibration_sinusoidal_vary_f(1)

    'TEST 3: Measures the behaviour of the joints near their limits'
    # dynamixel_driver.max_min_position_dynamics(1)
    # dynamixel_driver.max_min_position_dynamics_negative(1)
    # dynamixel_driver.max_min_position_dynamics_negative(0)

    'TEST 4: Measure the behaviour of the joints in torque control'
    dynamixel_driver.model_calibration_fixed_torque(0)
    # dynamixel_driver.model_calibration_fixed_torque_revert(0)

    'TEST 5: Current-based Position control mode'
    # dynamixel_driver.xm_current_posControl(0)
    # print("testing")
    # dynamixel_driver.max_min_position_dynamics_negative_current_pos(0)

    'TEST 6: Switch control mode'
    # dynamixel_driver.switch_control_mode()

    'TEST 7: Bulk read'
    # dynamixel_driver.bulk_read()

    'TEST 8: With Object'
    # dynamixel_driver.max_min_position_dynamics_with_obj(0)

    'TEST 9: Move Randomly'
    ' USE THIS TRAJECTORY FOR JOINT CALIBRATION'
    # dynamixel_driver.move_random(0)

    'Back to init position'
    print("--------------------------------------------------------")
    dynamixel_driver.xm_posControl(0, dynamixel_driver.MIN_POS[0])
    dynamixel_driver.xm_posControl(1, dynamixel_driver.MIN_POS[1])
    # dynamixel_driver.xl_posControl(0, 0)
    dynamixel_driver.xl_posControl(1, 0)
    time.sleep(2)

    dynamixel_driver.close_port()