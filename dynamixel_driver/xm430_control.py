import time

from dynamixel_sdk import *
import dynamixel_sdk as dxl
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def load_lists_from_csv(file_path):
    loaded_data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            loaded_data.append(list(map(eval, row)))  # 将每个元素转换回原来的数据类型
    return loaded_data[0], loaded_data[1], loaded_data[2]


class Dynamixel_Driver:
    def __init__(self):
        self.port_num = '/dev/tty.usbserial-FT5WIZLL'
        self.DXL_ID_aray = [1, 4, 3, 6]
        self.BAUDRATE = 57600

        # self.port_num = '/dev/tty.usbserial-FT5NUSO1'
        # self.DXL_ID_aray = [11]
        # self.BAUDRATE = 115200

        self.PROTOCOL_VERSION = 2.0
        self.portHandler = dxl.PortHandler(self.port_num)
        self.packetHandler = dxl.PacketHandler(self.PROTOCOL_VERSION)
        self.DRIVE_MODE_TIME = 4
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.SAVEFILEDIR = 'calib.scv'
        self.XM = {
            'ADDR_PRO_OPERATING_MODE': 11,
            'ADDR_DRIVE_MODE': 10,
            'ADDR_MOVING_STATUS': 123,
            'ADDR_PRO_TORQUE_ENABLE': 64,
            'ADDR_PRO_PRESENT_POSITION': 132,
            'ADDR_MAX_POS_LIM': 48,
            'ADDR_MIN_POS_LIM': 52,
            'ADDR_CURRENT_LIM': 38,
            'ADDR_VELOCITY_LIM': 44,
            'ADDR_PRESENT_CURRENT': 126,
            'ADDR_PRESENT_VELOCITY': 128,
            'ADDR_PROFILE_VELOCITY': 112,
            'ADDR_PROFILE_ACCELERATION': 108,
            'ADDR_PRO_GOAL_POSITION': 116,
            'ADDR_GOAL_CURRENT': 102,
            'ADDR_VELOCITY_TRAJECTORY': 136,
            'ADDR_POSITION_TRAJECTORY': 140,
            'ADDR_GOAL_VELOCITY': 104,
        }

        self.XL = {
            'ADDR_CONTROL_MODE': 11,
            'ADDR_PRO_TORQUE_ENABLE': 24,
            'ADDR_PRO_GOAL_POSITION': 30,
            'ADDR_PRO_SPEED': 32,
            'ADDR_PRESENT_SPEED': 100
        }

        self.MAX_POS = [1445, 2651]  # right, left [127.002, 232.998]
        self.MIN_POS = [2671, 1425]  # [234.756, 125.244]

        # Set port baudrate
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            print("Failed to change the baudrate")

        # Open port
        if not self.portHandler.openPort():
            print("Failed to open the port")

        print("-----------------------")

        # Enable Dynamixel Torque
        for i in range(len(self.DXL_ID_aray)):
            if i == 0 or i == 1:
                address = self.XM['ADDR_PRO_TORQUE_ENABLE']
            else:
                address = self.XL['ADDR_PRO_TORQUE_ENABLE']
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[i], address, self.TORQUE_ENABLE)
            print("ID: %d is now checking" % self.DXL_ID_aray[i])
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        print("-----------------------")

        for ID in range(4):
            if ID == 0 or ID == 1:  # XM
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], 5)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
                print(f"ID: {self.DXL_ID_aray[ID]} is under current position control")
            else:  # XL
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XL['ADDR_CONTROL_MODE'], 2)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID],  self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
                print(f"ID: {self.DXL_ID_aray[ID]} is under position control")

        print("-----------------------")

        print("Moving to initial state")
        self.move_around()

        print("Init successfully")

        # if os.path.exists(self.SAVEFILEDIR):
        #     [self.DXL_MAXIMUM_POSITION_VALUE, self.DXL_MINIMUM_POSITION_VALUE, self.ORIENTATION] = load_lists_from_csv(
        #         self.SAVEFILEDIR)
        # else:
        #     input("No limit file found, press any key to calibrate")
        #     self.caliberation()

    def move_around(self):
        self.xm_current_posControl(0, self.MIN_POS[0], 0)
        time.sleep(2)
        self.xm_current_posControl(0, self.MAX_POS[0], 0)
        time.sleep(2)
        self.xm_current_posControl(0, self.MIN_POS[0], 0)
        time.sleep(2)
        self.xm_current_posControl(1, self.MIN_POS[1], 0)
        time.sleep(2)
        self.xm_current_posControl(1, self.MAX_POS[1], 0)
        time.sleep(2)
        self.xm_current_posControl(1, self.MIN_POS[1], 0)
        time.sleep(2)
        self.xl_posControl(0, 0)
        time.sleep(1)
        self.xl_posControl(0, 1)
        time.sleep(1)
        self.xl_posControl(1, 0)
        time.sleep(1)
        self.xl_posControl(1, 1)
        time.sleep(1)

    def xm_posControl(self, ID, goal_pos, reset=True, profile_vel=50, acceleration_time=5):
        POS_CONTROL = 3
        DRIVE_MODE_VELOCITY = 4

        if self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'])[0] != POS_CONTROL or reset == True:
            print(f"Switching ID: {self.DXL_ID_aray[ID]} to position control")
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], POS_CONTROL)
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MAX_POS_LIM'], self.MAX_POS[ID])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MIN_POS_LIM'], self.MIN_POS[ID])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_DRIVE_MODE'], DRIVE_MODE_VELOCITY)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_VELOCITY'], profile_vel)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_ACCELERATION'], acceleration_time)

            print(f"ID: {self.DXL_ID_aray[ID]} is under position control")

        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_GOAL_POSITION'], goal_pos)
        return self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'])[0]

    def xl_posControl(self, servo, friction, movingSpeed=300):
        if friction == 0:  # 90 degree low friction
            if servo == 0:  # left
                ID = self.DXL_ID_aray[2]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 214)
            else:  # right
                ID = self.DXL_ID_aray[3]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 513)
        else:  # High friction
            assert friction == 1, f"friction should be either 0 (Low) or 1 (High): {friction}"
            if servo == 0:
                ID = self.DXL_ID_aray[2]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 520)
            else:
                ID = self.DXL_ID_aray[3]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 190)

    def xm_current_posControl(self, ID, goal_pos, control_mode):
        """
        1. The Profile Velocity(112), Profile Acceleration(108) : Reset to ‘0’
        2. The Goal PWM(100) and Goal Current(102): reset to the value of PWM Limit(36) and Current Limit(38) respectively
        3. Position PID(80, 82, 84) and PWM Limit(36) values will be reset
        """
        if control_mode == 0:  # position control
            goal_current = 1193
            profile_vel = 50
            acceleration_time = 5
            self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MAX_POS_LIM'], self.MAX_POS[ID])
            self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MIN_POS_LIM'], self.MIN_POS[ID])
        else:  # torque control
            assert control_mode == 1, f"control mode should be either position control or current control: {control_mode}"
            goal_current = 20
            profile_vel = 200
            acceleration_time = 100
            goal_pos = self.MIN_POS[ID] - 100

        DRIVE_MODE_VELOCITY = 4

        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_DRIVE_MODE'], DRIVE_MODE_VELOCITY)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_VELOCITY'], profile_vel)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_ACCELERATION'], acceleration_time)

        self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_GOAL_CURRENT'], goal_current)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_GOAL_POSITION'], goal_pos)

        return self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'])[0]

    def torque_control(self, ID, goal_torque=None, reset=True):

        TORQUE_CONTROL_MODE = 0  # Torque Control mode value

        # Check ID for setting goal_torque
        if goal_torque is None:
            if self.DXL_ID_aray[ID] == 1:
                goal_torque = 0xFFEC  # -20
                # goal_torque = 0x0020
            elif self.DXL_ID_aray[ID] == 4:
                goal_torque = 0x0020

        if self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE']) != TORQUE_CONTROL_MODE or reset == True:
            print(f"Switching ID: {self.DXL_ID_aray[ID]} to torque control")
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], self.TORQUE_DISABLE)
            self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], TORQUE_CONTROL_MODE)
            # self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_CURRENT_LIM'], self.XM['MAX_CURRENT'])
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], self.TORQUE_ENABLE)
            print(f"ID: {self.DXL_ID_aray[ID]} is under torque control")

        self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_GOAL_CURRENT'], goal_torque)

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

    def time_taken_for_control_read(self):
        'Test: time taken for control signal'
        start_control = time.time()
        for _ in range(100):
            self.xm_posControl(0, 2048)  # left finger, -90 to left, 1023=90
        print("Time taken for sending a control signal: ", time.time() - start_control, (time.time() - start_control) / 100)

        start_read = time.time()
        for _ in range(100):
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
        print("Time taken for reading a position: ", time.time() - start_read, (time.time() - start_read) / 100)

    def max_min_position_dynamics(self, ID):
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
        dynamixel_driver.xm_posControl(0, self.MIN_POS[ID], profile_vel=50, acceleration_time=5, reset=True)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_time)
            current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PROFILE_VELOCITY'])[0]
            goal_position.append(self.MIN_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(current_vel)
            vel_limit_buf.append(vel_limit)
            if abs(current_pos - self.MIN_POS[ID]) < 2:
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
        axs[0].plot(t, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--',  linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position', linewidth=1)
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
        # fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

    def max_min_position_dynamics_negative(self, ID):
        '''
        The adjusted parameters are listed below:
        * profile_vel = 50
        * acceleration_time = 5
        '''
        self.xm_posControl(0, self.MIN_POS[ID])
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_t = time.time()
        dynamixel_driver.xm_posControl(0, self.MAX_POS[ID], profile_vel=50, acceleration_time=5, reset=True)
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
        df.to_csv(data_filename, index=False)

        'Plot data'
        fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
        # Plot for position on the first subplot
        axs[0].plot(t, actual_position, label='Actual Position', color='r')
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position', linewidth=1)
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

        # plt.show()

    def model_calibration_sinusoidal(self, ID):
        time_duration = 5  # Total time of simulation in seconds
        sample_rate = 80  # How many samples per second
        min_freq = 0.1  # Minimum frequency in Hz
        max_freq = 1  # Maximum frequency in Hz

        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = 0.5
        sine_wave = np.array([np.sin(2 * np.pi * freq * t[i]) for i in range(len(t))])
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

        # plt.show()

    def model_calibration_fixed_torque(self, ID):
        self.xm_posControl(ID, self.MIN_POS[0])
        time.sleep(3)
        t = []
        actual_position = []
        actual_vel = []
        start_t = time.time()
        vel_limit_buf = []
        self.torque_control(ID)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_t)
            unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                    self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                         self.XM['ADDR_PROFILE_VELOCITY'])[0]
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(vel_limit)

            print(current_pos - self.MAX_POS[0])
            if abs(current_pos - self.MAX_POS[0]) < 50:
                self.torque_control(ID, 0)
                break

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
        axs[1].plot(t, vel_limit_buf, label='Velocity Limit', color='b', linestyle='--', linewidth=1)
        axs[1].set_title('XM430 Velocity Over Time with Torque = -20')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        plt.show()
        # image_filename = f'/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.png'
        # fig.savefig(image_filename)

    def model_calibration_fixed_torque_revert(self, ID):
        self.xm_posControl(ID, self.MAX_POS[0])
        time.sleep(3)
        t = []
        actual_position = []
        actual_vel = []
        start_t = time.time()
        vel_limit_buf = []
        self.torque_control(ID)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
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
                self.torque_control(ID, 0)
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
        dynamixel_driver.xm_current_posControl(0, self.MAX_POS[ID], 0)
        time.sleep(3)
        goal_position = []
        actual_position = []
        actual_vel = []
        vel_limit_buf = []
        t = []
        start_t = time.time()
        dynamixel_driver.xm_current_posControl(0, self.MIN_POS[ID], 1)
        while True:
            current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
            t.append(time.time() - start_t)
            unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PRESENT_VELOCITY'])[0]
            vel_limit = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                           self.XM['ADDR_PROFILE_VELOCITY'])[0]
            signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
            goal_position.append(self.MIN_POS[ID])
            actual_position.append(current_pos)
            actual_vel.append(signed_current_vel)
            vel_limit_buf.append(-vel_limit)
            if abs(current_pos - self.MIN_POS[ID]) < 2:
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
        axs[0].hlines(y=self.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='Goal Position (Control Signal)',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position', linewidth=1)
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
        # fig.savefig(image_filename)

        print(f"Data saved to {data_filename}")
        print(f"Image saved to {image_filename}")

        plt.show()

# goal_pos = 2048
# dynamixel_driver.xm_posControl(0, 3071) # right finger, +90 to right, 3071=270
# dynamixel_driver.xm_posControl(0, 2048, total_move_time, acceleration_time)  # left finger, -90 to left, 1023=90

dynamixel_driver = Dynamixel_Driver()

'TEST 1: Measure the time-taken for passing control signal or read signal'
# dynamixel_driver.time_taken_for_control_read()

'TEST 2: Measures the dynamic response of the finger to sinusoidal signla'
# dynamixel_driver.model_calibration_sinusoidal(0)
# dynamixel_driver.model_calibration_sinusoidal_vary_f(0)

'TEST 3: Measures the behaviour of the joints near their limits'
# dynamixel_driver.max_min_position_dynamics(0)
# dynamixel_driver.max_min_position_dynamics_negative(0)

'TEST 4: Measure the behaviour of the joints in torque control'
# dynamixel_driver.model_calibration_fixed_torque(0)
# dynamixel_driver.model_calibration_fixed_torque_revert(0)

'TEST 5: Current-based Position control mode'
# dynamixel_driver.xm_current_posControl(0)
# dynamixel_driver.max_min_position_dynamics_negative_current_pos(0)