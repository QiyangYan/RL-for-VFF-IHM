"""
Left finger: 0
Right finger: 1
self.DXL_ID_array = [left, left friction, right, right friction]
"""

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
        self.DXL_ID_aray = [4, 3, 1, 6]
        # self.DXL_ID_aray = [4, 1]
        # self.DXL_ID_aray = [1, 4]
        self.BAUDRATE = 1000000

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
            'ADDR_BAUD_RATE': 8,
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
            'ADDR_PRO_PRESENT_POSITION': 37,
            'ADDR_PRO_TORQUE_ENABLE': 24,
            'ADDR_PRO_GOAL_POSITION': 30,
            'ADDR_PRO_SPEED': 32,
            'ADDR_PRESENT_VELOCITY': 39,
            'ADDR_BAUD_RATE': 4
        }

        # self.MAX_POS = [1445, 2651]  # right, left [127.002, 232.998]
        # self.MIN_POS = [2671, 1425]  # [234.756, 125.244]

        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate!")
        else:
            print("Failed to change the baudrate")

        # Open port
        if not self.portHandler.openPort():
            print("Failed to open the port")

        print("| Enable Dynamixel Torque")

        # Enable Dynamixel Torque
        for i in range(len(self.DXL_ID_aray)):
            if i == 0 or i == 2:
                address = self.XM['ADDR_PRO_TORQUE_ENABLE']
            else:
                address = self.XL['ADDR_PRO_TORQUE_ENABLE']

            # if self.DXL_ID_aray[i] == 3:
            #     continue
            # elif self.DXL_ID_aray[i] == 6:
            #     address = self.XL['ADDR_PRO_TORQUE_ENABLE']

            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, self.DXL_ID_aray[i], address, self.TORQUE_ENABLE)
            print("ID: %d is now checking" % self.DXL_ID_aray[i])
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception("Communication failed: %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                raise Exception("Dynamixel error: %s" % self.packetHandler.getRxPacketError(dxl_error))

        print("-----------------------")
        print("| Enable Position Control Mode")

        self.groupBulkWrite = dxl.GroupBulkWrite(self.portHandler, self.packetHandler)
        self.groupBulkRead = dxl.GroupBulkRead(self.portHandler, self.packetHandler)

        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 2:  # XM
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_BAUD_RATE'], 3)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
                # self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], 5)
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], 2)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
                print(f"ID: {self.DXL_ID_aray[ID]} is under position control")
            else:  # XL
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XL['ADDR_BAUD_RATE'], 3)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XL['ADDR_CONTROL_MODE'], 2)
                self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID],  self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
                print(f"ID: {self.DXL_ID_aray[ID]} is under position control")

        ' ----------------------- '
        ' | Bulk read init '
        # self.xm_start_address = self.XM["ADDR_PRESENT_VELOCITY"]
        # self.xl_start_address = self.XL["ADDR_PRO_PRESENT_POSITION"]
        # self.xm_data_len = 8  # byte
        # self.xl_data_len = 4
        # self.bulk_init()

        # print("-----------------------")
        # print("| Moving to initial state")
        # self.move_around()

        print("-----------------------")
        print("| Init successfully")

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

    def xm_posControl(self, finger_index, goal_pos, reset=True, profile_vel=50, acceleration_time=5):
        """

        :param finger_index: left_finger = 0, right_finger = 1
        :param goal_pos: goal position
        :param reset: True if the finger is currently under torque control
        :param profile_vel: DON'T CHANGE
        :param acceleration_time: DON'T CHANGE

        :return: None
        """
        POS_CONTROL = 3
        DRIVE_MODE_VELOCITY = 4

        'ID[0] is left, ID[2] is right'
        ID = finger_index * 2

        if reset == True:
            # print(f"Switching ID: {self.DXL_ID_aray[ID]} to position control")
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 0)
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], POS_CONTROL)
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], 1)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MAX_POS_LIM'], self.MAX_POS[finger_index])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MIN_POS_LIM'], self.MIN_POS[finger_index])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_DRIVE_MODE'], DRIVE_MODE_VELOCITY)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_VELOCITY'], profile_vel)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_ACCELERATION'], acceleration_time)
            # print(f"ID: {self.DXL_ID_aray[ID]} is under position control")

        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_GOAL_POSITION'], goal_pos)
        # hex_data = format(goal_pos, '08X')

        # data_bytes = [(goal_pos >> (i * 8)) & 0xFF for i in range(4)]
        # self.groupBulkWrite.addParam(self.DXL_ID_aray[ID], self.XL['ADDR_PRO_GOAL_POSITION'], 4, data_bytes)

    def xl_posControl(self, servo, friction, movingSpeed=300):
        if friction == 0:  # 90 degree low friction
            if servo == 0:  # left`low
                ID = self.DXL_ID_aray[1]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 204)
            else:  # right low
                ID = self.DXL_ID_aray[3]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 820)
        else:  # High friction
            assert friction == 1, f"friction should be either 0 (Low) or 1 (High): {friction}"
            if servo == 0:  # left high
                ID = self.DXL_ID_aray[1]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 512)
            else:  # right high
                ID = self.DXL_ID_aray[3]
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_SPEED'], movingSpeed)
                self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.XL['ADDR_PRO_GOAL_POSITION'], 512)

    def xm_current_posControl(self, ID, goal_pos, control_mode):
        """
        1. The Profile Velocity(112), Profile Acceleration(108) : Reset to ‘0’
        2. The Goal PWM(100) and Goal Current(102): reset to the value of PWM Limit(36) and Current Limit(38) respectively
        3. Position PID(80, 82, 84) and PWM Limit(36) values will be reset
        """
        if control_mode == 0:  # position control
            goal_current = 1193
            profile_vel = 1000
            acceleration_time = 1
            self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MAX_POS_LIM'], self.MAX_POS[ID])
            self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_MIN_POS_LIM'], self.MIN_POS[ID])
        else:  # torque control
            assert control_mode == 1, f"control mode should be either position control or current control: {control_mode}"
            goal_current = 20
            profile_vel = 0
            acceleration_time = 0
            goal_pos = self.MIN_POS[ID] - 100

        DRIVE_MODE_VELOCITY = 0

        # self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_DRIVE_MODE'], DRIVE_MODE_VELOCITY)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_VELOCITY'], profile_vel)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PROFILE_ACCELERATION'], acceleration_time)

        self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_GOAL_CURRENT'], goal_current)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_GOAL_POSITION'], goal_pos)

        return self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_PRESENT_POSITION'])[0]

    def xm_torque_control(self, ID, goal_torque=None, reset=True):

        TORQUE_CONTROL_MODE = 0  # Torque Control mode value
        ID *= 2

        # Check ID for setting goal_torque
        if goal_torque is None:
            if self.DXL_ID_aray[ID] == 1:
                goal_torque = 0xFFEC  # -20
                # goal_torque = 0x0020
            elif self.DXL_ID_aray[ID] == 4:
                goal_torque = 0x0020

        if reset == True:
            # print(f"Switching ID: {self.DXL_ID_aray[ID]} to torque control")
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], self.TORQUE_DISABLE)
            self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_OPERATING_MODE'], TORQUE_CONTROL_MODE)
            # self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_CURRENT_LIM'], self.XM['MAX_CURRENT'])
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_PRO_TORQUE_ENABLE'], self.TORQUE_ENABLE)
            # print(f"ID: {self.DXL_ID_aray[ID]} is under torque control")

        # print(goal_torque)
        self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID_aray[ID], self.XM['ADDR_GOAL_CURRENT'], goal_torque)

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

    def close_port(self):
        for i in range(len(self.DXL_ID_aray)):
            if i == 0 or i == 2:
                address = self.XM['ADDR_PRO_TORQUE_ENABLE']
            else:
                address = self.XL['ADDR_PRO_TORQUE_ENABLE']
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID_aray[i], address, self.TORQUE_DISABLE)
            print("ID: %d is now checking" % self.DXL_ID_aray[i])
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception("Communication failed: %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                raise Exception("Dynamixel error: %s" % self.packetHandler.getRxPacketError(dxl_error))

        self.portHandler.closePort()


if __name__ == '__main__':
    # goal_pos = 2048
    # dynamixel_driver.xm_posControl(0, 3071) # right finger, +90 to right, 3071=270
    # dynamixel_driver.xm_posControl(0, 2048, total_move_time, acceleration_time)  # left finger, -90 to left, 1023=90

    dynamixel_driver = Dynamixel_Driver()

    'Move to init position'
    print("| Start moving to Initial Position")
    dynamixel_driver.xm_posControl(0, dynamixel_driver.MIN_POS[0])  # left finger
    dynamixel_driver.xm_posControl(1, dynamixel_driver.MIN_POS[1])  # right finger
    dynamixel_driver.xl_posControl(0, 0)  # left finger
    dynamixel_driver.xl_posControl(1, 0)  # right finger
    time.sleep(1)
    dynamixel_driver.xl_posControl(0, 1)  # left finger
    dynamixel_driver.xl_posControl(1, 1)  # right finger
    time.sleep(1)
    dynamixel_driver.xl_posControl(0, 0)  # left finger
    dynamixel_driver.xl_posControl(1, 0)  # right finger
    time.sleep(2)
    print("| Moved to Initial Position")

    # dynamixel_driver.bulk_read()