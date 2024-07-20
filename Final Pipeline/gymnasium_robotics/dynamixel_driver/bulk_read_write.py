from dynamixel_sdk import *
import numpy as np
from gymnasium_robotics.dynamixel_driver.dynamixel_control import Dynamixel_Driver


class BULK(Dynamixel_Driver):
    def __init__(self):
        super().__init__()
        self.xm_start_address = self.XM["ADDR_PRESENT_VELOCITY"] # the starting address for reading
        self.xl_start_address = self.XL["ADDR_PRO_PRESENT_POSITION"]
        self.xm_bulk_len_full = 8  # byte, 4+4, position byte + velocity byte
        self.xl_bulk_len_full = 4
        self.bulk_init()

        # use name: "Present Position". "Present Velocity"
        # add more term in the following four list if needed
        self.xm_bulk_read_addr = {
            "Present Position": self.XM["ADDR_PRO_PRESENT_POSITION"],
            "Present Velocity": self.XM["ADDR_PRESENT_VELOCITY"]
        }
        self.xl_bulk_read_addr = {
            "Present Position": self.XL["ADDR_PRO_PRESENT_POSITION"],
            "Present Velocity": self.XL["ADDR_PRESENT_VELOCITY"]
        }
        self.xm_bulk_len_list = {
            "Present Position": 4,  # byte, check in the e-manual
            "Present Velocity": 4
        }
        self.xl_bulk_len_list = {
            "Present Position": 2,  # byte
            "Present Velocity": 2
        }

    def bulk_init(self):
        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 2:
                result = self.groupBulkRead.addParam(self.DXL_ID_aray[ID], self.xm_start_address, self.xm_bulk_len_full)
                if not result:
                    raise Exception(f"Failed to add parameter for Dynamixel ID: {self.DXL_ID_aray[ID]}")
            else:
                # Add parameter storage for Dynamixel#1 present position value
                result = self.groupBulkRead.addParam(self.DXL_ID_aray[ID], self.xl_start_address, self.xl_bulk_len_full)
                if not result:
                    raise Exception(f"Failed to add parameter for Dynamixel ID: {self.DXL_ID_aray[ID]}")

    def get_obs_dynamixel(self):
        """
        This function read through the parameters specified in the _init_, and save the 'Present Position' and 'Present
        Velocity' into the buffer that are later concatenated as the observation [joint position, joint velocity].

        Add parameters: If more parameters are needed, add them into the list in init and add extra buffer in this
        function if they need to be in the observation space.

        Todo:
        1. Align the structure with the observation in simulation:
            [0] — left motor
            [1] — left friction motor
            [2] — right motor
            [3] — right friction motor

        :return: [xm_R_pos, xm_L_pos, xl_R_pos, xl_L_pos, xm_R_vel, xm_L_vel, xl_R_vel, xl_L_vel]
        """
        dxl_comm_result = self.groupBulkRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            Exception("%s" % self.groupBulkRead.txRxPacket())

        observation_pos = []
        observation_vel = []

        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 2:  # XM
                if self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.xm_start_address, self.xm_bulk_len_full):
                    for param, addr in self.xm_bulk_read_addr.items():
                        info = self.groupBulkRead.getData(self.DXL_ID_aray[ID], addr, self.xm_bulk_len_list[param])
                        if param == "Present Position":
                            observation_pos.append(info)
                        elif param == "Present Velocity":
                            signed_current_vel = np.array(info, dtype=np.uint32).astype(np.int32)
                            observation_vel.append(signed_current_vel)
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")
                pass

            else:  # XL
                if self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.xl_start_address, self.xl_bulk_len_full):
                    for param, addr in self.xl_bulk_read_addr.items():
                        info = self.groupBulkRead.getData(self.DXL_ID_aray[ID], addr, self.xl_bulk_len_list[param])
                        if param == "Present Position":
                            observation_pos.append(info)
                        elif param == "Present Velocity":
                            signed_current_vel = np.array(info, dtype=np.uint32).astype(np.int32)
                            observation_vel.append(signed_current_vel)
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")

        observation = np.concatenate([np.array(observation_pos), np.array(observation_vel)])
        # print(observation)
        return observation

    def bulk_read_origin(self):
        dxl_comm_result = self.groupBulkRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise Exception("%s" % self.groupBulkRead.txRxPacket())

        observation_pos = []
        observation_vel = []

        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 2:  # XM
                if self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.xm_start_address, self.xm_bulk_len_full):
                    dxl_present_position = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XM['ADDR_PRO_PRESENT_POSITION'], 4)
                    dxl_present_velocity = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XM['ADDR_PRESENT_VELOCITY'], 4)
                    observation_pos.append(dxl_present_position)
                    observation_vel.append(dxl_present_velocity)
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")

            else:  # XL
                if self.groupBulkRead.isAvailable(self.DXL_ID_aray[ID], self.xl_start_address, self.xl_bulk_len_full):
                    dxl_present_position = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XL['ADDR_PRO_PRESENT_POSITION'], 2)
                    dxl_present_velocity = self.groupBulkRead.getData(self.DXL_ID_aray[ID],
                                                                      self.XL['ADDR_PRESENT_VELOCITY'], 2)
                    observation_pos.append(dxl_present_position)
                    observation_vel.append(dxl_present_velocity)
                else:
                    print(f"Failed to read data from Dynamixel ID: {self.DXL_ID_aray[ID]}")

        observation = np.concatenate([np.array(observation_pos), np.array(observation_vel)])
        return observation
        # print(observation)


class BulkPerformanceTest(BULK):
    def __init__(self):
        super().__init__()

    def regular_read(self):
        """
        This is used for: reading speed comparison with bulk read.

        This read method uses regular self.packetHandler.read4ByteTxRx function to read multiple times for multiple
        parameters in each actuator, which would takes longer time and result in slow data reading.

        :return: NONE
        """
        observation_pos = []
        observation_vel = []

        for ID in range(len(self.DXL_ID_aray)):
            if ID == 0 or ID == 2:  # XM
                current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID],
                                                               self.XM['ADDR_PRO_PRESENT_POSITION'])[0]
                unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                        self.XM['ADDR_PRESENT_VELOCITY'])[0]
                signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
                observation_pos.append(current_pos)
                observation_vel.append(signed_current_vel)

            else:  # XL
                current_pos = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[ID],
                                                               self.XL['ADDR_PRO_PRESENT_POSITION'])[0]
                unsigned_current_vel = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID_aray[0],
                                                                        self.XL['ADDR_PRESENT_VELOCITY'])[0]
                signed_current_vel = np.array(unsigned_current_vel, dtype=np.uint32).astype(np.int32)
                observation_pos.append(current_pos)
                observation_vel.append(signed_current_vel)

        observation = np.concatenate([np.array(observation_pos), np.array(observation_vel)])
        return observation
        # print(observation)

    def read_and_write_speed_analysis(self):
        """
        This is: Regular Read and Bulk Read Speed Comparison
        It measures the average time taken for two reading methods for 20 cycles and find the average time taken for
        each reading, which is then used for reading frequency calculation

        Reading frequency: 7.812789768483247 Hz
        Bulk Reading frequency: 31.252637728156035 Hz
        Writing frequency: 31.253324709610666 Hz
        Writing with reset frequency: 6.25032588676032 Hz

        :return: NONE
        """
        print("-----------------------")
        print("| Regular Read and Bulk Read Speed Comparison")
        print("Measuring..")
        start = time.time()
        for _ in range(20):
            self.regular_read()
        print(f"Reading frequency: {1 / ((time.time() - start) / 20)} Hz")

        print("Measuring..")
        start = time.time()
        for _ in range(20):
            self.get_obs_dynamixel()
        print(f"Bulk Reading frequency: {1 / ((time.time() - start) / 20)} Hz")

        print("Measuring..")
        start = time.time()
        for _ in range(20):
            self.xm_posControl(0, self.MIN_POS[0], reset=False)
        print(f"Writing frequency: {1 / ((time.time() - start) / 20)} Hz")

        print("Measuring..")
        start = time.time()
        for _ in range(20):
            self.xm_posControl(0, self.MIN_POS[0], reset=True)
        print(f"Writing with reset frequency: {1 / ((time.time() - start) / 20)} Hz")


if __name__ == "__main__":
    # print("-----------------------")
    # print("| Bulk Init ")
    # bulk = BULK()
    # print(bulk.get_obs_dynamixel())

    'Bulk Experiment'
    print("-----------------------")
    print("| Bulk Experiment ")
    bulk_test = BulkPerformanceTest()

    'TEST 1: Regular Read, Bulk Read, Write Speed Analysis'
    bulk_test.read_and_write_speed_analysis()
