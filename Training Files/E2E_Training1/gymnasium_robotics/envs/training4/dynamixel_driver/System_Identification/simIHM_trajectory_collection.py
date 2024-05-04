import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from common.common import COMMON
from Friction_Change.friction_change import FRICTION
from dynamixel_driver.angle_conversion import AngleConversion


class SimulationIHMWithObject:
    def __init__(self, env_):
        self.env = env_
        self.env.reset()
        self.friction = FRICTION()
        self.common = COMMON(self.env)
        self.inAir = False
        self.AngleConvert = AngleConversion()
        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]

    def adjust_parameter_right_object(self, friction):
        pass

    def adjust_parameter_left_object(self, friction):
        pass

    def slide_trajectory(self, env):
        env.reset()
        action = np.zeros(2)
        """
        This is pick up + slide. So it starts moving from the middle.
        :return:
        """
        # pick up
        env_dict, _ = self.common.pick_up(self.inAir)

        # slide
        control_mode = -1
        last_f = 0
        friction_state = 1
        pos_index = 0

        # change to goal friction
        # print(f"| Change friction stage 1 - to Goal Friction, control mode: {control_mode} ")
        if friction_state != last_f:
            friction_start_time = time.time()
            next_env_dict, reward_dict, terminated, _, infos = self.friction.friction_change_to_low(friction_state,
                                                                                                    env)
            friction_end_time = time.time()
            friction_change_time = friction_end_time - friction_start_time
            if terminated is True:
                print("terminated during friction changing2")
                return terminated
        else:
            print("same friction state, no change")

        # slide up on right finger
        # print(next_env_dict["observation"][pos_index * 2])
        action[0] = 0  # relative-pos
        action[1] = control_mode  # control mode

        observation_list = []
        # pos_list_dyn = [245, 200, 245, 200, 245]
        # pos_list_dyn = [10, 30, 10, 20, 50, 100, 150, 245, 150, 50]
        pos_list_dyn = [1, 10, 100, 10, 10, 10, 200, 245, 10, 40, 50, 140, 245, 245, 10, 10, 10]
        # pos_list_dyn = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # pos_list_dyn = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        start = 1.05
        current_pos = start
        start_dyn = self.AngleConvert.rad_2_xm(1.05)
        min = self.AngleConvert.xm_2_rad(self.MIN_POS[0])

        sim_time = []
        sim_control = []
        sim_pos = []
        sim_vel = []
        sim_pos_torque = []
        sim_vel_torque = []

        i = 0
        sim_time.append(i * env.unwrapped.dt)
        sim_control.append(np.clip(self.AngleConvert.rad_2_xm(start - 0/1226 + min), self.MIN_POS[0], self.MAX_POS[0]))
        sim_pos.append(self.AngleConvert.rad_2_xm(next_env_dict["observation"][0] + min))
        sim_vel.append(self.AngleConvert.xm_rad_per_sec_to_rpm(next_env_dict["observation"][4]))
        sim_pos_torque.append(self.AngleConvert.rad_2_xm(next_env_dict["observation"][2] + min))
        sim_vel_torque.append(self.AngleConvert.xm_rad_per_sec_to_rpm(next_env_dict["observation"][6]))

        print("Start Slide")
        time.sleep(1)

        start_start = current_pos
        for pos in pos_list_dyn:
            start = current_pos
            action[0] = self.AngleConvert.xm_2_policy(pos)  # dynamixel to policy action range conversion
            # time.sleep(1)r
            for _ in range(11):
                i += 1
                next_env_dict, reward_dict, terminated, _, infos = env.step(action)
                observation_list.append(next_env_dict["observation"])
                current_pos = next_env_dict["observation"][0]
                sim_time.append(i * env.unwrapped.dt)
                sim_control.append(np.clip(self.AngleConvert.rad_2_xm((start - self.AngleConvert.xm_2_rad(pos)) + min), self.MIN_POS[0], self.MAX_POS[0]))
                # sim_control.append(np.clip(self.AngleConvert.rad_2_xm((start_start - self.AngleConvert.xm_2_rad(pos)) + min),
                #                            self.MIN_POS[0], self.MAX_POS[0]))

        sim_pos += [self.AngleConvert.rad_2_xm(array[0] + min) for array in observation_list]
        sim_vel += [self.AngleConvert.xm_rad_per_sec_to_rpm(array[4]) for array in observation_list]
        sim_pos_torque += [self.AngleConvert.rad_2_xm(- array[2] + self.AngleConvert.xm_2_rad(self.MIN_POS[1])) for array in observation_list]
        sim_vel_torque += [self.AngleConvert.xm_rad_per_sec_to_rpm(array[6]) for array in observation_list]

        return sim_time, sim_pos, sim_vel, sim_control, sim_pos_torque, sim_vel_torque

    def plt_combined_results_2x2(self, sim_time, sim_pos, sim_vel, sim_control, sim_pos_torque, sim_vel_torque):

        # real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_20240307_204451.csv'
        # real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Trajectory With Object/Model_Dynamics_20240306_215759.csv'
        # real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_20240310_191655.csv'
        real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_20240310_222211.csv'
        df = pd.read_csv(real_record_path)
        real_position = np.array(df['Actual Position'])
        real_time = np.array(df['t'])
        real_velocity = np.array(df['Actual Velocity'])
        real_control = np.array(df['Goal Position'])

        # Create a 2x2 grid for plotting
        fig, axs = plt.subplots(2, 2, figsize=(8, 5))  # 2 rows, 2 columns

        # Plot for regular position on the first subplot (top left)
        # axs[0, 0].plot(self.real_time, self.real_pos, label='Real Position', color='r')
        axs[0, 0].hlines(y=self.MAX_POS[0], xmin=0, xmax=sim_time[-1], colors='b', linestyles='--', label='max_position',
                  linewidth=1)
        axs[0, 0].hlines(y=self.MIN_POS[0], xmin=0, xmax=sim_time[-1], colors='b', linestyles='--',
                         label='max_position',
                         linewidth=1)
        axs[0, 0].plot(real_time, real_position, label="Real Position", color='r')
        axs[0, 0].plot(real_time, real_control, label="Goal Position", color='b', linewidth=1, linestyle='--')
        axs[0, 0].plot(sim_time, sim_pos, label='Simulated Position', color='g')
        axs[0, 0].plot(sim_time, sim_control, label='Sim Control', color='k', linewidth=1, linestyle='--')
        axs[0, 0].set_title('XM430 Angular Position Over Time')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Position')
        axs[0, 0].legend(fontsize=6)

        # Plot for regular velocity on the second subplot (top right)
        # axs[0, 1].plot(self.real_time, self.real_vel, label='Real Velocity', color='k')
        axs[0, 1].plot(real_time, real_velocity, label='Real Velocity', color='k')
        axs[0, 1].plot(sim_time, sim_vel, label='Simulated Velocity', color='b')
        axs[0, 1].set_title('XM430 Angular Velocity Over Time')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Velocity')
        axs[0, 1].legend(fontsize=6)

        # Plot for sine position on the third subplot (bottom left)
        # axs[1, 0].plot(self.real_time_sin, self.real_pos_sin, label='Real Position', color='r')
        axs[1, 0].hlines(y=self.MAX_POS[1], xmin=0, xmax=sim_time[-1], colors='b', linestyles='--',
                         label='max_position',
                         linewidth=1)
        axs[1, 0].hlines(y=self.MIN_POS[1], xmin=0, xmax=sim_time[-1], colors='b', linestyles='--',
                         label='max_position',
                         linewidth=1)
        axs[1, 0].plot(sim_time, sim_pos_torque, label='Simulated Position', color='g')
        axs[1, 0].set_title('Torque XM430 Angluar Position Over Time')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Position')
        axs[1, 0].legend(fontsize=6)

        # Plot for sine velocity on the fourth subplot (bottom right)
        axs[1, 1].plot(sim_time, sim_vel_torque, label='Simulated Velocity', color='b')
        axs[1, 1].set_title('Torque XM430 Angular Velocity Over Time')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity')
        axs[1, 1].legend(fontsize=6)

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = gym.make("VariableFriction-v2", render_mode="human")
    # env = gym.make("VariableFriction-v2")

    sim = SimulationIHMWithObject(env)
    sim_time, sim_pos, sim_vel, sim_control, sim_pos_torque, sim_vel_torque = sim.slide_trajectory(env)
    sim.plt_combined_results_2x2(sim_time, sim_pos, sim_vel, sim_control, sim_pos_torque, sim_vel_torque)
    # t, pos, _ = sim.step_response(0)
    # plt.plot(t, pos)
    # plt.show()