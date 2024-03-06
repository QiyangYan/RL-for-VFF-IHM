import numpy as np
import gymnasium as gym
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dynamixel_driver.angle_conversion import AngleConversion
from scipy.interpolate import interp1d
from dynamixel_driver.dynamixel_control import Dynamixel_Driver

AngleConvert = AngleConversion()

class MuJoCo_Simulation_executor():
    def __init__(self, render):
        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]
        if render is True:
            self.env = gym.make("VariableFriction-v3", render_mode="human")
        else:
            assert render is False, f"render is not bool: {render}"
            self.env = gym.make("VariableFriction-v3")
        self.render = render
        self.env.reset()
        time_duration = 3  # Total time of simulation in seconds
        sample_rate = 110  # How many samples per second
        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = 0.5
        self.sine_wave = np.array([-np.cos(2 * np.pi * freq * t[i]) for i in range(len(t))])
        self.control_sine = (self.sine_wave + 1) * (1.8807 / 2)

    # def step_response(self,real_time):
    #     T = round(round(real_time/self.env.unwrapped.dt)+5)
    #     obs,info = self.env.reset()
    #     action = [0, 0, 0, 0]
    #     history = [AngleConvert.rad_2_xm(obs["observation"][18])]
    #     t=[self.env.unwrapped.n_steps*self.env.unwrapped.dt]
    #     # for i in range(5):
    #     #     obs, _, _, _, _ = self.env.step(action)
    #
    #     for i in range(T-1):
    #         # print(i)
    #         if i == 1:
    #             action = [0, 0, 0, 1]
    #         obs, _, _, _, _ = self.env.step(action)
    #         robot_joint_pos_calib = obs["observation"][18:24]
    #
    #         current_pos = robot_joint_pos_calib[0]
    #         t.append(self.env.unwrapped.n_steps*self.env.unwrapped.dt)
    #         history.append(AngleConvert.rad_2_xm(current_pos))
    #         # print(AngleConvert.rad_2_xm(current_pos), current_pos)
    #     return t, history

    def step_response(self, real_time, ID=1):
        T = round(round(real_time/self.env.unwrapped.dt)+5)
        obs, info = self.env.reset()
        action = 1.8807
        start_pos_right = AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        # t = [self.env.unwrapped.n_steps * self.env.unwrapped.dt]
        t = []
        vel_history = []
        for i in range(T):
            obs, _, _, _, _ = self.env.step(action)
            current_pos = obs["observation"][ID*2]
            vel_history.append(AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID*2+4]*(ID*2-1)))
            t.append((i+1)*self.env.unwrapped.dt)
            history.append(AngleConvert.rad_2_xm(start_pos_right - current_pos*(ID*2-1)))

        return t, history, vel_history

    def sin_response(self, real_time, ID=1):
        # T = round(round(real_time / self.env.dt) + 5)
        obs, info = self.env.reset()
        action = 1.8807
        start_pos = AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        vel_history = []
        t = []
        control = []

        start_t = time.time()
        for i, pos in enumerate(self.control_sine):
            t_1 = (i + 1) * self.env.unwrapped.dt * 11
            # t_1 = (i + 1) * self.env.unwrapped.dt
            if t_1 > real_time:
                break
            for _ in range(11):
                obs, _, _, _, _ = self.env.step(pos)
            current_pos = obs["observation"][ID*2]
            t.append(t_1)
            control.append(AngleConvert.rad_2_xm(start_pos - pos*(ID*2-1)))
            history.append(AngleConvert.rad_2_xm(start_pos - current_pos*(ID*2-1)))
            vel_history.append(AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID*2+4]*(ID*2-1)))
        # frequency = i / t[-1]
        # print(frequency)

        return t, history, vel_history, control

    def manual_policy(self, real_time, ID=1):
        # T = round(round(real_time / self.env.dt) + 5)
        obs, info = self.env.reset()
        start_pos = AngleConvert.xm_2_rad(self.MAX_POS[ID])
        min = AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        vel_history = []
        t = []
        control = []

        while True:
            obs, _, _, _, _ = self.env.step(1.8807)
            current_pos = obs["observation"][ID * 2]
            if abs(current_pos - 1.8807) > 0.003:
                # print("Ready to receive commend")
                break

        for _ in range(100):
            obs, _, _, _, _ = self.env.step(1.8807)

        goal_pos_list_dyn = [1, 10, 100, 10, 10, 10, 200, 245, 10, 40, 50, 140, 245, 245, 10, 10, 10]
        goal_pos_list = []
        for pos in goal_pos_list_dyn:
            goal_pos_list.append(AngleConvert.xm_2_rad(pos))

        current_pos = start_pos
        step = 0
        for i, pos in enumerate(goal_pos_list):
            # t_1 = (i + 1) * self.env.unwrapped.dt
            goal_pos = np.clip(current_pos + pos * (ID * 2 - 1), 0, 1.8807)
            for _ in range(11):
                step += 1
                t1 = (step + 1) * self.env.unwrapped.dt
                if t1 > real_time:
                    break
                obs, _, _, _, _ = self.env.step(goal_pos)
                current_pos = obs["observation"][ID * 2]
                t.append(t1)
                control.append(AngleConvert.rad_2_xm(min + goal_pos))
                history.append(AngleConvert.rad_2_xm(min + current_pos))
                vel_history.append(AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID * 2 + 4] * (ID * 2 - 1)))
            if t1 > real_time:
                break

        return t, history, vel_history, control

    def torque(self, real_time, ID=1):
        T = round(round(real_time / self.env.unwrapped.dt))
        _, _ = self.env.reset()
        action = 1
        start_pos_right = AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        t = []
        vel_history = []
        for i in range(T):
            obs, _, _, _, _ = self.env.step(action)
            current_pos = obs["observation"][ID * 2]
            vel_history.append(AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID * 2 + 4] * (ID * 2 - 1)))
            t.append((i + 1) * self.env.unwrapped.dt)
            history.append(AngleConvert.rad_2_xm(start_pos_right - current_pos * (ID * 2 - 1)))

        return t, history, vel_history

    def adjust_parameter_right(self, damping=1.084, armature=0.045, frictionloss=0.03, gainprm=21.1, biastype=1, forcerange=1.3, gear=1, torque=False):
        '''
        Actuator
            :param forcerange: -1.3 ~ 1.3
            :param kp:

        Joint
            :param damping:
            :param armature: increase
            :param frictionloss:
        '''
        self.env.unwrapped.model.dof_damping[3] = damping
        self.env.unwrapped.model.dof_armature[3] = armature
        self.env.unwrapped.model.dof_frictionloss[3] = frictionloss

        # self.env.unwrapped.model.actuator_gear[1][0] = gear
        if not torque:
            self.env.unwrapped.model.actuator_gainprm[1][0] = gainprm
            self.env.unwrapped.model.actuator_biasprm[1][1] = -gainprm

        # self.env.unwrapped.model.actuator_biastype[1] = biastype
        # self.env.unwrapped.model.actuator_forcerange = [-forcerange, forcerange]

    def adjust_parameter_left(self, damping=1.084, armature=0.045, frictionloss=0.03, gainprm=21.1, biastype=1, gear=1, torque=False):
        '''
        Actuator
            :param forcerange: -1.3 ~ 1.3
            :param kp:

        Joint
            :param damping:
            :param armature: increase
            :param frictionloss:
        '''

        self.env.unwrapped.model.dof_damping[1] = damping
        self.env.unwrapped.model.dof_armature[1] = armature
        self.env.unwrapped.model.dof_frictionloss[1] = frictionloss

        # self.env.unwrapped.model.actuator_gear[0][0] = gear
        if not torque:
            self.env.unwrapped.model.actuator_gainprm[0][0] = gainprm
            self.env.unwrapped.model.actuator_biasprm[0][1] = -gainprm

        # self.env.unwrapped.model.actuator_biastype[1] = biastype
        # self.env.unwrapped.model.actuator_forcerange = [-forcerange, forcerange]

    def check_parameter(self):
        print("Damping: ", self.env.unwrapped.model.dof_damping)
        print("Armature: ", self.env.unwrapped.model.dof_armature)
        print("Frictionless: ", self.env.unwrapped.model.dof_frictionloss)
        print("Gear: ", self.env.unwrapped.model.actuator_gear)
        print("Gainprm: ", self.env.unwrapped.model.actuator_gainprm)
        print("Biasprm: ", self.env.unwrapped.model.actuator_biasprm)
        print("Force Range: ", self.env.unwrapped.model.actuator_forcerange)


if __name__ == "__main__":
    sim = MuJoCo_Simulation_executor(True)
    t, actual_position, actual_vel, goal_position = sim.manual_policy(100, 0)
    # t, actual_position, actual_vel = sim.torque(2, 0)
    ID = 0

    'Save data'
    df = pd.DataFrame({
        't': t,
        # 'Goal Position': goal_position,
        'Actual Position': actual_position,
        'Actual Velocity': actual_vel,
        # 'Velocity Limit': vel_limit_buf
    })
    # print(goal_position)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_{timestamp}.csv"
    # df.to_csv(data_filename, index=False)

    'Plot data'
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
    # Plot for position on the first subplot
    axs[0].plot(t, actual_position, label='Actual Position', color='r')
    # axs[0].plot(t, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)
    axs[0].hlines(y=sim.MAX_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='max_position',
                  linewidth=1)
    axs[0].hlines(y=sim.MIN_POS[ID], xmin=0, xmax=t[-1], colors='b', linestyles='--', label='min_position',
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
    # fig.savefig(image_filename)

    print(f"Data saved to {data_filename}")
    print(f"Image saved to {image_filename}")

    plt.show()