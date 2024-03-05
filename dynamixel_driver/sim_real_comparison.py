import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

from common.common import COMMON
from Friction_Change.friction_change import FRICTION
from angle_conversion import AngleConversion

class SimulationTrajectory():
    def __init__(self):
        env_name = "VariableFriction-v2"
        self.env = gym.make(env_name, render_mode="human")
        # self.env = gym.make(env_name)
        self.env.reset()
        self.friction = FRICTION()
        self.common = COMMON(self.env)
        self.inAir = False
        self.action = np.zeros(3)
        self.AngleConvert = AngleConversion()
        self.MAX_POS = [2651, 1445]  # left, right [127.002, 232.998]
        self.MIN_POS = [1425, 2671]  # [234.756, 125.244]
        time_duration = 10  # Total time of simulation in seconds
        sample_rate = 80  # How many samples per second
        t = np.linspace(0, time_duration, int(time_duration * sample_rate))
        freq = 0.5
        self.sine_wave = np.array([-np.cos(2 * np.pi * freq * t[i]) for i in range(len(t))])
        self.control_sine = (self.sine_wave + 1) * (1.8807 / 2)

    def step_response(self, ID):
        # prms = [0, 2.32669025,4.41389059, 7.95477761,4.10019415]
        prms = [ 0.9041239 ,  0.42817789, 17.66408197 , 0.12312853]
        # self.adjust_parameter_left(damping=prms[0], armature=prms[1], gainprm=prms[2], forcerange=prms[3], frictionloss=prms[4])
        self.adjust_parameter_left(damping=prms[0], armature=prms[1], gainprm=prms[2], frictionloss=prms[3])
        action = np.array([1.05, -2, False])
        start_pos_right = self.AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        # t = [self.env.unwrapped.n_steps * self.env.unwrapped.dt]
        t = []
        vel_history = []
        i = 0
        while True:
            i += 1
            obs, _, _, _, _ = self.env.step(action)
            current_pos = obs["observation"][ID * 2]
            vel_history.append(self.AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID * 2 + 4] * (ID * 2 - 1)))
            t.append((i + 1) * self.env.unwrapped.dt)
            history.append(self.AngleConvert.rad_2_xm(start_pos_right - current_pos * (ID * 2 - 1)))
            print(current_pos)
            if abs(current_pos - action[0]) < 0.05:
                break
        return t, history, vel_history


    def sin_response(self, ID=0):
        # T = round(round(real_time / self.env.dt) + 5)
        start_pos = self.AngleConvert.xm_2_rad(self.MIN_POS[ID])
        history = []
        vel_history = []
        t = []
        control = []
        start_t = time.time()
        for i, pos in enumerate(self.control_sine):
            t_1 = (i + 1) * self.env.unwrapped.dt * 2
            # t_1 = (i + 1) * self.env.unwrapped.dt
            obs, _, _, _, _ = self.env.step(pos)
            obs, _, _, _, _ = self.env.step(pos)
            current_pos = obs["observation"][ID*2]
            t.append(t_1)
            control.append(self.AngleConvert.rad_2_xm(start_pos - pos*(ID*2-1)))
            history.append(self.AngleConvert.rad_2_xm(start_pos - current_pos*(ID*2-1)))
            vel_history.append(self.AngleConvert.xm_rad_per_sec_to_rpm(-obs["observation"][ID*2+4]*(ID*2-1)))
        # frequency = i / t[-1]
        # print(frequency)

        return t, history, vel_history, control

    def slide_trajectory(self):
        self.adjust_parameter_left(damping=0, armature=2.32669025, gainprm=4.41389059, frictionloss=4.10019415)
        # pick up
        time.sleep(2)
        env_dict, _ = self.common.pick_up(self.inAir)
        control_mode = 0
        last_f = 0
        friction_state = 1
        pos_index = 0
        time.sleep(2)

        # change to goal friction
        print(f"| Change friction stage 1 - to Goal Friction, control mode: {control_mode} ")
        if friction_state != last_f:
            friction_start_time = time.time()
            next_env_dict, reward_dict, terminated, _, infos = self.friction.friction_change_to_low(friction_state,
                                                                                                    self.env)
            friction_end_time = time.time()
            friction_change_time = friction_end_time - friction_start_time
            if terminated is True:
                print("terminated during friction changing2")
                return terminated
        else:
            print("same friction state, no change")

        # slide up on right finger
        print(env_dict["observation"][pos_index * 2])
        self.action[0] = env_dict["observation"][pos_index * 2]  # relative-pos
        self.action[1] = control_mode  # control mode
        self.action[2] = False  # not friction change
        observation_list = []
        while True:
            self.action[0] -= 0.01
            next_env_dict, reward_dict, terminated, _, infos = self.env.step(self.action)
            observation_list.append(next_env_dict["observation"])
            if next_env_dict["observation"][pos_index * 2] - 0 < 0.045:
                time.sleep(2)
                break

        pos_pos = [array[1] for array in observation_list]
        plt.plot(pos_pos)
        plt.show()

    def adjust_parameter_right(self, damping=1.084, armature=0.045, frictionloss=0.03, gainprm=21.1, biastype=1, forcerange=1.3, gear=1):
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

        self.env.unwrapped.model.actuator_gear[1][0] = gear
        self.env.unwrapped.model.actuator_gainprm[1][0] = gainprm
        self.env.unwrapped.model.actuator_biasprm[1][1] = -gainprm

        # self.env.unwrapped.model.actuator_biastype[1] = biastype
        # self.env.unwrapped.model.actuator_forcerange = [-forcerange, forcerange]

    def adjust_parameter_left(self, damping=1.084, armature=0.045, frictionloss=0.03, gainprm=21.1, biastype=1, forcerange=5, gear=1):
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

        self.env.unwrapped.model.actuator_gear[0][0] = gear
        self.env.unwrapped.model.actuator_gainprm[0][0] = gainprm
        self.env.unwrapped.model.actuator_biasprm[0][1] = -gainprm

        # self.env.unwrapped.model.actuator_biastype[1] = biastype
        # self.env.unwrapped.model.actuator_forcerange = [-forcerange, forcerange]

    def check_parameter(self):
        print("Damping: ", self.env.unwrapped.model.dof_damping)
        print("Armature: ", self.env.unwrapped.model.dof_armature)
        print("Frictionloss: ", self.env.unwrapped.model.dof_frictionloss)
        print("Gear: ", self.env.unwrapped.model.actuator_gear)
        print("Gainprm: ", self.env.unwrapped.model.actuator_gainprm)
        print("Biasprm: ", self.env.unwrapped.model.actuator_biasprm)
        print("Force Range: ", self.env.unwrapped.model.actuator_forcerange)




if __name__ == "__main__":
    sim = SimulationTrajectory()
    # sim.slide_trajectory()
    t, pos, _ = sim.step_response(0)
    plt.plot(t, pos)
    plt.show()