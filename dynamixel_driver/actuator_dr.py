from System_Identification.Sim_executor import MuJoCo_Simulation_executor
from System_Identification.simIHM_trajectory_collection import SimulationIHMWithObject
from DomainRandomisation.randomisation import RandomisationModule
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym


class ActuatorDR(MuJoCo_Simulation_executor, SimulationIHMWithObject):
    def __init__(self, env_):
        MuJoCo_Simulation_executor.__init__(self, env_=env_)
        SimulationIHMWithObject.__init__(self, env_=env_)
        self.Randomise = RandomisationModule()

    def randomise_parameter(self):
        self.env.unwrapped.model.dof_damping[1] = self.Randomise.uniform_randomise("damping")
        self.env.unwrapped.model.dof_armature[1] = self.Randomise.uniform_randomise("armature")
        self.env.unwrapped.model.actuator_gainprm[0][0] = self.Randomise.uniform_randomise("kp")
        self.env.unwrapped.model.actuator_biasprm[0][1] = -self.Randomise.uniform_randomise("kp")
        pass

    def adjust_parameter_left_(self, damping, armature, frictionloss, gainprm, biastype=1, gear=1, torque=False):
        """

        :param damping:
        :param armature:
        :param frictionloss:
        :param gainprm:
        :param biastype:
        :param gear:
        :param torque:
        :return:
        """

        self.env.unwrapped.model.dof_damping[1] = damping
        self.env.unwrapped.model.dof_armature[1] = armature
        self.env.unwrapped.model.dof_frictionloss[1] = frictionloss

        # self.env.unwrapped.model.actuator_gear[0][0] = gear
        if not torque:
            self.env.unwrapped.model.actuator_gainprm[0][0] = gainprm
            self.env.unwrapped.model.actuator_biasprm[0][1] = -gainprm

        # self.env.unwrapped.model.actuator_biastype[1] = biastype
        # self.env.unwrapped.model.actuator_forcerange = [-forcerange, forcerange]

    def actuator_parameter_dr(self, duration, ID_, real_data, num_sim):
        t_list = []
        pos_list = []
        vel_list = []
        goal_list = []
        for i in range(num_sim):
            self.adjust_parameter_left_(damping=4.99979599, armature=0.39951026, gainprm=12.87708353, frictionloss=0.03)
            # self.check_parameter_left()
            if i != 0:
                self.randomise_parameter()
            # t_, sim_present_position_, sim_present_vel_, sim_goal_position_ = self.manual_policy(duration, ID_)
            t_, sim_present_position_, sim_present_vel_, sim_goal_position_, _, _ = self.slide_trajectory(self.env)
            t_list.append(t_)
            pos_list.append(sim_present_position_)
            vel_list.append(sim_present_vel_)
            goal_list.append(sim_goal_position_)

        # Plot all trajectories
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        for i in range(num_sim):
            if i == 0:
                axs[0].plot(t_list[i], pos_list[i], color='lightsteelblue', linewidth=1, label="Sim Position")
                axs[0].plot(t_list[i], goal_list[i], color='moccasin', linestyle='--', linewidth=1, label="Sim Control")
                axs[1].plot(t_list[i], vel_list[i], color='lightsteelblue', linewidth=1, label="Sim Velocity")
            else:
                axs[0].plot(t_list[i], pos_list[i], color='lightsteelblue', linewidth=1)
                axs[0].plot(t_list[i], goal_list[i], color='moccasin', linestyle='--', linewidth=1)
                axs[1].plot(t_list[i], vel_list[i], color='lightsteelblue', linewidth=1)

        axs[0].plot(t_list[0], pos_list[0], color='r', label="Sim Position with Default Parameters", linewidth=1)
        axs[0].plot(np.array(real_data['t']), np.array(real_data['Actual Position']), label="Real Position")
        axs[0].plot(np.array(real_data['t']), np.array(real_data['Goal Position']), label="Real Control", linestyle='--', linewidth=1)
        axs[0].hlines(y=self.MAX_POS[ID_], xmin=0, xmax=duration, colors='b', linestyles='--', label='max_position',
                      linewidth=1)
        axs[0].hlines(y=self.MIN_POS[ID_], xmin=0, xmax=duration, colors='b', linestyles='--', label='min_position',
                      linewidth=1)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Position')
        axs[0].set_title('Actuator Trajectories')
        axs[0].legend(fontsize='small')  # Omit or modify if you don't need a legend for each trajectory

        axs[1].plot(t_list[0], vel_list[0], color='r', label="Sim Velocity with Default Parameters", linewidth=1)
        axs[1].plot(np.array(real_data['t']), np.array(real_data['Actual Velocity']), label="Real Velocity")
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Velocity')
        axs[1].set_title('Actuator Trajectories')
        axs[1].legend(fontsize='small')  # Omit or modify if you don't need a legend for each trajectory

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration IHM-like Trajectory/Model_Dynamics_20240305_181840.csv'
    real_record_path = '/Users/qiyangyan/Desktop/FYP/Sim2Real/Model_Dynamics_20240307_204451.csv'
    df = pd.read_csv(real_record_path)
    real_position = np.array(df['Actual Position'])
    real_time = np.array(df['t'])
    real_velocity = np.array(df['Actual Velocity'])

    # env = gym.make("VariableFriction-v2", render_mode="human")
    env = gym.make("VariableFriction-v2")  # slide_trajectory
    # env = gym.make("VariableFriction-v3")  # manual_policy

    ID_ = 0
    sim_DR = ActuatorDR(env)
    sim_DR.actuator_parameter_dr(real_time[-1], ID_, df, num_sim=100)
