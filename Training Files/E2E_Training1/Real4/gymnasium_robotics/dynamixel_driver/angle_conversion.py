import numpy as np

class AngleConversion():
    def __init__(self, min_pos=[1425, 2671], max_pos=[2651, 1445]):
        self.MIN_POS = min_pos
        self.MAX_POS = max_pos
        pass

    def xm_2_sim(self, xm_joint_pos, ID):
        sim_joint_pos = self.xm_2_rad((self.MIN_POS[ID]-xm_joint_pos) / (ID*2-1))
        return sim_joint_pos

    def sim_2_xm(self, sim_joint_pos, ID):
        xm_joint_pos = int(self.MIN_POS[ID] - self.rad_2_xm(sim_joint_pos) * (ID * 2 - 1))
        return xm_joint_pos

    def xl_2_sim(self, xl_pos, ID):
        if ID == 1:
            if xl_pos > 800:  # low friction, control 820
                return 0
            elif xl_pos < 550:  # high friction, control 512
                return 0.0032
        else:
            assert ID == 0, f"Wrong ID, check {ID}"
            if xl_pos < 250:  # low friction, 204
                return 0
            elif xl_pos > 480:  # high friction, 512
                return 0.0032

    def sim_2_xl(self):
        return

    def sim_2_xm_vel(self, sim_vel):
        xm_vel = self.xm_rad_per_sec_to_rpm(sim_vel)
        return xm_vel

    @staticmethod
    def xm_2_sim_vel(xm_vel):
        sim_vel = xm_vel * 0.229 / (60 / (2 * np.pi))
        return sim_vel

    @staticmethod
    def sim_2_xl_vel(sim_vel):
        xl_vel = sim_vel * (60 / (2 * np.pi)) / 0.111
        return xl_vel

    @staticmethod
    def xl_2_sim_vel(xl_vel):
        sim_vel = xl_vel * 0.111 / (60 / (2 * np.pi))
        return sim_vel

    @staticmethod
    def rad_2_xm(angle):
        angle_deg = np.rad2deg(angle)
        return (angle_deg / 360) * 4096 % 4096

    @staticmethod
    def xm_2_rad(actuator_angle):
        angle = (actuator_angle) / 4096
        return np.deg2rad(angle * 360)

    @staticmethod
    def xm_rad_per_sec_to_rpm(velocity_rad_per_sec):
        """
        sim velocity to real velocity
        """
        return velocity_rad_per_sec * (60 / (2 * np.pi)) / 0.229

    @staticmethod
    def xm_2_policy(pos):
        policy_action = (pos / 1226) * 2 - 1
        return np.clip(policy_action, -1, 1)


if __name__ == "__main__":
    AngleConvert = AngleConversion()
    # print(AngleConvert.xm_2_sim(2048, 0))
    # print(AngleConvert.sim_2_xm(0.9557, 0))
    print(AngleConvert.xm_2_policy(2671-2029))