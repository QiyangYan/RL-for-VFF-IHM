import numpy as np

class AngleConversion():
    def __init__(self):
        pass

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
        return velocity_rad_per_sec * (60 / (2 * np.pi)) / 0.229
