import time
import numpy as np
from typing import Tuple
from aruco_pose_estimation import ARUCO
import pickle
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


class CalibrationEvaluation(ARUCO):
    def __init__(self):
        super().__init__()

    def get_actual_pos(self) -> Tuple[int, int]:
        start = time.time()
        pose = np.zeros(2)
        self.get_pose(store=False, object_size=0.0355, display_fps=False, auto_terminate=True, record=False,
                      show_plot=True)

        if time.time() - start > 0.44:
            return pose  # TODO: implement this method

    def relative_error_measurement(self, save, from_video):
        """
        Collect a list of position
        :return: None
        """
        self.init_camera(video_path=from_video)
        print("Start placing the marker on the specified position")
        poses = []
        base_poses = []

        while True:
            if from_video is None:
                user_input = input(
                    "Press 'q' and ENTER to quit. Press Enter to continue reading the next position...").lower()
                if user_input == 'q':
                    break
                print("Reading")
                self.get_pose(store=False,
                              object_size=0.0335,
                              display_fps=False,
                              auto_terminate=True,
                              terminate_length=50,
                              display=False,
                              record=True,
                              show_plot=False,
                              fixed_pose=True,
                              release=False)

            else:
                done = self.get_pose_from_video(object_size=0.0335,
                                                display=False,
                                                show_plot=False)
                if done:
                    break

            base_poses.append([self.mean_std_list_base[0][0] * 100, self.mean_std_list_base[0][1] * 100, self.mean_std_list_base[1][0] * 100,
                          self.mean_std_list_base[1][1] * 100])
            poses.append([self.mean_std_list[0][0] * 100, self.mean_std_list[0][1] * 100, self.mean_std_list[1][0] * 100,
                          self.mean_std_list[1][1] * 100])
            print(f"| x: {self.mean_std_list[0][0] * 100} cm, std: {self.mean_std_list[0][1] * 100} cm")
            print(f"| y: {self.mean_std_list[1][0] * 100} cm, std: {self.mean_std_list[1][1] * 100} cm")

            self.reset_camera()
            self.filter.reset_filter()

        if save:
            file_path = f'calibration_eval_pose_data_{self.timestamp}.pkl'  # You can specify your own file path
            with open(file_path, 'wb') as file:
                pickle.dump(poses, file)

        return poses, base_poses

    def slide_vertical_evaluation(self):
        pass

    def get_origin_nearby_coord(self, poses):
        poses_array = np.array(poses)
        x = poses_array[:, 0]
        y = poses_array[:, 2]
        distances = np.sqrt(x ** 2 + y ** 2)
        closest_origin_index = np.argmin(distances)
        closest_coordinate = poses_array[closest_origin_index, [0, 2]]
        return closest_coordinate, distances

    def calculate_deviations(self, dots, grid_spacing):
        threshold = 1
        start, distance_from_origin = self.get_origin_nearby_coord(dots)
        expected_positions = [(start[0] + i * grid_spacing, start[1] + j * grid_spacing)
                              for i in range(-20, 20) for j in range(-20, 20)]

        # Filter the grid positions based on the threshold
        filtered_expected_positions = []
        for expected_pos in expected_positions:
            if any(np.linalg.norm(np.array([dot[0], dot[2]]) - np.array(expected_pos)) <= threshold for dot in dots):
                filtered_expected_positions.append(expected_pos)

        # Now calculate the deviations only for the filtered grid positions
        deviations = []
        for dot_4 in dots:
            dot = [dot_4[0], dot_4[2]]
            closest_expected = min(filtered_expected_positions,
                                   key=lambda x: np.linalg.norm(np.array(dot) - np.array(x)))
            deviation = np.linalg.norm(np.array(dot) - np.array(closest_expected))
            deviations.append(deviation)

        print(f"Calibration Evaluation - mean error: {np.mean(deviations)}, error std: {np.std(deviations)}")

        return deviations, np.array(filtered_expected_positions), distance_from_origin

    @staticmethod
    def plot_calib_eval_pose(poses, expected_positions, base_poses):
        x = np.array([p[0] for p in poses])
        y = np.array([p[2] for p in poses])

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, marker='o', label='Path')
        plt.scatter(expected_positions[:, 0], expected_positions[:, 1], marker='o', color='r')
        plt.scatter(base_poses[:, 0], base_poses[:, 1], marker='o', color='r')
        plt.xlabel('X Coordinate (cm)')
        plt.ylabel('Y Coordinate (cm)')
        plt.title('Path of the Marker')
        plt.axis('equal')  # This ensures the scale of x and y axes are equal
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_error_distribution(distance_from_origin, deviations):
        plt.figure(figsize=(6, 6))
        plt.scatter(distance_from_origin, deviations)
        plt.xlabel('Distance from origin')
        plt.ylabel('Error (cm)')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    calib_eval = CalibrationEvaluation()

    'Load Camera Parameters'
    # calib_eval.load_camera_params('/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/camera_matrix_20240314_013415.npy',
    #                               '/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/dist_coeffs_20240314_013415.npy')
    # Calibration Evaluation - mean error: 0.21506103185369616, error std: 0.12033852985362284

    'Default, random'
    calib_eval.load_camera_params()
    # Calibration Evaluation - mean error: 0.10861868473322646, error std: 0.055257341239734865

    'object height, less marker'
    # Calibration Evaluation - mean error: 0.10812287891031301, error std: 0.05428935214535761

    'object height, A4 - good'
    # Calibration Evaluation - mean error: 0.10756208010391076, error std: 0.054446713500309434

    'object height, more marker, smaller board'
    # Calibration Evaluation - mean error: 0.10817720527369559, error std: 0.0560499903370759

    # calib_eval.load_camera_params('/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/camera_matrix_20240314_181628.npy',
    #                               '/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/dist_coeffs_20240314_181628.npy')
    # Calibration Evaluation - mean error: 0.10861868473322646, error std: 0.055257341239734865

    # calib_eval.load_camera_params('/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/camera_matrix_20240318_210450.npy',
    #                               '/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/dist_coeffs_20240318_210450.npy')
    # Calibration Evaluation - mean error: 0.21506103185369616, error std: 0.12033852985362284


    'Get new pose'
    poses, base_poses = calib_eval.relative_error_measurement(save=False, from_video='/Users/qiyangyan/Desktop/FYP/Vision/output.mp4')

    'Get stored pose'
    # file_path = 'calibration_eval_pose_data.pkl'
    # with open(file_path, 'rb') as file:
    #     poses = pickle.load(file)

    'Plot'
    grid_spacing = 2
    deviations, expected_positions, distance_from_origin = calib_eval.calculate_deviations(poses, grid_spacing)
    calib_eval.plot_calib_eval_pose(poses, expected_positions, base_poses)
    calib_eval.plot_error_distribution(distance_from_origin, deviations)
