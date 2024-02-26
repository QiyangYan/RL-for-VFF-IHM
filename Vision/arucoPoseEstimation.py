"""
Todo: apply filter
"""""

import numpy as np
import cv2
import sys
import time
import pickle
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import pickle
from collections import deque
from filters import FILTER


class SMOOTH():
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = np.zeros((window_size, 3))  # Assuming 3D vectors for translation or rotation
        self.index = 0
        # self.kf = self.create_kalman_filter()
        self.kf = self.create_pose_kalman_filter()
        # self.kf = self.create_translation_kalman_filter()

        window_name = "Frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.resizeWindow(window_name, 640, 480)  # Set the window size

    def update(self, new_value):
        self.values[self.index % self.window_size] = new_value
        self.index += 1
        return np.mean(self.values, axis=0)

    def create_kalman_filter(self):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.])  # Initial state (position and velocity)
        kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
        kf.H = np.array([[1., 0.]])  # Measurement function
        kf.P *= 1000.  # Covariance matrix
        kf.R = 5  # Measurement noise
        kf.Q = np.array([[1., 0.], [0., 1.]])  # Process noise
        return kf

    def create_pose_kalman_filter(self, dt=1):
        # 6D state [x, y, z, pitch, roll, yaw]
        kf = KalmanFilter(dim_x=6, dim_z=6)
        kf.x = np.zeros(6)  # Initial state (position and orientation)
        kf.F = np.eye(6) + np.eye(6, k=3) * dt  # State transition matrix (assuming constant velocity model for simplicity)
        kf.H = np.eye(6)  # Measurement function
        kf.P *= 1000.  # Initial covariance matrix, large uncertainty
        kf.R = np.eye(6) * 5  # Adjust this based on your measurement noise, Measurement noise
        kf.Q = np.eye(6) * 0.1  # Process noise, adjust based on the expected amount of noise in your system
        return kf

    def create_translation_kalman_filter(self, dt=1):
        kf = KalmanFilter(dim_x=3, dim_z=3)
        kf.x = np.zeros(3)
        kf.F = np.eye(3) + np.eye(3, k=3) * dt
        kf.H = np.eye(3)
        kf.P *= 1000.
        kf.R = np.eye(3) * 5
        kf.Q = np.eye(3) * 0.1
        return kf

    # Update the filter with new measurements
    def update_kalman_filter(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x  # Estimated position

    def apply_kalman_realtime(self, filtered_data, process_noise=1e-5, measurement_noise=1e-2, estimation_error=1):
        kalman_filtered = []
        x_est = filtered_data[0]
        error_var = estimation_error

        for measurement in filtered_data:
            error_var += process_noise
            kalman_gain = error_var / (error_var + measurement_noise)
            x_est = x_est + kalman_gain * (measurement - x_est)
            error_var = (1 - kalman_gain) * error_var
            kalman_filtered.append(x_est)

        return kalman_filtered


class RealTimeFilter:
    def __init__(self, median_window_size=5, process_noise=1e-5, measurement_noise=1e-2, estimation_error=1):
        self.median_window = deque(maxlen=median_window_size)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimation_error = estimation_error
        self.x_est = None  # Will be initialized on the first call
        self.error_var = estimation_error

    def apply_median_filter(self, new_data_point):
        self.median_window.append(new_data_point)
        return np.median(np.array(self.median_window))

    def apply_kalman_filter(self, median_filtered_data):
        if self.x_est is None:
            self.x_est = median_filtered_data  # Initialize on first call

        # Prediction update
        self.error_var += self.process_noise

        # Measurement update
        kalman_gain = self.error_var / (self.error_var + self.measurement_noise)
        self.x_est = self.x_est + kalman_gain * (median_filtered_data - self.x_est)
        self.error_var = (1 - kalman_gain) * self.error_var

        return self.x_est

    def filter_data(self, new_data_point):
        median_result = self.apply_median_filter(new_data_point)
        kalman_result = self.apply_kalman_filter(median_result)
        return kalman_result

class ARUCO():
    def __init__(self, aruco_type="DICT_4X4_100"):
        # with open("CameraCalibration/cameraMatrix.pkl", "rb") as f:
        #     self.intrinsic_camera = pickle.load(f)
        # with open("CameraCalibration/dist.pkl", "rb") as f:
        #     self.distortion = pickle.load(f)

        with open("CameraCalibration/camera_matrix.npy", "rb") as f:
            self.intrinsic_camera = np.load(f)
        # Load the distortion coefficients
        with open("CameraCalibration/dist_coeffs.npy", "rb") as f:
            self.distortion = np.load(f)

        self.aruco_type = aruco_type
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        self.smooth = SMOOTH(2)
        self.cap = cv2.VideoCapture(1)

    def aruco_display(self, corners, ids, rejected, image):
        if len(corners) > 0:

            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))

        return image

    def pose_estimation(self, frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        # detector = cv2.aruco.ArucoDetector(cv2.aruco_dict, parameters)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict)

        tvec_list_str = ""  # Initialize an empty string to compile tvec info
        vertical_offset = 50  # Starting vertical offset for drawing text

        pose_list = []

        if ids is not None and len(corners) > 0:
            for i, corner in enumerate(corners):
                if ids[i] == 2:
                    continue
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.0335, matrix_coefficients,
                                                                    distortion_coefficients)

                pose = np.concatenate((tvec[0][0], rvec[0][0]))

                pose_list.append(pose)

                # Generating tvec string for the current marker
                tvec_str = 'ID {}: x={:.2f}, y={:.2f}, z={:.2f}'.format(ids[i][0], tvec[0][0][0], tvec[0][0][1],
                                                                        tvec[0][0][2])

                # Optionally, add this to a cumulative list string
                tvec_list_str += tvec_str + "\n"
                # Display the tvec info near the marker
                cv2.putText(frame, tvec_str, (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                vertical_offset += 50  # Increment the vertical offset for the next text line

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

        return frame, pose_list

    def get_pose(self):
        # Initialize the camera capture

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Error: Failed to open the camera.")
            return

        # Variables for FPS calculation
        frames_to_skip = 30  # Number of initial frames to skip for stabilization
        frame_count = 0
        poses = []
        filter_poses = []
        smooth_poses = []
        start_time = time.time()
        median_window_size = 11
        median_window = deque(maxlen=median_window_size)
        filter = FILTER()

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Skip initial frames for stabilization
                if frame_count < frames_to_skip:
                    frame_count += 1
                    continue

                if frame_count > 500:
                    break

                output, pose = self.pose_estimation(frame, self.ARUCO_DICT[self.aruco_type], self.intrinsic_camera, self.distortion)
                if not pose:
                    poses.append(poses[-1])
                else:
                    poses.append(pose[0])

                # Append new pose to the window for median filtering
                # median_window.append(pose)
                # if len(median_window) == median_window_size:
                #     # Apply median filter
                #     median_filtered_pose = np.median(np.array(median_window), axis=0)
                # else:
                #     median_filtered_pose = median_window[-1]  # Use the latest value if window is not full
                filter_poses.append(filter.apply_filter(poses[-1]))
                # print(filter.apply_filter(poses[-1]))

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                # Display FPS
                # print(f"FPS: {fps:.2f}")
                # height, width, _ = frame.shape
                # print(f"RGB Resolution: {width}x{height}")

                # Display the frame
                cv2.imshow("Frame", frame)

                # Check for the 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Release the camera
            self.cap.release()
            cv2.destroyAllWindows()


        time_axis = np.arange(len(poses))
        labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']

        # print(np.array(poses))
        pickle_file_path = 'poses_data.pkl'
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(poses, file)

        file_path = 'filter_poses_data.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(filter_poses, file)

        print(f"Poses data saved to {pickle_file_path}")
        print(f"Filtered poses data saved to {file_path}")

        plt.figure(figsize=(10, 7))
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(time_axis, np.array(poses)[:, i], label='Pose')
            plt.plot(time_axis, np.array(filter_poses)[:, i], label='Smooth Pose', linestyle='--')
            plt.title(labels[i])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
        plt.show()

    def test_filter(self):
        pickle_file_path = 'poses_data.pkl'
        with open(pickle_file_path, 'rb') as file:
            poses = pickle.load(file)

        time_axis = np.arange(len(poses))
        labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
        plt.figure(figsize=(12, 8))
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(time_axis, np.array(poses)[:, i], label='Pose')
            # plt.plot(time_axis, np.array(smooth_poses)[:, i], label='Smooth Pose', linestyle='--')
            plt.title(labels[i])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
        plt.show()


aruco_pose = ARUCO()
aruco_pose.get_pose()
# aruco_pose.test_filter()