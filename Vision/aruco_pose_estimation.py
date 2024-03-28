import numpy as np
import cv2
import sys
import time
import pickle
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle
from collections import deque
from filters import FILTER
from datetime import datetime
from scipy.spatial.transform import Rotation as R


class ARUCO:
    def __init__(self, quat=False, aruco_type="DICT_4X4_100"):

        self.intrinsic_camera = None
        self.distortion = None

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
        self.filter = FILTER()
        self.filter_base = FILTER(base=True)
        self.filter_base_quat = FILTER(base=True, quat=True)
        self.filter_quat = FILTER(quat=True)
        self.poses = []
        self.poses_base = []
        self.filter_poses = []
        self.filter_poses_base = []
        self.cap = None
        self.frame = None
        self.loss_track_history = []
        self.loss_track_history_base = []
        self.mean_std_list = []
        self.mean_std_list_base = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.video_writer = None
        self.total_frames = None

        self.object_poses = []
        self.corner_poses = []

    def load_camera_params(self, matrix_path="/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/images_random_pose/camera_matrix.npy",
                           coef_path="/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/images_random_pose/dist_coeffs.npy"):
        self.intrinsic_camera = np.load(matrix_path)
        self.distortion = np.load(coef_path)

        # print(self.intrinsic_camera)
        # print(self.distortion)
        # self.intrinsic_camera =

        'object height, less marker'
        # self.intrinsic_camera = np.array([
        #     [1043.59243033918, 0, 1023.66782109748],
        #     [0, 1043.84004662865, 584.174328831662],
        #     [0, 0, 1]
        # ])
        #
        # self.distortion = np.array([
        #     [0.125011098753749,	-0.0907534056654978, 0, 0, 0]
        # ])

        'object height, A4'
        self.intrinsic_camera = np.array([
            [1030.92898850933 ,   0 ,   1023.77462992780],
            [0 ,   1029.90567981573  ,  591.144422554890],
            [0  ,  0 ,   1]
        ])

        self.distortion = np.array([
            [0.108371584297978,	-0.0797053729569890, 0, 0, 0]
        ])

        'object height, more marker, smaller board'
        # self.intrinsic_camera = np.array([
        #     [1057.45769026514, 0, 1010.45900505463],
        #     [0, 1057.01532134254, 581.702745015303],
        #     [0, 0, 1]
        # ])
        #
        # self.distortion = np.array([
        #     [0.146511573076407,	-0.120097629740936, 0, 0, 0]
        # ])

        'random place'
        # self.intrinsic_camera = np.array([
        #     [1036.97596542647   , 0   , 1010.34205932359],
        #     [0   , 1036.83968591485  ,  584.925387979896],
        #     [0  ,  0    ,1]
        # ])
        #
        # self.distortion = np.array([
        #     [0.148928059919677,	-0.124086193134545, 0, 0, 0]
        # ])

    def reset_camera(self):
        self.poses = []
        self.filter_poses = []
        self.loss_track_history = []
        self.mean_std_list = []

    def init_camera(self, video_path=None):
        if video_path is None:
            # Initialize the camera capture
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("Error: Failed to open the camera.")
        else:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not self.cap.isOpened():
                raise Exception("Error: Failed to open the camera.")

    def add_text_2_frame(self, pose, ids, i):
        tvec_list_str = ""  # Initialize an empty string to compile tvec info
        vertical_offset = 50  # Starting vertical offset for drawing text

        smooth = self.filter.apply_filter(pose)
        tvec_str = 'ID {}: x={:.4f}, y={:.4f}, z={:.4f}'.format(ids[i][0], smooth[0], smooth[1],
                                                                smooth[2])
        # print(tvec_str)

        # Optionally, add this to a cumulative list string
        tvec_list_str += tvec_str + "\n"
        # Display the tvec info near the marker
        cv2.putText(self.frame, tvec_str, (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),
                    2)
        vertical_offset += 50  # Increment the vertical offset for the next text line

    @staticmethod
    def convert_corner_2_world_coord(marker_length, rotation_matrix, tvec):
        half_side = marker_length / 2
        marker_points = np.array([
            [-half_side, half_side, 0],
            [half_side, half_side, 0],
            [half_side, -half_side, 0],
            [-half_side, -half_side, 0]
        ])
        world_coordinates = np.dot(rotation_matrix, marker_points.T).T + tvec
        return world_coordinates

    def pose_estimation(self, display, object_size, object_aruco_id=None):
        """
        return the pose (3+3) of the aruco marker once, this is assuming only single marker is considered
        store it into self.poses. This is the foundation for get_pose and get_pose from video.

        :param object_aruco_id:
        :param object_size:
        :param display: display aruco marker on frame or not
        :return:
        """
        aruco_dict_type = self.ARUCO_DICT[self.aruco_type]

        # Capture frame-by-frame
        ret, self.frame = self.cap.read()
        if not ret:
            return None, None, None

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict)

        pose_list = []
        corner_poses = []
        base_pose_list = []

        # vision_obs = {
        #     "base": None,
        #     "object_centre": None,
        #     "object_corners": None,
        #     "base_filter": None,
        #     "object_centre_filter": None,
        #     "object_corners_filter": None,
        # }

        if ids is not None and len(corners) > 0:
            for i, corner in enumerate(corners):
                if ids[i] == 2:  # the aruco marker on base
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, object_size, self.intrinsic_camera,
                                                                        self.distortion)
                    rotation = R.from_rotvec(rvec[0][0])
                    euler = rotation.as_euler('xyz', degrees=True)
                    pose = np.concatenate((tvec[0][0], euler))
                    base_pose_list.append(pose)

                    # vision_obs['base_filter'] = self.filter_base.apply_filter(base_pose_list[0])
                    # vision_obs['base'] = base_pose_list[0]

                    if display:
                        # self.add_text_2_frame(pose, ids, i)
                        cv2.aruco.drawDetectedMarkers(self.frame, corners)
                        cv2.drawFrameAxes(self.frame, self.intrinsic_camera, self.distortion, rvec, tvec, 0.01)

                elif ids[i] == object_aruco_id:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, object_size, self.intrinsic_camera,
                                                                        self.distortion)
                    rotation = R.from_rotvec(rvec[0][0])
                    euler = rotation.as_euler('xyz', degrees=True)
                    pose = np.concatenate((tvec[0][0], euler))
                    pose_list.append(pose)

                    # vision_obs['object_centre_filter'] = self.filter.apply_filter(pose_list[0])
                    # vision_obs['object_centre'] = pose_list[0]

                    if display:
                        # self.add_text_2_frame(pose, ids, i)
                        cv2.aruco.drawDetectedMarkers(self.frame, corners)
                        cv2.drawFrameAxes(self.frame, self.intrinsic_camera, self.distortion, rvec, tvec, 0.01)
                else:
                    pass

        pose_list = self.store_poses(pose_list)
        base_pose_list = self.store_poses_base(base_pose_list)
        self.filter_poses_base.append(self.filter_base.apply_filter(base_pose_list[0]))
        self.filter_poses.append(self.filter.apply_filter(pose_list[0]))

        return self.frame, pose_list[0], self.filter_poses[-1], base_pose_list[0], self.filter_poses_base[-1]
        # return self.frame, pose_list[0], self.filter_poses[-1]

    def pose_estimation_quat(self, display, object_size, object_aruco_id=None):
        """
        return the pose (3+4) of the aruco marker once, this is assuming only single marker is considered
        store it into self.poses. This is the foundation for get_pose and get_pose from video.

        FORMATE:
        [centre_pose
        corner_pose_0
        corner_pose_1
        corner_pose_2
        corner_pose_3
        ]

        For simplification, The angle part of the corner is the same as the centre

        :param object_aruco_id:
        :param object_size:
        :param display: display aruco marker on frame or not
        :return:
        """
        aruco_dict_type = self.ARUCO_DICT[self.aruco_type]

        # Capture frame-by-frame
        ret, self.frame = self.cap.read()
        if not ret:
            return None, None, None

        base_aruco_id = 2
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict)

        centre_pose = np.array([])
        base_pose = np.array([])

        if ids is not None and len(corners) > 0:
            for i, corner in enumerate(corners):
                if ids[i] == base_aruco_id:  # ignore the aruco marker on base
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, object_size, self.intrinsic_camera,
                                                                        self.distortion)
                    rotation = R.from_rotvec(rvec[0][0])
                    quaternion = rotation.as_quat()
                    base_pose = np.concatenate((tvec[0][0], quaternion))

                    if display:
                        # self.add_text_2_frame(pose, ids, i)
                        cv2.aruco.drawDetectedMarkers(self.frame, corners)
                        cv2.drawFrameAxes(self.frame, self.intrinsic_camera, self.distortion, rvec, tvec, 0.01)

                elif ids[i] == object_aruco_id:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, object_size, self.intrinsic_camera,
                                                                        self.distortion)
                    rotation = R.from_rotvec(rvec[0][0])
                    quaternion = rotation.as_quat()
                    centre_pose = np.concatenate((tvec[0][0], quaternion))

                    if display:
                        # self.add_text_2_frame(pose, ids, i)
                        cv2.aruco.drawDetectedMarkers(self.frame, corners)
                        cv2.drawFrameAxes(self.frame, self.intrinsic_camera, self.distortion, rvec, tvec, 0.01)

                else:  # ignore other aruco markers
                    pass
        # print(np.array(centre_pose))
        if centre_pose.any():
            object_pose = []
            centre_pose_filter = self.filter_quat.apply_filter(centre_pose)
            object_pose.append(centre_pose_filter)
            rotation_matrix = rotation.as_matrix()
            corner_pos = self.convert_corner_2_world_coord(object_size, rotation_matrix, centre_pose_filter[:3])
            for corner_ in corner_pos:
                object_pose.append(np.concatenate((corner_, quaternion)))

            self.object_poses.append(np.array(object_pose))  # filtered centre + corners
            self.poses.append(centre_pose)  # no filter
            self.loss_track_history.append(0)
        else:
            if self.poses:
                self.object_poses.append(self.object_poses[-1])  # filtered pose
                self.poses.append(self.poses[-1])  # no filter
                self.loss_track_history.append(1)
            else:
                self.object_poses.append(np.zeros(7))  # filtered pose
                self.poses.append(np.zeros(7))  # no filter

        if base_pose.any():
            base_pose_filter = self.filter_base_quat.apply_filter(base_pose)
            self.filter_poses_base.append(base_pose_filter)
            self.poses_base.append(base_pose)
        else:
            if self.poses_base:
                self.filter_poses_base.append(self.filter_poses_base[-1])  # filtered pose
                self.poses_base.append(self.poses_base[-1])  # no filter
            else:
                self.filter_poses_base.append(np.zeros(7))
                self.poses_base.append(np.zeros(7))

        '''Plot on the frame'''
        # self.draw_path_on_frame(self.frame)

        return self.frame, self.object_poses[-1], self.poses[-1], self.filter_poses_base[-1]

    def draw_path_on_frame(self, frame):
        """
        Draws the historical path of the object's centers and corners on the frame.
        """
        corner_colors = [
            (230, 230, 250),  # Lavender
            (176, 196, 222),  # Light Steel Blue
            (100, 149, 237),  # Cornflower Blue
            (65, 105, 225)  # Royal Blue
        ]

        for pose in self.object_poses:
            # Draw center
            center = (int(pose[0][0]), int(pose[0][1]))  # Convert to integer for pixel coordinates
            cv2.circle(frame, center, radius=3, color='red', thickness=-1)

            # Draw corners
            for i, corner in enumerate(pose[1:5]):  # Assuming the first item is center, next 4 are corners
                corner_pos = (int(corner[0]), int(corner[1]))
                cv2.circle(frame, corner_pos, radius=3, color=corner_colors[i % 4], thickness=-1)

    def get_pose(self,
                 store,
                 object_size,
                 display_fps=False,
                 auto_terminate=True,
                 terminate_length=1000,
                 display=False,
                 record=False,
                 show_plot=True,
                 fixed_pose=False,
                 release=True,
                 quat=False):
        """
        Keep returning pose at each frame until the specified terminate_length is reached.

        :param release:
        :param fixed_pose:
        :param object_size:
        :param store:
        :param display_fps:
        :param auto_terminate:
        :param terminate_length:
        :param display:
        :param record:
        :param show_plot:
        :return:
        """
        # Variables for FPS calculation
        frames_to_skip = 30  # Number of initial frames to skip for stabilization
        frame_count = 0
        start_time = time.time()

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        if display:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Create a resizable window
            cv2.resizeWindow('Frame', 640, 480)  # Set the window size
        if self.video_writer is None and record is True:
            self.video_writer = cv2.VideoWriter(f'output_video_{self.timestamp}.avi',
                                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                                (frame_width, frame_height))

        try:
            while True:
                if frame_count < frames_to_skip:
                    frame_count += 1
                    continue
                frame_count += 1

                if auto_terminate:
                    if frame_count > terminate_length:
                        break

                ''' --------------------------- Key part --------------------------- '''
                if not quat:
                    _, _, _, _, _ = self.pose_estimation(display=display, object_size=object_size, object_aruco_id=3)
                else:
                    self.pose_estimation_quat(display=display, object_size=object_size, object_aruco_id=3)
                ''' --------------------------------------------------------------- '''

                # record
                if record:
                    self.video_writer.write(self.frame)

                # Display FPS
                if display_fps:
                    # Calculate FPS
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    height, width, _ = self.frame.shape
                    print(f"RGB Resolution: {width}x{height}")

                # Display the frame
                if display:
                    cv2.imshow("Frame", self.frame)

                # Check for the 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if release:
                # Release the camera
                self.cap.release()
                if record:
                    self.video_writer.release()  # Make sure to release the video writer
                if display:
                    cv2.destroyAllWindows()

        self.generate_summary()

        if store:
            if not quat:
                pickle_file_path = f'poses_data_{self.timestamp}.pkl'
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(self.poses, file)
                file_path = f'filter_poses_data_{self.timestamp}.pkl'
                with open(file_path, 'wb') as file:
                    pickle.dump(self.filter_poses, file)
                print(f"Poses data saved to {pickle_file_path}")
                print(f"Filtered poses data saved to {file_path}")
            else:
                pass

        if fixed_pose:
            if not quat:
                for i in range(6):
                    self.mean_std_list.append(self.get_mean_std(np.array(self.filter_poses)[:, i]))
                print(self.mean_std_list)
            else:
                pass

        if show_plot:
            if not quat:
                self.plot_quat_scatter(self.poses, onlyCentre=True)
                self.plot_track(self.poses, self.filter_poses)
                self.plot_track(self.poses_base, self.filter_poses_base)
                self.plot_tracking_success_rate()
                plt.show()
            else:
                self.plot_quat_scatter(self.object_poses)
                self.plot_track_quat(self.poses, self.object_poses, self.loss_track_history, self.poses_base, self.filter_poses_base)
                # self.plot_tracking_success_rate()
                # plt.show()

    def get_pose_from_video(self,
                            object_size,
                            display=False,
                            show_plot=True
                            ):
        """
        This function estimate the pose based on the video input, for calibration evaluation

        :param object_size:
        :param display:
        :param show_plot:
        :return:
        """
        for _ in range(20):
            self.total_frames -= 1
            if self.total_frames == 0:
                return True
            ''' --------------------------- Key part --------------------------- '''
            frame, _, _, _, _, _ = self.pose_estimation(display, object_size, 3)
            ''' --------------------------------------------------------------- '''

            # Display the frame
            if display:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Frame', 640, 480)
                cv2.imshow('Frame', self.frame)

        for i in range(6):
            self.mean_std_list.append(self.get_mean_std(np.array(self.filter_poses)[:, i]))
            self.mean_std_list_base.append(self.get_mean_std(np.array(self.filter_poses_base[:, i])))

        if show_plot:
            self.plot_track(self.poses, self.filter_poses)
            self.plot_tracking_success_rate()
            plt.show()

        return False

    def test_filter(self):
        """
        This function tests the result of median and final filter

        :return:
        """
        pickle_file_path = 'poses_data.pkl'
        with open(pickle_file_path, 'rb') as file:
            poses = pickle.load(file)

        filter_poses = []
        filter_poses_1 = []
        for pos in poses:
            filter_poses.append(self.filter.apply_filter(pos))
            filter_poses_1.append(self.filter.apply_filter_median(pos))

        time_axis = np.arange(len(poses))
        labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
        plt.figure(figsize=(8, 8))
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            if i < 3:
                plt.plot(time_axis, np.array(poses)[:, i] * 100, label='Pose', color='midnightblue')
                plt.plot(time_axis, np.array(filter_poses_1)[:, i] * 100, label='Smooth Pose Median', color='r',
                         linewidth='1')
                plt.plot(time_axis, np.array(filter_poses)[:, i] * 100, label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Value/cm')
                plt.legend(fontsize='small')
                plt.tight_layout()
            else:
                plt.plot(time_axis, np.rad2deg(np.array(poses)[:, i]), label='Pose', color='midnightblue')
                plt.plot(time_axis, np.rad2deg(np.array(filter_poses_1)[:, i]), label='Smooth Pose Median', color='r',
                         linewidth='1')
                plt.plot(time_axis, np.rad2deg(np.array(filter_poses)[:, i]), label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Angle/rad')
                plt.legend(fontsize='small')
                plt.tight_layout()

        y_mean_median, y_std_median = self.get_mean_std(np.array(filter_poses_1)[:, 1])
        y_mean_final, y_std_final = self.get_mean_std(np.array(filter_poses)[:, 1])
        z_mean_median, z_std_median = self.get_mean_std(np.array(filter_poses_1)[:, 2])
        z_mean_final, z_std_final = self.get_mean_std(np.array(filter_poses)[:, 2])
        pitch_mean_median, pitch_std_median = self.get_mean_std(np.array(filter_poses_1)[:, 3])
        pitch_mean_final, pitch_std_final = self.get_mean_std(np.array(filter_poses)[:, 3])
        roll_mean_median, roll_std_median = self.get_mean_std(np.array(filter_poses_1)[:, 4])
        roll_mean_final, roll_std_final = self.get_mean_std(np.array(filter_poses)[:, 4])
        yaw_mean_median, yaw_std_median = self.get_mean_std(np.array(filter_poses_1)[:, 5])
        yaw_mean_final, yaw_std_final = self.get_mean_std(np.array(filter_poses)[:, 5])

        print("---------------------------------------------")
        print("| Median Filter ")
        print(f"| Y -- mean: {y_mean_median * 100} cm, std: {y_std_median * 100} cm")
        print(f"| Z -- mean: {z_mean_median * 100} cm, std: {z_std_median * 100} cm")
        print(f"| Pitch -- mean: {np.rad2deg(pitch_mean_median)} deg, std: {np.rad2deg(pitch_std_median)} deg")
        print(f"| Roll -- mean: {np.rad2deg(roll_mean_median)} deg, std: {np.rad2deg(roll_std_median)} deg")
        print(f"| Yaw -- mean: {np.rad2deg(yaw_mean_median)} deg, std: {np.rad2deg(yaw_std_median)} deg")
        print("---------------------------------------------")
        print("| Final Filter ")
        print(f"| Y -- mean: {y_mean_final * 100} cm, std: {y_std_final * 100} cm")
        print(f"| Z -- mean: {z_mean_final * 100} cm, std: {z_std_final * 100} cm")
        print(f"| Pitch -- mean: {np.rad2deg(pitch_mean_final)} deg, std: {np.rad2deg(pitch_std_final)} deg")
        print(f"| Roll -- mean: {np.rad2deg(roll_mean_final)} deg, std: {np.rad2deg(roll_std_final)} deg")
        print(f"| Yaw -- mean: {np.rad2deg(yaw_mean_final)} deg, std: {np.rad2deg(yaw_std_final)} deg")
        print("---------------------------------------------")
        plt.show()

    def check_pose(self):
        """
        This is for experiment purpose, to check if everything works fine and pose keeps returning

        :return:
        """
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

                # if frame_count > 500:
                #     break

                output, pose, _, _, _ = self.pose_estimation(display=True, object_size=0.035, object_aruco_id=3)
                if np.all(pose):
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

    def generate_summary(self):
        """
        Report
            * Tracking loss rate: this can be used as the probability of the cover the marker in simulation
        """
        print("| ------------------------------------ |")
        print("| SUMMARY ")
        print(f"| Tracking Loss Rate: {self.loss_track_history.count(1) / len(self.loss_track_history)}")
        print("| ------------------------------------ |")

    @staticmethod
    def read_video_frames(video_path):
        """
        This function is used to check the video frame is read properly

        :param video_path:
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Failed to open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")

        frame_count = 0
        # Read and display each frame
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            frame_count += 1
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        print(f"Total frames read: {frame_count}")
        cap.release()
        cv2.destroyAllWindows()

    def plot_tracking_success_rate(self):
        time_axis = np.arange(len(self.loss_track_history))
        plt.subplot(4, 2, 7)
        plt.plot(time_axis, self.loss_track_history, label='Lose Track History')
        plt.title("Lose Track History")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(fontsize='6')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_track(poses, filter_poses):
        """
        This function is used for plot the history of pose estimation process and other information

        :param poses:
        :param filter_poses:
        :return:
        """
        print(np.shape(poses))
        time_axis = np.arange(len(poses))
        # labels = ['x', 'y', 'z', 'x', 'y', 'z', 'w']
        labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        plt.figure(figsize=(8, 8))
        for i in range(6):
            plt.subplot(4, 2, i + 1)
            if i < 3:
                plt.plot(time_axis, np.array(poses)[:, i] * 100, label='Pose', color='midnightblue')
                plt.plot(time_axis, np.array(filter_poses)[:, i] * 100, label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Value/cm')
                plt.legend(fontsize='small')
                plt.tight_layout()
            else:
                plt.plot(time_axis, np.array(poses)[:, i], label='Pose', color='midnightblue')
                plt.plot(time_axis, np.array(filter_poses)[:, i], label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Angle/deg')
                plt.legend(fontsize='small')
                plt.tight_layout()

    @staticmethod
    def plot_track_quat(poses, filter_poses, loss_track_history, poses_base, filter_poses_base):
        """
        This function is used for plot the history of pose estimation process and other information

        :param poses:
        :param filter_poses:
        :return:
        """
        print(np.shape(filter_poses))
        time_axis = np.arange(len(poses))
        labels = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
        plt.figure(figsize=(8, 8))
        for i in range(6):
            plt.subplot(4, 2, i + 1)
            if i < 3:
                plt.plot(time_axis, np.array(poses)[:, i] * 100, label='Pose', color='midnightblue')
                plt.plot(time_axis, np.array(filter_poses)[:, 0, i] * 100, label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Value/cm')
                plt.legend(fontsize='small')
                plt.tight_layout()
            else:
                plt.plot(time_axis, np.array(poses)[:, i], label='Pose', color='midnightblue')
                plt.plot(time_axis, np.array(filter_poses)[:, 0, i], label='Smooth Pose', color='orange',
                         linewidth='1')
                plt.title(labels[i])
                plt.xlabel('Time/s')
                plt.ylabel('Angle/deg')
                plt.legend(fontsize='small')
                plt.tight_layout()

        plt.subplot(4, 2, 7)
        plt.plot(time_axis, np.array(poses_base)[:, 0] * 100, label='Base - x', color='midnightblue')
        plt.plot(time_axis, np.array(filter_poses_base)[:, 0] * 100, label='Base - x', color='orange')
        # plt.plot(time_axis, np.array(poses_base)[:, 1] * 100, label='Base - y', color='midnightblue')
        plt.title("Base Pose")
        plt.xlabel('Time/s')
        plt.ylabel('Value/cm')
        plt.legend(fontsize='small')
        plt.tight_layout()

        plt.subplot(4, 2, 8)
        plt.plot(time_axis, loss_track_history, label='Lose Track History', color='midnightblue')
        plt.title("Lose Track History")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(fontsize='small')
        plt.tight_layout()

        plt.show()
    @staticmethod
    def get_mean_std(data):
        """
        Calculate the pose by finding the mean of the filtered pose of a certain period of time.
        :param data:
        :return:
        """
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

    @staticmethod
    def aruco_display(corners, ids, rejected, image):
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

    def store_poses(self, pose_list):
        if not pose_list:
            if not self.poses:  # if no aruco maker pose and the list is empty
                self.loss_track_history.append(1)
                self.poses.append(np.zeros(6))
                pose_list.append(np.zeros(6))
            else:  # use last pose if there is one
                self.loss_track_history.append(1)
                self.poses.append(self.poses[-1])
                pose_list.append(self.poses[-1])
        else:
            if len(pose_list) != 1:  # more than one aruco marker is detected
                self.loss_track_history.append(2)
            else:  # Everything is normal
                self.loss_track_history.append(0)
            self.poses.append(pose_list[0])
        return pose_list

    def store_poses_base(self, base_pose_list):
        if not base_pose_list:
            if not self.poses_base:  # if no aruco maker pose and the list is empty
                self.loss_track_history_base.append(1)
                self.poses_base.append(np.zeros(6))
                base_pose_list.append(np.zeros(6))
            else:  # use last pose if there is one
                self.loss_track_history_base.append(1)
                self.poses_base.append(self.poses_base[-1])
                base_pose_list.append(self.poses[-1])
        else:
            if len(base_pose_list) != 1:  # more than one aruco marker is detected
                self.loss_track_history_base.append(2)
            else:  # Everything is normal
                self.loss_track_history_base.append(0)
            self.poses_base.append(base_pose_list[0])
        return base_pose_list

    def plot_quat_scatter(self, object_poses, onlyCentre=False):
        # Define a list of colors for the corners
        corner_colors = ['lavender', 'lightsteelblue', 'cornflowerblue', 'royalblue']

        if not onlyCentre:
            for i, data in enumerate(object_poses):
                center = data[0]
                corners = data[1:5]

                # Plot center
                plt.scatter(center[0], center[1], s=50, zorder=5, color='red', label='Center' if i == 0 else "")
                plt.scatter(self.poses_base[i][0], self.poses_base[i][1], s=50, zorder=5, color='red', label='Base' if i == 0 else "")

                # Plot corners with different colors
                for j, corner in enumerate(corners):
                    plt.scatter(corner[0], corner[1], s=50, color=corner_colors[j],
                                label=f'Corner {j + 1}' if i == 0 else "")

                # Connect corners to form a square/rectangle and connect the last corner back to the first
                for j in range(len(corners)):
                    next_corner = corners[(j + 1) % len(corners)]  # Use modulo to wrap around
                    plt.plot([corners[j][0], next_corner[0]], [corners[j][1], next_corner[1]], 'gray', linestyle='--', linewidth=0.5)

                # Optional: connect center to corners
                for corner in corners:
                    plt.plot([center[0], corner[0]], [center[1], corner[1]], 'gray', linestyle='--', linewidth=0.5)
        else:
            for i, data in enumerate(object_poses):
                center = data[0]
                plt.scatter(center[0], center[1], s=50, zorder=5, color='red', label='Center' if i == 0 else "")

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.axis('equal')
        plt.title('Scatter Plot of Centers and Corners')
        if not onlyCentre:
            plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def swap_xy_axes(env_obs_vision):
        """
        Swaps the x and y axes for each 3D coordinate in the given dictionary.

        :param env_obs_vision: A dictionary containing 3D coordinates in NumPy arrays.
        :return: A dictionary with the x and y axes swapped for all 3D coordinates.
        """
        for key in env_obs_vision.keys():
            if isinstance(env_obs_vision[key], np.ndarray):
                # Check if the array represents multiple points or a single point
                if env_obs_vision[key].ndim == 2:  # For 2D arrays
                    env_obs_vision[key][:, [0, 1]] = env_obs_vision[key][:, [1, 0]]
                elif env_obs_vision[key].ndim == 1 and len(env_obs_vision[key]) >= 2:  # For 1D array
                    env_obs_vision[key][[0, 1]] = env_obs_vision[key][[1, 0]]
                # No else needed, as we only expect 1D or 2D arrays
        return env_obs_vision

    def camera_to_base_frame(self, object_poses, base_poses):
        """
        This provides the coordinate relative to the aruco marker on the base

        :param object_poses:
        :param base_poses:
        :return:
        """
        base_camera = np.array(base_poses)[:3]
        actuator_height = 0.025

        corner_camera = np.array(object_poses)[1:5][:, :3]
        corner_gripper = corner_camera - base_camera
        corner_gripper[:, 2] = actuator_height

        object_camera = object_poses[0][:3]
        object_gripper = object_camera - base_camera  # with respect to the aruco on the base link
        object_gripper[2] = actuator_height

        # this is the offset of the aruco relative to the base surface of the base link
        aruco_offset_on_base = [-0.018, 0, 0]

        env_obs_vision = {
            'left_xm': np.array([-0.08025, 0.0366, actuator_height])-np.array(aruco_offset_on_base),
            'right_xm': np.array([-0.08025, -0.0366, actuator_height])-np.array(aruco_offset_on_base),
            'object_corner': corner_gripper,
            'object_centre': object_gripper,
        }

        return self.swap_xy_axes(env_obs_vision)

    def get_obs_aruco(self, object_size):
        frame, object_pose_filter, object_pose, base_pose = self.pose_estimation_quat(display=False, object_size=object_size, object_aruco_id=3)
        obs_dict = self.camera_to_base_frame(object_pose_filter, base_pose)
        return obs_dict

    def init_camera_for_obs(self, object_size):
        """
        Wait for the camera and filters to stable
        :return:
        """
        self.load_camera_params()
        self.init_camera()
        for _ in range(50):
            self.get_obs_aruco(object_size)


if __name__ == "__main__":
    '''Get euler pose'''
    aruco_pose = ARUCO()
    # aruco_pose.load_camera_params()
    # aruco_pose.init_camera()
    # aruco_pose.get_pose(store=False,
    #                     object_size=0.0335,
    #                     display_fps=False,
    #                     auto_terminate=True,
    #                     terminate_length=100,
    #                     display=True,
    #                     record=False,
    #                     show_plot=True,
    #                     quat=True)

    '''Get obs for policy'''
    aruco_pose.init_camera_for_obs(0.0355)
    print(aruco_pose.get_obs_aruco(0.0355))

    '''From video'''
    # aruco_pose.load_camera_params()
    # aruco_pose.init_camera('/Users/qiyangyan/Desktop/FYP/Vision/output.mp4')
    # aruco_pose.get_pose_from_video(object_size=0.0335,
    #                                display=True,
    #                                show_plot=True)

    '''Real time'''
    # aruco_pose.pose_estimation(object_size=0.0335, display=True, object_aruco_id=)

    # aruco_pose.test_filter()

    # aruco_pose.read_video_frames('/Users/qiyangyan/Desktop/FYP/Vision/output.mp4')