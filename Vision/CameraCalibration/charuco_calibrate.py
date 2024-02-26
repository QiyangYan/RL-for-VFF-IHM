import os
import numpy as np
import cv2

# Configuration parameters
ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_VERTICALLY = 14
SQUARES_HORIZONTALLY = 10
SQUARE_LENGTH = 0.019  # meters
MARKER_LENGTH = 0.0145  # meters
PATH_TO_YOUR_IMAGES = '/Users/qiyangyan/Desktop/FYP/Vision/CameraCalibration/images'


def calibrate_and_save_parameters():
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Create the CharucoBoard object
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard_create(SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, SQUARE_LENGTH, MARKER_LENGTH,
                                          dictionary)

    all_charuco_corners = []
    all_charuco_ids = []
    images_used = []

    # Load images
    image_files = [f for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
    image_files.sort()  # Ensure files are in order
    print("Image Load Complete")

    for image_file in image_files:
        # print("scanning")
        image_path = os.path.join(PATH_TO_YOUR_IMAGES, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load image at {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # If at least one marker detected
        if marker_ids is not None:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners,
                                                                                               marker_ids, gray, board)
            if charuco_retval > 0:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                images_used.append(image_path)

    if not all_charuco_corners:
        print("No corners detected across any images. Calibration failed.")
        return

    print("calibrating")
    # Calibrate camera using detected corners
    calibration, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None)

    if not calibration:
        print("Calibration could not be completed.")
        return

    # Save calibration data
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("Calibration successful. Parameters saved.")

    # Optionally: show undistorted images
    for image_path in images_used:
        image = cv2.imread(image_path)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None)
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(500)  # Display each for 500ms

    cv2.destroyAllWindows()


print("Starting calibration...")
calibrate_and_save_parameters()
print("Calibration process complete.")
