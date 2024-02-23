import serial
# import keyboard
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

from cv2 import aruco
import cv2
import os
from subprocess import PIPE, run
import signal
import sys
import csv
import datetime


class CAMERA:

    def __init__(self, cam_index=0):
        # Attempt to open the first external camera
        self.cam = cv2.VideoCapture(cam_index)

        if not self.cam.isOpened():
            print(f"Failed to open camera at index {cam_index}. Trying next available index...")
            self.cam = None
            # Increment through camera indexes to find one that works
            for i in range(1, 10):
                self.cam = cv2.VideoCapture(i)
                if self.cam.isOpened():
                    print(f"Camera found and opened at index {i}")
                    break
                else:
                    self.cam = None

        if self.cam is None:
            print("No external USB camera found.")

    def get_Aruco(self):
        def deg2rad(deg):
            return deg / 180 * np.pi

        def rad2deg(rad):
            return rad * 180 / np.pi

        def rotate(vector, deg):
            rad = deg2rad(deg)
            rotationMatrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            return np.matmul(rotationMatrix, vector)

        def angle(vec1, vec2):
            # return -rad2deg(np.arccos(np.dot(np.ndarray.flatten(vec1),np.ndarray.flatten(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))
            return -rad2deg(np.arctan2(vec2[1][0] * vec1[0][0] - vec2[0][0] * vec1[1][0],
                                       vec1[0][0] * vec2[0][0] + vec1[1][0] * vec2[1][0]))

        def distToFinger_pix(P):
            P = np.ndarray.flatten(P)
            fingerangle = angle(unitvector_x, np.array([[-1], [0]]))
            return -np.cos(deg2rad(fingerangle)) * (fingerCorners[1][1] - P[1]) + np.sin(deg2rad(fingerangle)) * (
                        fingerCorners[1][0] - P[0])

        x_data = []
        y_data = [[] for _ in range(2)]

        if self.cam is None or not self.cam.isOpened():
            print("Camera is not available")
            return

        while cv2.waitKey(1) & 0xFF != ord('q'):
            s, img = self.cam.read()
            if s:  # frame captured without any errors
                # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("Camera Feed", img)
                arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
                arucoParams = aruco.DetectorParameters()
                (corners, ids, rejected) = aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.imshow("Camera Feed", img)
                    # cv2.drawFrameAxes(img, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                    if len(ids) == 2:
                        if (ids[0][0] == 0 and ids[1][0] == 11) or (ids[0][0] == 11 and ids[1][0] == 0):
                            for index, id in enumerate(ids):
                                aruco_id = id[0]
                                if aruco_id == 0:
                                    objectCorners = corners[index][0]  # objectCorner[whichCorner][whichDimension]

                                if aruco_id == 11:
                                    fingerCorners = corners[index][0]  # objectCorner[whichCorner][whichDimension]

                            unitvector_x = np.divide(np.subtract(fingerCorners[1], fingerCorners[2]),
                                                     18.5)  # 1mm from finger base towards tip
                            unitvector_x = np.array([[unitvector_x[0]], [unitvector_x[1]]])
                            unitvector_y = rotate(unitvector_x, 90)
                            pix2mmScale = 1 / np.linalg.norm(unitvector_x)

                            touchcounter = 0
                            for objectCorner in objectCorners:
                                if distToFinger_pix(objectCorner) * pix2mmScale > -18:  # -15
                                    touchcounter += 1
                                    if touchcounter == 1:
                                        projectedPoint = objectCorner + unitvector_y.flatten() * (
                                                    distToFinger_pix(objectCorner) * pix2mmScale)
                                        dist_x = np.linalg.norm(projectedPoint - fingerCorners[1]) * pix2mmScale - 9
                                    if touchcounter == 2:
                                        projectedPoint = objectCorner + unitvector_y.flatten() * (
                                                    distToFinger_pix(objectCorner) * pix2mmScale)
                                        dist_x = (dist_x + np.linalg.norm(
                                            projectedPoint - fingerCorners[1]) * pix2mmScale - 9) / 2
                            if touchcounter > 0:
                                y_data[1].append(dist_x)
                                print(dist_x)
                            else:
                                y_data[1].append(np.nan)
                        else:
                            y_data[1].append(np.nan)
                    else:
                        y_data[1].append(np.nan)
                else:
                    y_data[1].append(np.nan)
            else:
                y_data[1].append(np.nan)
                print('no camera feed')
                break


        self.cam.release()
        cv2.destroyAllWindows()

    def show_camera_feed(self):
        if self.cam is None or not self.cam.isOpened():
            print("Camera is not available")
            return

        while True:
            ret, frame = self.cam.read()
            print("read")
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Camera Feed", frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = CAMERA()
    # print(dir(cv2.aruco))
    # camera.show_camera_feed()
    camera.get_Aruco()
