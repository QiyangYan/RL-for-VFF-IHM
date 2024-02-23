import serial
# import keyboard
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

import cv2
import os
from subprocess import PIPE, run
import signal
import sys
import csv
import datetime


max_mean_values = [443.8350000000064, 347.8600000000006, 551.9349999999977, 737.7100000000064, 771.5050000000047, 790.0800000000017, 355.75500000000466, 859.5449999999983, 786.3699999999953, 958.7250000000058, 1110.179999999993, 843.7649999999994, 2394.854999999996, 778.1150000000052]
# Load the model
model_reg = pickle.load(open('model_reg.sav', 'rb'))

com = serial.Serial('/dev/tty.usbmodem1201', 115200, timeout=0.01) #commented for debug
sensorFlag = 0
mean_sensors = []  # Initialize mean_sensors
max_sensors = []  # Initialize max_sensors
sensor_avg = []  # Initialize sensor_avg
counter = 0

dir_path = ''
current_datatime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = os.path.join(dir_path, f'{current_datatime}.csv')

with open(dir_path + file_name, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    Table_head = ['Time (s)', 'Predicted (Tactile)', 'Predicted (Visual)']
                    csv_writer.writerow(Table_head)


# Create a new figure and two subplots, sharing both axes
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# # Initialize x and y arrays to store the data
x_data = []
y_data = [[] for _ in range(2)]
sensor_data = [[] for _ in range(14)]  # Create a list for each sensor

# # Create line objects for the data
# line_plot, = ax1.plot(x_data, y_data[0], label = 'tact. predict.')
# line_plot_vis, = ax1.plot(x_data, y_data[1], label = 'visual tracking')
# lines_sensor = [ax2.plot(x_data, sensor_data[i], label=f'Sensor {i+1}')[0] for i in range(14)]

# # Set the plot limits
# ax1.set_xlim([0, 100])
# ax1.set_ylim([0, 80])
# ax2.set_ylim([0, 1])  # Adjust the y-axis limit based on your sensor readings range
# ax1.set_title('Predicted x')
# ax2.set_title('Sensor readings')
# ax1.set_ylabel('x (mm)')
# ax2.set_ylabel('Normalized sensor readings')
# ax2.legend(loc='upper left')
# ax1.legend(loc='upper left')


camera_name = "USB  Camera"
command = ['ffmpeg','-f', 'avfoundation','-list_devices','true','-i','""']
result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
cam_id = -1

for item in result.stderr.splitlines():
    if camera_name in item:
        cam_id = int(item.split("[")[2].split(']')[0])
print("cam id", cam_id)


cam = cv2.VideoCapture(cam_id)

def interrupt_handler(signum, frame):
    print(f'Handling signal {signum} ({signal.Signals(signum).name}).')
    cam.release()


    time.sleep(3)
    sys.exit(0)

def main():
    # ani = FuncAnimation(fig, update_plot, interval=100)  # Update every 100 milliseconds
    global counter, dir_path
    maxCount = 20000 #200
    
    while counter < maxCount:
        update_plot()
        # print(counter)
    plt.show()
    # Show the plot
    # plt.show()



def update_plot():
    def deg2rad(deg):
        return deg/180*np.pi

    def rad2deg(rad):
        return  rad*180/np.pi

    def rotate(vector,deg):
        rad = deg2rad(deg)
        rotationMatrix = np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
        return np.matmul(rotationMatrix,vector)

    def angle(vec1,vec2):
        # return -rad2deg(np.arccos(np.dot(np.ndarray.flatten(vec1),np.ndarray.flatten(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))
        return -rad2deg(np.arctan2(vec2[1][0]*vec1[0][0]-vec2[0][0]*vec1[1][0],vec1[0][0]*vec2[0][0]+vec1[1][0]*vec2[1][0]))

    def distToFinger_pix(P):
        P = np.ndarray.flatten(P)
        fingerangle = angle(unitvector_x,np.array([[-1],[0]]))
        return -np.cos(deg2rad(fingerangle))*(fingerCorners[1][1]-P[1])+np.sin(deg2rad(fingerangle))*(fingerCorners[1][0]-P[0])
    
    

    
    global sensorFlag, mean_sensors, max_sensors,counter  # Add this line to declare variables as global
    #debug:
    # sensorFlag = 1
    # serial_data = "600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600"



    #debug end

    serial_data = com.readline() #commented for debug
    if (serial_data == b'Done\r\n'):
        sensorFlag = 1
        print('receiving tactile data')
                    
    elif (sensorFlag == 1):
        serial_data = serial_data.decode().strip() #commented for debug
        data = serial_data.split(' ')  # Split the line by space
        if len(data) == 16:
            # print(data)

            if (len(sensor_avg) < 15):
                # if np.mean(np.array(data[1:]).astype(float)) > 500:
                sensor_avg.append(data[1:])
                mean_sensors = [np.mean(np.array(sensor_avg)[:, i].astype(float)) for i in range(14)]
                max_sensors = [np.max(np.array(sensor_avg)[:, i].astype(float)) + max_mean_values[i] for i in range(14)]
                print(len(sensor_avg))
            
            else:



                print('visual tracking')
                s, img = cam.read()
                if s:    # frame captured without any errors
                    counter += 1
                    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                    arucoParams = cv2.aruco.DetectorParameters_create()
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
                        parameters=arucoParams)
                    if ids is not None:
                        if len(ids) == 2:
                            if (ids[0][0] == 0 and ids[1][0] == 11) or (ids[0][0] == 11 and ids[1][0] == 0):         
                                for index,id in enumerate(ids):
                                    aruco_id = id[0]
                                    if aruco_id == 0:
                                        objectCorners = corners[index][0] #objectCorner[whichCorner][whichDimension]

                                    if aruco_id == 11:
                                        fingerCorners = corners[index][0] #objectCorner[whichCorner][whichDimension]

                            
                                unitvector_x = np.divide(np.subtract(fingerCorners[1],fingerCorners[2]),18.5) #1mm from finger base towards tip
                                unitvector_x = np.array([[unitvector_x[0]],[unitvector_x[1]]])
                                unitvector_y = rotate(unitvector_x,90)
                                pix2mmScale = 1/np.linalg.norm(unitvector_x)

                                
                                touchcounter = 0
                                for objectCorner in objectCorners:
                                    if distToFinger_pix(objectCorner)*pix2mmScale > -18: #-15
                                        touchcounter += 1
                                        if touchcounter == 1:
                                            projectedPoint = objectCorner + unitvector_y.flatten()*(distToFinger_pix(objectCorner)*pix2mmScale)
                                            dist_x = np.linalg.norm(projectedPoint-fingerCorners[1])*pix2mmScale-9
                                        if touchcounter == 2:
                                            projectedPoint = objectCorner + unitvector_y.flatten()*(distToFinger_pix(objectCorner)*pix2mmScale)
                                            dist_x = (dist_x + np.linalg.norm(projectedPoint-fingerCorners[1])*pix2mmScale-9)/2
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

                sensor = [(float(data[i+1]) - mean_sensors[i]) / (max_sensors[i] - mean_sensors[i]) for i in range(14)]

                # print('-----------------')
                # print(sensor)

                
                
                if max(sensor) < 0.1:
                    # print('No prediction')
                    x_data.append(time.time())
                    y_data[0].append(np.nan)  # Add NaN for no prediction

                    if max(sensor) < 0.05:
                        # Update the mean and max sensor values, make it as a queue, pop the first element and append the new one
                        sensor_avg.pop(0)
                        sensor_avg.append(data[1:])
                        mean_sensors = [np.mean(np.array(sensor_avg)[:, i].astype(float)) for i in range(14)]
                        max_sensors = [np.max(np.array(sensor_avg)[:, i].astype(float)) + max_mean_values[i] for i in range(14)]
                        # print(mean_sensors)

                else:
                    x_pred = model_reg.predict([sensor])[0][0] + 18
                    
                    # # Smooth the prediction by taking the average with the last prediction
                    # if len(y_data) > 0 and not np.isnan(y_data[-1]):
                    #     x_pred = 0.5 * x_pred + 0.5 * y_data[-1]
                    #     print('Prediction: ', x_pred)
                    #     print('last prediction: ', y_data)

                    # Check if there are at least 10 readings
                    if len(y_data[0]) >= 10:
                        # Get the last 5 readings
                        last_five_readings = y_data[0][-10:]
                        
                        # Check if any of them are NaN
                        if not any(np.isnan(reading) for reading in last_five_readings):
                            # Calculate the average of the last 5 readings
                            average_last_five = np.mean(last_five_readings)
                            
                            # Update the prediction
                            x_pred = 0.5 * x_pred + 0.5 * average_last_five
                            
                            # print('Prediction: ', x_pred)
                            # print('last prediction: ', y_data[0])

                    x_data.append(time.time())
                    y_data[0].append(x_pred)

                # Update the line data
                # line_plot.set_data(x_data, y_data[0])
                # line_plot_vis.set_data(x_data, y_data[1])
                # Update sensor data for each sensor


                #SAVE INTO CSV: Time, visual, tactile prediction












                for i in range(14):
                    sensor_data[i].append(float(sensor[i]))
                    # lines_sensor[i].set_data(x_data, sensor_data[i])

                # Update the plot limits
                # ax1.set_xlim([min(x_data), max(x_data)])
                # ax2.set_ylim([min(min(sensor_data)), max(max(sensor_data))])
                with open(dir_path + file_name, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    # csv_writer.writerow([x_data[-1], y_data[0][-1], y_data[1][-1]])
                    csv_writer.writerow([x_data[-1], y_data[0][-1], y_data[1][-1]])

if __name__ == '__main__':
    signal.signal(signal.SIGINT, interrupt_handler)
    main()



# Close the serial port when 'c' is pressed
# if keyboard.is_pressed('c'):
#     com.close()
