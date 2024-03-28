# import numpy as np
# import pandas as pd
# from scipy.signal import medfilt
#
#
# # Define the Kalman filter function
# def apply_kalman_filter(data, process_noise=1e-5, measurement_noise=1e-2, estimation_error=1):
#     x_est = data[0]
#     kalman_filtered = [x_est]
#     error_var = estimation_error
#
#     for measurement in data[1:]:
#         error_var += process_noise
#         kalman_gain = error_var / (error_var + measurement_noise)
#         x_est = x_est + kalman_gain * (measurement - x_est)
#         error_var = (1 - kalman_gain) * error_var
#         kalman_filtered.append(x_est)
#
#     return kalman_filtered
#
#
# # Define the combined Median and Kalman filter application function
# def apply_median_then_kalman(data, median_kernel_size=11):
#     median_filtered_data = medfilt(data, kernel_size=median_kernel_size)
#     kalman_filtered_data = apply_kalman_filter(median_filtered_data)
#     return kalman_filtered_data
#
# import pickle
# pickle_file_path = 'poses_data.pkl'
# with open(pickle_file_path, 'rb') as file:
#     poses_df = pickle.load(file)
#
# if isinstance(poses_df, list):
#     import numpy as np
#     # Assuming each array in the list represents a pose at a time step and has the structure [x, y, z, roll, pitch, yaw]
#     poses_df = pd.DataFrame(np.array(poses_df), columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw'])
#
#
# # Assuming 'poses_df' is your DataFrame with the pose data
# # Apply the combined Median and Kalman Filters to all parameters
# poses_df_median_filtered = poses_df[column_names].apply(lambda x: medfilt(x, kernel_size=11))
# poses_df_combined_filtered = poses_df[column_names].apply(lambda x: apply_median_then_kalman(x.values))
#
# # Plotting the results for all parameters
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
# axes_flat = axes.flatten()
#
# for i, col in enumerate(column_names):
#     axes_flat[i].plot(poses_df.index, poses_df[col], label='Original', alpha=0.5)
#     axes_flat[i].plot(poses_df.index, poses_df_median_filtered[col], label='Median Filtered', linestyle='--')
#     axes_flat[i].plot(poses_df.index, poses_df_combined_filtered[col], label='Combined Median & Kalman Filtered',
#                       linestyle='-.')
#     axes_flat[i].set_title(f'{col.upper()} Filtering Comparison')
#     axes_flat[i].legend()
#
# plt.tight_layout()
# plt.show()

# Load your time series data
# data_path = 'path_to_your_pickle_file.pkl'  # Replace with your file path
# poses_df = pd.read_pickle(data_path)

# # Function to apply Kalman filter for real-time data
# def apply_kalman_realtime(filtered_data, process_noise=1e-5, measurement_noise=1e-2, estimation_error=1):
#     kalman_filtered = []
#     x_est = filtered_data[0]
#     error_var = estimation_error
#
#     for measurement in filtered_data:
#         error_var += process_noise
#         kalman_gain = error_var / (error_var + measurement_noise)
#         x_est = x_est + kalman_gain * (measurement - x_est)
#         error_var = (1 - kalman_gain) * error_var
#         kalman_filtered.append(x_est)
#
#     return kalman_filtered
#
#
# # Simulate real-time processing
# window_size = 11
# median_window = deque(maxlen=window_size)
# realtime_filtered_data = {'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
#
# for index, row in poses_df.iterrows():
#     # Update median window and apply median filter
#     median_window.append(row)
#     if len(median_window) == window_size:
#         median_result = np.median(np.array(median_window), axis=0)
#     else:
#         median_result = np.array(median_window)[-1]  # Use the latest value if window is not full
#
#     # Update the realtime filtered data with the median result
#     for i, col in enumerate(['x', 'y', 'z', 'roll', 'pitch', 'yaw']):
#         realtime_filtered_data[col].append(median_result[i])
#
# # Apply Kalman filter to the median-filtered data
# for col in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
#     kalman_result = apply_kalman_realtime(realtime_filtered_data[col])
#     realtime_filtered_data[col] = kalman_result
#
# # At this point, `realtime_filtered_data` contains the data processed in a simulated real-time manner
# # You can convert it back to a DataFrame for visualization or further analysis
# realtime_filtered_df = pd.DataFrame(realtime_filtered_data)
#
# # Visualization or further processing can be done with `realtime_filtered_df`
#
# # Plotting the results for all parameters
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
# axes_flat = axes.flatten()
#
# for i, col in enumerate(column_names):
#     axes_flat[i].plot(poses_df.index, poses_df[col], label='Original', alpha=0.5)
#     # axes_flat[i].plot(poses_df.index, poses_df_median_filtered[col], label='Median Filtered', linestyle='--')
#     axes_flat[i].plot(poses_df.index, realtime_filtered_df[col], label='Combined Median & Kalman Filtered',
#                       linestyle='-.')
#     axes_flat[i].set_title(f'{col.upper()} Filtering Comparison')
#     axes_flat[i].legend()
#
# plt.tight_layout()
# plt.show()


import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import pickle


# Assuming poses_df is your DataFrame containing the pose data
def initialize_kalman_vars():
    return {
        'error_var': 1,
        'process_noise': 1e-5,
        'measurement_noise': 4e-5,
    }


def initialize_kalman_vars_base():
    return {
        'error_var': 1,
        'process_noise': 1e-1,
        'measurement_noise': 1e-1,
    }


class FILTER:
    def __init__(self, quat=False, base=False):
        # Real-time filtering simulation
        if quat:
            self.parameters = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
        else:
            self.parameters = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        if base:
            self.window_size = 50
            self.kalman_states = {param: initialize_kalman_vars_base() for param in self.parameters}
            self.median_windows = {param: [] for param in self.parameters}
        elif quat:
            self.window_size = 5
            self.kalman_states = {param: initialize_kalman_vars() for param in self.parameters}
            self.median_windows = {param: deque(maxlen=self.window_size) for param in self.parameters}
        else:
            self.window_size = 5
            self.kalman_states = {param: initialize_kalman_vars() for param in self.parameters}
            self.median_windows = {param: deque(maxlen=self.window_size) for param in self.parameters}

        self.realtime_filtered_data = {param: [] for param in self.parameters}
        self.moving_avg_windows = {param: [] for param in self.parameters}

        self.quat = quat
        self.base = base


    def reset_filter(self):
        self.realtime_filtered_data = {param: [] for param in self.parameters}
        self.median_windows = {param: deque(maxlen=self.window_size) for param in self.parameters}
        self.kalman_states = {param: initialize_kalman_vars() for param in self.parameters}

    def update_kalman(self, kalman_vars, x_est, measurement):
        """Update Kalman filter state with a new measurement."""
        kalman_vars['error_var'] += kalman_vars['process_noise']
        kalman_gain = kalman_vars['error_var'] / (kalman_vars['error_var'] + kalman_vars['measurement_noise'])
        x_est = x_est + kalman_gain * (measurement - x_est)
        kalman_vars['error_var'] = (1 - kalman_gain) * kalman_vars['error_var']
        return x_est

    # Initialize Kalman filter variables for each parameter

    def apply_filter(self, row):
        # initial_estimates = poses_df.iloc[0]
        estimate = []
        for idx, param in enumerate(self.parameters):
            # Update median window and apply filter
            self.median_windows[param].append(row[idx])
            if len(self.median_windows[param]) == self.window_size:
                median_result = np.median(np.array(self.median_windows[param]), axis=0)
            else:
                median_result = self.median_windows[param][-1]  # Use the latest value if window is not full

            # Update Kalman filter with the median-filtered data point
            if not self.realtime_filtered_data[param]:  # First data point uses the initial estimate
                # kalman_estimate = initial_estimates[param]
                kalman_estimate = row[idx]
            else:
                kalman_estimate = self.update_kalman(self.kalman_states[param], self.realtime_filtered_data[param][-1], median_result)

            self.realtime_filtered_data[param].append(kalman_estimate)
            estimate.append(kalman_estimate)
        return estimate

    def apply_filter_median(self, row):
        # initial_estimates = poses_df.iloc[0]
        estimate = []
        for idx, param in enumerate(self.parameters):
            # Update median window and apply filter
            self.median_windows[param].append(row[idx])
            if len(self.median_windows[param]) == self.window_size:
                median_result = np.median(np.array(self.median_windows[param]), axis=0)
            else:
                median_result = self.median_windows[param][-1]  # Use the latest value if window is not full
            estimate.append(median_result)
        return estimate

    def apply_filter_median_moving_avg(self, row, filter_passes=2):
        estimate = row
        for _ in range(filter_passes):  # Apply filtering process multiple times
            filtered_estimate = []
            for idx, param in enumerate(self.parameters):
                # Update and apply median filter
                self.median_windows[param].append(estimate[idx])
                if len(self.median_windows[param]) > self.window_size:
                    self.median_windows[param] = self.median_windows[param][-self.window_size:]
                median_result = np.median(self.median_windows[param]) if len(self.median_windows[param]) else estimate[
                    idx]

                # Update and apply moving average filter
                self.moving_avg_windows[param].append(median_result)
                if len(self.moving_avg_windows[param]) > self.window_size:
                    self.moving_avg_windows[param] = self.moving_avg_windows[param][-self.window_size:]
                moving_avg_result = np.mean(self.moving_avg_windows[param]) if len(
                    self.moving_avg_windows[param]) else median_result

                filtered_estimate.append(moving_avg_result)
            estimate = filtered_estimate  # Use filtered data for the next pass

        return estimate

    def visualise(self, poses_df, filter_poses_df):
        # Convert the real-time filtered data back to a DataFrame for visualization or further analysis
        realtime_filtered_df = pd.DataFrame(self.realtime_filtered_data)

        # Visualization for all parameters
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 7))
        for i, param in enumerate(self.parameters):
            ax = axes.flatten()[i]
            ax.plot(poses_df.index, poses_df[param], label='Original', alpha=0.5)
            # ax.plot(filter_poses_df.index, filter_poses_df[param], label='Original', alpha=0.5)
            ax.plot(filter_poses_df.index, filter_poses_df[param], label='Real-time Filtered', linestyle='--')
            ax.set_title(f'Real-time Filtering for {param.upper()}')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def process_csv(self):
        pickle_file_path = 'filter_poses_data.pkl'
        with open(pickle_file_path, 'rb') as file:
            filter_poses_df = pickle.load(file)
        with open('poses_data.pkl', 'rb') as file:
            poses_df = pickle.load(file)
        # Convert list of numpy arrays to DataFrame if necessary
        if isinstance(poses_df, list):
            poses_df = pd.DataFrame(poses_df, columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw'])
            filter_poses_df = pd.DataFrame(filter_poses_df, columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        # for i, row in poses_df.iterrows():
        #     poses_df[i] = self.apply_filter(row)
        filter_poses_df = poses_df.apply(lambda row: pd.Series(self.apply_filter_median_moving_avg(row), index=poses_df.columns), axis=1)
        self.visualise(poses_df, filter_poses_df)


if __name__ == "__main__":
    filter = FILTER(base=True)
    filter.process_csv()