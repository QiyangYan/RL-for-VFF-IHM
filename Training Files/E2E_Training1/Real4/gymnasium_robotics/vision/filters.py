import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import pickle


# Assuming poses_df is your DataFrame containing the pose data
def initialize_kalman_vars():
    return {
        'error_var': 1,
        'process_noise': 1e-1,
        'measurement_noise': 4e-4,
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
            self.window_size = 10
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