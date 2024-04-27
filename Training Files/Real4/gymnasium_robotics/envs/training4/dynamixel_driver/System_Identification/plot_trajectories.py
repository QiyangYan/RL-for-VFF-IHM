import pandas as pd
import matplotlib.pyplot as plt

files = ['/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 1/Model_Dynamics_sinusoidal_vary_f_20240202_214058.csv',
         '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 2/Model_Dynamics_sinusoidal_vary_f_20240202_214303.csv',
         '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 3/Model_Dynamics_sinusoidal_vary_f_20240202_214502.csv',
         '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 4/Model_Dynamics_sinusoidal_vary_f_20240202_214807.csv',
         '/Users/qiyangyan/Desktop/FYP/Sim2Real/XM430 Calibration 5/Model_Dynamics_sinusoidal_vary_f_20240202_221026.csv'
        ]

# Create a figure for plotting
plt.figure(figsize=(6, 4))

for file in files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Plot "t" vs "position"
    # Assuming your CSV files have columns named 't' and 'position'.
    # If the column names are different, adjust them accordingly.
    plt.plot(df['t'], df['Goal Position'], linewidth=1, color=(75/255,134/255,140/255))  # Splitting the path to get a label

# Configure the plot
plt.title('Position Over Time for Various Calibrations')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend(loc='best')
plt.tight_layout()

# Display the plot
plt.show()
