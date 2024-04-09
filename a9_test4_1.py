import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Initialize Kalman Filter Parameters
# Define state transition matrix (F), measurement matrix (H), covariance matrices (Q and R)
# Initial F setup, dt will be dynamically updated
F = np.array([[1, 0, 0, 1, 0, 0],  
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

Q = np.eye(6)  # Process noise covariance
R = np.eye(3)  # Measurement noise covariance

# Step 3: Generate Measurements and Times from CSV File
def generate_measurements_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    measurements = df.iloc[:, [7, 8, 9]].to_numpy()  # Extract measurement columns (range, azimuth, elevation)
    times = df.iloc[:, 11].to_numpy()  # Extract time column
    # Remove NaN values from times and corresponding measurements
    valid_indices = ~np.isnan(times)
    measurements = measurements[valid_indices]
    times = times[valid_indices]
    return measurements, times

# Step 4: Kalman Filtering with Dynamic Time Update
def kalman_filter(measurements, times):
    if len(measurements) == 0:
        print("No measurements found in the CSV file.")
        return [], None, None
    
    num_steps = measurements.shape[0]
    filtered_states = []
    state_estimate = np.zeros((6, 1))  # Initial state estimate with all variables (range, azimuth, elevation, vx, vy, vz)
    P = np.eye(6)  # Initial state covariance
    previous_time = times[0]  # Initialize with the first time stamp

    filtered_fx = []
    filtered_fy = []
    filtered_fz = []
    filtered_fvx = []
    filtered_fvy = []
    filtered_fvz = []

    for i, (z, time) in enumerate(zip(measurements, times)):
        dt = time - previous_time
        # Update the state transition matrix F with the new dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Prediction
        state_estimate_priori = np.dot(F, state_estimate)
        P_priori = np.dot(np.dot(F, P), F.T) + Q

        # Update
        K = np.dot(np.dot(P_priori, H.T), np.linalg.inv(np.dot(np.dot(H, P_priori), H.T) + R))
        innovation = z.reshape(-1, 1) - np.dot(H, state_estimate_priori)  # Reshape z for matrix operation
        state_estimate_posteriori = state_estimate_priori + np.dot(K, innovation)
        P_posteriori = np.dot((np.eye(6) - np.dot(K, H)), P_priori)

        state_estimate = state_estimate_posteriori
        P = P_posteriori
        previous_time = time
        
        filtered_states.append(state_estimate)
        filtered_fx.append(state_estimate[0])
        filtered_fy.append(state_estimate[1])
        filtered_fz.append(state_estimate[2])
        filtered_fvx.append(state_estimate[3])
        filtered_fvy.append(state_estimate[4])
        filtered_fvz.append(state_estimate[5])
    
    # Convert filtered_states to numpy array
    filtered_states = np.array(filtered_states).squeeze()
    filtered_fx = np.array(filtered_fx).squeeze()
    filtered_fy = np.array(filtered_fy).squeeze()
    filtered_fz = np.array(filtered_fz).squeeze()
    filtered_fvx = np.array(filtered_fvx).squeeze()
    filtered_fvy = np.array(filtered_fvy).squeeze()
    filtered_fvz = np.array(filtered_fvz).squeeze()
    
    return filtered_states, P, filtered_fx, filtered_fy, filtered_fz, filtered_fvx, filtered_fvy, filtered_fvz  # Return filtered states, covariance matrix, and filtered state components

# Main function to run simulation
def main():
    csv_file_measurements = "test2.csv"  # Change this to your CSV file path for measurements

    # Generate measurements from CSV file
    measurements, times = generate_measurements_from_csv(csv_file_measurements)

    # Perform Kalman filtering
    filtered_states, final_covariance, filtered_fx, filtered_fy, filtered_fz, filtered_fvx, filtered_fvy, filtered_fvz = kalman_filter(measurements, times)
    if len(filtered_states) == 0:
        print("No filtered states found.")
        return

    # Print predicted values
    print("Predicted Values (Calculated from Code):")
    for i in range(len(times)):
        print("Time: {:.2f}, Filtered State (fx): {:.6f}, Filtered State (fy): {:.6f}, Filtered State (fz): {:.6f}, Filtered Velocity State (fvx): {:.6f}, Filtered Velocity State (fvy): {:.6f}, Filtered Velocity State (fvz): {:.6f}".format(times[i], filtered_fx[i], filtered_fy[i], filtered_fz[i], filtered_fvx[i], filtered_fvy[i], filtered_fvz[i]))

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot measured values
    plt.subplot(3, 1, 1)
    plt.plot(times, measurements[:, 0], label='Measured Range', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Range')
    plt.title('Range vs Time')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, measurements[:, 1], label='Measured Azimuth', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Azimuth')
    plt.title('Azimuth vs Time')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, measurements[:, 2], label='Measured Elevation', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Elevation')
    plt.title('Elevation vs Time')
    plt.legend()

    # Plot filtered states
    plt.subplot(3, 1, 1)
    plt.plot(times, filtered_fx, label='Filtered State (fx)', color='green')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, filtered_fy, label='Filtered State (fy)', color='green')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, filtered_fz, label='Filtered State (fz)', color='green')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
