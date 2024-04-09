import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Initialize Kalman Filter Parameters
# Define state transition matrix (F), measurement matrix (H), covariance matrices (Q and R)
# Initial F setup, dt will be dynamically updated
F = np.array([[1, 0, 0, 0, 0, 0],  
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
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
def kalman_filter(measurements, times, fx, fy, fz, fvx, fvy, fvz):
    if len(measurements) == 0:
        print("No measurements found in the CSV file.")
        return [], None
    
    num_steps = measurements.shape[0]
    filtered_states = []
    state_estimate = np.array([[fx], [fy], [fz], [fvx], [fvy], [fvz]])  # Initial state estimate with provided values
    P = np.eye(6)  # Initial state covariance
    previous_time = times[0]  # Initialize with the first time stamp

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
    
    # Convert filtered_states to numpy array
    filtered_states = np.array(filtered_states).squeeze()
    
    return filtered_states, P  # Return filtered states and final covariance matrix

# Main function to run simulation
def main():
    csv_file_measurements = "test2.csv"  # Change this to your CSV file path for measurements

    # Provided Fx, Fy, Fz, Fvx, Fvy, Fvz values
    fx = -22155.2
    fy = 15273.03
    fz = 6.664073
    fvx = 24.34633
    fvy = 37.07997
    fvz = 6.316395

    # Generate measurements from CSV file
    measurements, times = generate_measurements_from_csv(csv_file_measurements)

    # Perform Kalman filtering
    filtered_states, final_covariance = kalman_filter(measurements, times, fx, fy, fz, fvx, fvy, fvz)
    if len(filtered_states) == 0:
        print("No filtered states found.")
        return

    # Print predicted values
    print("Predicted Values (Calculated from Code):")
    print("Time\t\tfx\t\tfy\t\tfz\t\tfvx\t\tfvy\t\tfvz")
    for i in range(len(times)):
        print("{:.2f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(times[i], 
                                                                               filtered_states[i][0], 
                                                                               filtered_states[i][1], 
                                                                               filtered_states[i][2], 
                                                                               filtered_states[i][3],
                                                                               filtered_states[i][4],
                                                                               filtered_states[i][5]))

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

    # Plot predicted values from code
    plt.subplot(3, 1, 1)
    plt.plot(times, filtered_states[:, 0], label='Predicted Range (Code)', color='red')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, filtered_states[:, 1], label='Predicted Azimuth (Code)', color='red')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, filtered_states[:, 2], label='Predicted Elevation (Code)', color='red')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
