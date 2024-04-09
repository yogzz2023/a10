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

Q = np.eye(6) * 1e-4  # Process noise covariance
R = np.eye(3) * 1e-3  # Measurement noise covariance

# Step 3: Generate Measurements and Times from CSV File
# Step 3: Generate Measurements and Times from CSV File
def generate_measurements_from_csv(csv_file):
    df = pd.read_csv(csv_file, usecols=[7, 8, 9], delimiter=',', skiprows=[0], header=None)  # Adjust parameters according to your CSV format
    measurements = df.to_numpy()[:, :11].astype(float)  # Convert to floats
    times = np.arange(len(measurements))  # Use index as time
    return measurements, times


# Step 4: Kalman Filtering with Dynamic Time Update
def kalman_filter(measurements, times):
    num_steps = measurements.shape[0]
    filtered_states = []
    state_estimate = np.zeros((6, 1))
    P = np.eye(6) * 1e-2  # Initial state covariance
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

    # Generate measurements from CSV file
    measurements, times = generate_measurements_from_csv(csv_file_measurements)

    # Perform Kalman filtering
    filtered_states, final_covariance = kalman_filter(measurements, times)

    # Print predicted values
    print("Predicted Values (Calculated from Code):")
    for i in range(len(times)):
        print("Time: {:.2f}, Predicted Range: {:.5f}, Predicted Azimuth: {:.5f}, Predicted Elevation: {:.5f}".format(times[i], filtered_states[i, 0], filtered_states[i, 1], filtered_states[i, 2]))

    # Plotting (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(times, measurements[:, 0], label='Measured Range', color='blue')
    plt.plot(times, filtered_states[:, 0], label='Predicted Range (Kalman Filter)', color='red')
    plt.xlabel('Time')
    plt.ylabel('Range')
    plt.title('Range vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

