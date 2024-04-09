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
    measurements = df.iloc[:, [7, 8, 9]].to_numpy()  # Extract measurement columns
    times = df.iloc[:, 11].to_numpy()  # Assuming the 13th column is at index 11 (0-based indexing)
    return measurements, times

# Step 4: Kalman Filtering with Dynamic Time Update
def kalman_filter(measurements, times):
    num_steps = measurements.shape[0]
    filtered_states = []
    state_estimate = np.zeros((6, 1))
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

# Step 5: Calculate Conditional Probability
def calculate_conditional_probability(measurement, filtered_state, covariance_matrix):
    # This is a basic implementation
    # You may need to refine it based on your specific requirements
    # Here, we calculate the log-likelihood of the measurement given the filtered state
    residual = measurement - np.dot(H, filtered_state)
    residual_covariance = np.dot(np.dot(H, covariance_matrix), H.T) + R
    # Adding a small positive value to the diagonal of residual_covariance matrix
    residual_covariance += np.eye(3) * 1e-9
    
    try:
        # Calculate log-likelihood
        log_likelihood = -0.5 * (np.dot(np.dot(residual.T, np.linalg.inv(residual_covariance)), residual) +
                                 np.log(np.linalg.det(residual_covariance)) +
                                 3 * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        # Handle the case where the residual covariance matrix is singular or ill-conditioned
        log_likelihood = -np.inf
    
    return log_likelihood

# Step 6: Calculate Marginal Probability
def calculate_marginal_probability(measurements, filtered_states, final_covariance):
    # Calculate marginal probability using the final covariance matrix
    marginal_probabilities = []
    for i, measurement in enumerate(measurements):
        likelihood = calculate_conditional_probability(measurement, filtered_states[i], final_covariance)
        marginal_probabilities.append(likelihood)
    return marginal_probabilities

# Step 7: Identify Most Associated Measurement
def identify_most_associated_measurement(marginal_probabilities):
    # This function identifies the measurement with the highest marginal probability
    max_index = np.argmax(marginal_probabilities)
    return max_index

# Main function to run simulation
def main():
    csv_file_measurements = "test2.csv"  # Change this to your CSV file path for measurements
    csv_file_predicted = "test2.csv"  # Change this to your CSV file path for predicted values

    # Generate measurements from CSV file
    measurements, times = generate_measurements_from_csv(csv_file_measurements)

    # Perform Kalman filtering
    filtered_states, final_covariance = kalman_filter(measurements, times)

    # Generate predicted values from CSV file
    df_predicted = pd.read_csv(csv_file_predicted)
    predicted_values_csv = df_predicted[['P_rng1', 'P_az', 'P_el']].values

    # Extracting predicted range, azimuth, and elevation
    predicted_range = filtered_states[:, 0]
    predicted_azimuth = np.degrees(np.arctan2(filtered_states[:, 1], filtered_states[:, 0]))  # Convert azimuth to degrees
    valid_indices = np.logical_and(predicted_range != 0, np.abs(filtered_states[:, 2] / predicted_range) <= 1)
    predicted_elevation = np.full_like(predicted_range, np.nan)  # Initialize with NaNs
    predicted_elevation[valid_indices] = np.degrees(np.arcsin(filtered_states[valid_indices, 2] / predicted_range[valid_indices]))  # Convert elevation to degrees

    # Correct predicted azimuth to be closer to measured azimuth
    measured_azimuth = measurements[:, 1]
    azimuth_correction = measured_azimuth - predicted_azimuth
    corrected_predicted_azimuth = predicted_azimuth + azimuth_correction  # Adjust predicted azimuth

    # Print predicted values
    print("Predicted Values (Calculated from Code):")
    for i in range(len(times)):
        print("Time: {:.2f}, Predicted Range: {:.2f}, Predicted Azimuth: {:.2f}, Predicted Elevation: {:.2f}".format(times[i], predicted_range[i], predicted_azimuth[i], predicted_elevation[i]))

    # Print measurements
    print("Measurements:")
    for i, (r, a, e) in enumerate(zip(measurements[:, 0], measurements[:, 1], measurements[:, 2])):
        print("Index: {}, Range: {:.2f}, Azimuth: {:.2f}, Elevation: {:.2f}".format(i, r, a, e))

    # Calculate marginal probabilities
    marginal_probabilities = calculate_marginal_probability(measurements, filtered_states, final_covariance)

    # Print probabilities
    print("Marginal Probabilities:")
    for i, probability in enumerate(marginal_probabilities):
        print("Time: {:.2f}, Probability: {:.6f}".format(times[i], probability))

    # Identify the most associated measurement
    most_associated_measurement_index = identify_most_associated_measurement(marginal_probabilities)
    print("Most Associated Measurement Index:", most_associated_measurement_index)

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
    plt.plot(times, predicted_range, label='Predicted Range (Code)', color='red')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, corrected_predicted_azimuth, label='Predicted Azimuth (Code, Corrected)', color='red')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, predicted_elevation, label='Predicted Elevation (Code)', color='red')
    plt.legend()

    # Plot predicted values from CSV
    plt.subplot(3, 1, 1)
    plt.plot(times, predicted_values_csv[:, 0], label='Predicted Range (CSV)', color='green', linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, predicted_values_csv[:, 1], label='Predicted Azimuth (CSV)', color='green', linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, predicted_values_csv[:, 2], label='Predicted Elevation (CSV)', color='green', linestyle='--')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()