import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

sys.path.append('../DATA/')

from dataLoader import load_data

# Define Input Parameters
sampling_frequency = 1e-3 / 1e-5
period = 2e-3
n_periods = 2
n_epochs = 50000
batch_size = 10
n_nr = 32
time = np.arange(0, 2.5e-3, 1e-5)
lr = 0.01

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/CapacityDataset.csv'
U_dataset_filename = '../DATA/CSV/UDataset.csv'
V_dataset_filename = '../DATA/CSV/VDataset.csv'

# Load the data using the load_data function and Obtain Capacity variation
C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=2.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=2.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=2.49ms'].values) / 1e-5

# Select the input and output columns
input_cols = ['AmplitudeX', 'T_X', 'Overetch']
output_cols = dC_df.columns[7:-1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dC_df[input_cols].values, 1e12 * (dC_df[output_cols].values), test_size=0.2, random_state=2)

# Flatten the output and concatenate time column
X_train_rep = np.repeat(X_train, len(output_cols), axis=0)
X_train_rep = np.column_stack((X_train_rep, np.tile(time, len(X_train_rep) // len(time))))
y_train_rep = y_train.flatten()

X_test_rep = np.repeat(X_test, len(output_cols), axis=0)
X_test_rep = np.column_stack((X_test_rep, np.tile(time, len(X_test_rep) // len(time))))
y_test_rep = y_test.flatten()

# Get the number of samples
num_samples = X_train_rep.shape[0]

# Generate a random permutation of indices
perm = np.random.permutation(num_samples)

X_train_rep = X_train_rep[perm]
y_train_rep = y_train_rep[perm]

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rep)

# Create a scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dC_df['AmplitudeX'], 1e3*dC_df['T_X'], dC_df['Overetch'], c='blue', label='Test Set', alpha=0.5)
ax.scatter(X_train[:, 0], 1e3*X_train[:, 1], X_train[:, 2], c='red', label='Training Set', alpha=0.7)

# Set labels and title
ax.set_xlabel('AmplitudeX', fontsize=12)
ax.set_ylabel('PeriodX [ms]', fontsize=12)
ax.set_zlabel(r'Overetch [$\mu$m]', fontsize=12)

xtick_values = [10*i for i in range(11)]
xtick_labels = [f'{i}g' for i in range(11)] # Generate the tick labels from 0g to 10g
plt.xticks(xtick_values, xtick_labels, fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.show()

# Assuming your test outputs are stored in a variable called 'test_outputs'
noise_std = 0.5                                         # Standard deviation of the Gaussian noise
noise = np.random.normal(0, noise_std, y_test.shape)        # Generating the noise
noisy_test_outputs = y_test + noise                         # Adding noise to the test outputs

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(time, noisy_test_outputs[0, :], 'b.', label='Noisy Output',alpha=0.5)
ax.plot(time, y_test[0, :], 'r-', label='Real Output')

# Set labels and title
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('dC', fontsize=12)
ax.set_title('Noisy and Real Outputs', fontsize=14)

# Set grid and legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

# Adjust tick labels and font size
ax.tick_params(axis='both', which='major', labelsize=10)

# Set tight layout
fig.tight_layout()

plt.show()

from scipy.optimize import least_squares, curve_fit
model = tf.keras.models.load_model('./modelNo1_latin.h5')


# Define your model function that takes input parameters and returns the predicted output
def model_function(time, amplitudeX, T_X, overetch):
    params_rep = np.repeat(np.array([amplitudeX, T_X, overetch]).reshape((1,3)), len(output_cols), axis=0)
    params_rep = np.column_stack((params_rep, np.tile(time, len(params_rep) // len(time))))
    # Assuming your model is already defined and loaded
    predicted_output = model.predict(scaler.transform(params_rep))
    return predicted_output.flatten()

# Define the objective function for least squares
def objective_function(params,output):
    predicted_outputs = model_function(time, params[0], params[1], params[2])
    residuals = predicted_outputs - output
    return residuals

# Import the scipy.stats module for probability distribution calculations
from scipy.stats import norm

for i in range(noisy_test_outputs.shape[0]):

    # Define the initial guess for the input parameters
    initial_params = [100,1.5e-3,0.5]  # Adjust the initial guess as needed

    print(noisy_test_outputs.shape)

    # Perform least squares optimization
    result = least_squares(objective_function,
                           initial_params,
                           args=([noisy_test_outputs[i,:]]), 
                           bounds=([0.0,0.5e-3,0.3],[100,1.5e-3,0.5]), 
                           jac='3-point')

    optimized_params = result.x

    cov_matrix = np.linalg.inv(result.jac.T @ result.jac)
    print(cov_matrix)

    # Calculate standard errors for each parameter
    standard_errors = np.sqrt(np.diagonal(cov_matrix))

    # Calculate the confidence intervals (95% for example)
    confidence_intervals = 1.96 * standard_errors

    # The optimized input parameters# The optimized input parameters
    optimized_amplitudeX, optimized_TX, optimized_overetch = optimized_params

    print("Real AmplitudeX:",      X_test[i,0])
    print("Real TX:",              X_test[i,1])
    print("Real Overetch:",        X_test[i,2])    
    print("Optimized AmplitudeX:", optimized_amplitudeX)
    print("Optimized TX:",         optimized_TX)
    print("Optimized Overetch:",   optimized_overetch)

    # Create separate plots for each parameter
    for param_idx, param_name in enumerate(['Max Amplitude', 'Inpulse Period', 'Overetch']):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the true parameter value as a vertical line
        true_value = X_test[i, param_idx]
        ax.axvline(x=true_value, ymin=0.01, ymax=0.99, color='red', linestyle='-', label='True Value', linewidth=2)

        # Plot the parameter prediction as a vertical line
        pred_value = optimized_params[param_idx]
        ax.axvline(x=pred_value, ymin=0.01, ymax=0.99, color='black', linestyle='--', label='Mean Value', linewidth=2)

        # Calculate the normal distribution associated with the standard deviation
        parameter_values = np.linspace(optimized_params[param_idx] - 5 * standard_errors[param_idx], optimized_params[param_idx] + 5 * standard_errors[param_idx], 100)
        normal_distribution = norm.pdf(parameter_values, loc=optimized_params[param_idx], scale=standard_errors[param_idx])

        # Plot the normal distribution
        ax.plot(parameter_values, normal_distribution, label='Posterior Distribution', color='blue', linewidth=2, alpha=0.8)

        # Highlight the 95% confidence interval
        lower_bound = optimized_params[param_idx] - 1.96 * standard_errors[param_idx]
        upper_bound = optimized_params[param_idx] + 1.96 * standard_errors[param_idx]
        ax.fill_between(parameter_values, normal_distribution, where=(parameter_values >= lower_bound) & (parameter_values <= upper_bound), alpha=0.2, color='gray', label='95% Confidence Interval')

        # Set labels and title
        ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
        ax.set_title(f'{param_name} Prediction', fontsize=16, fontweight='bold')

        # Set grid and legend
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, loc='upper left')

        # Remove top and right spines for better aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Increase tick label size for better readability
        ax.tick_params(axis='both', labelsize=12)

        # Remove x-axis ticks and label to avoid clutter
        # ax.set_xticks([])

        # Set y-axis minor ticks for grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, which='both')

        # Display the plot
        plt.tight_layout()  # Adjusts spacing to avoid overlap
        plt.show()