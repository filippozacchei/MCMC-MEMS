import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the directory containing the dataLoader module to the sys.path
sys.path.append('../DATA/')
sys.path.append('../SurrogateModeling/')

# Import the load_data function from the dataLoader modulec
from dataLoader import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

noise_factors = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
N=int(1.25e5)
Nb=int(2.5e4)
Nt=5

# CONFIGURATION FILE
CONFIGURATION_FILE = '../SurrogateModeling/Config_files/config_VoltageSignal.json'
config = parse_config(CONFIGURATION_FILE)  
model_path = '../SurrogateModeling/' + config['MODEL_PATH']
forward_model_tf = tf.keras.models.load_model(model_path)

config = parse_config(CONFIGURATION_FILE)  

# Data preparation
C_df, dC_df = load_data_derivative(config)
output_cols = dC_df.columns[5:-1]

X_train, X_test, y_train, y_test = split_data(C_df, config)

# Flatten the output and concatenate time column
time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
X_train_rep, y_train_rep = stack_data(X_train, y_train, time)
X_test_rep, y_test_rep = stack_data(X_test, y_test, time)

# Data shuffling and scaling
X_train_rep, y_train_rep = shuffle_data(X_train_rep, y_train_rep)
X_train_scaled, X_test_scaled, scaler = scale_data(X_train_rep, X_test_rep, 'minmax')  

def forward_model(x):
    x2 = tf.convert_to_tensor(np.array(x),dtype=tf.float32)
    output = forward_model_tensorflow(x2)
    return output

import tensorflow as tf

def forward_model_tensorflow(x):
    """Prepares input data and predicts output using a TensorFlow model.

    Args:
        x (array-like): Input features to the model.
        output_cols (int): Number of output columns to replicate the data for.
        time (array-like): Time steps to append to the input features.
        scaler (MinMaxScaler): Scaler object used for normalizing the data.
        forward_model (tf.keras.Model): Pre-trained TensorFlow model for prediction.

    Returns:
        array-like: The predicted output from the TensorFlow model.
    """
    # Convert input to a TensorFlow tensor and ensure the correct shape
    concatenated_tensor = tf.reshape(x, (1, -1))
    concatenated_tensor = tf.tile(concatenated_tensor, (len(output_cols), 1))

    # Convert the time vector to a tensor and concatenate it with the input features
    time_tensor = tf.convert_to_tensor(time, dtype=tf.float32)
    time_tensor = tf.reshape(time_tensor, (len(output_cols), 1))
    input_tensor = tf.concat([concatenated_tensor, time_tensor], axis=1)

    # Scale the concatenated tensor using the scaler's parameters
    data_mean = tf.convert_to_tensor(scaler.data_min_, dtype=tf.float32)
    data_scale = tf.convert_to_tensor(scaler.data_range_, dtype=tf.float32)
    scaled_input_tensor = (input_tensor - data_mean) / data_scale

    # Use the forward model to predict the output
    predictions = forward_model_tf(scaled_input_tensor)
    return predictions.numpy().flatten()

# Example usage
# You need to define `output_cols`, `time`,

# Define the objective function for least squares
def objective_function(input_tensor,output):
    predicted_outputs = forward_model(input_tensor)
    residuals = predicted_outputs - output
    return residuals

def compute_jacobian(wrt):    

    input_tensor = tf.convert_to_tensor(wrt, dtype=tf.float32)

    # Use GradientTape to compute the Jacobian
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_tensor)
        output = forward_model_tensorflow(input_tensor)

    # Compute the Jacobian matrix
    jacobian = tape.jacobian(output, input_tensor)
    result = tf.squeeze(jacobian).numpy()
    return result

# Uncertainty Quantififcation ENvironment
import cuqi
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH, NUTS
from cuqi.geometry import Continuous1D, Discrete

A = Model(forward=forward_model,
          range_geometry=Continuous1D(len(time)),
          domain_geometry=Discrete(["Overetch","Offset","Thickness"]))

# Define colors and line styles
real_color = 'red'
surrogate_color = 'blue'
line_width = 1.5

from scipy.optimize import least_squares, curve_fit
from scipy.stats import norm

# C_df, dC_df = load_data_derivative(config,file='TESTING_PATH')
# output_cols = dC_df.columns[5:-1]
# X_test, y_test = C_df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (C_df[output_cols].values)
# # data = pd.DataFrame(columns=['x_true', 'noise', 'mean', 'ci'])
data_list=[]

for i in range(1,X_test.shape[0],100):
    for noise_factor in noise_factors:
        print(noise_factor)
        x_true = X_test[i,:]
        y_true = y_test[i,:]
        noise=(noise_factor*np.mean(y_true))**2
        print(x_true)
        # tx  = Gaussian(mean=x_true[3],cov=1e-8)
        # etch = Uniform(low=0.3,high=0.5)
        y_obs = Gaussian(mean=y_true, cov=noise*(np.eye(len(time)))).sample()

        # print(y_obs)
        # jac = compute_jacobian(x_true[[0,3,6]])
        # print(jac.shape)

        x0 = np.array([0.3, 0.0, 30.0])
        x1 = np.array([0.4,0.25,30.0])
        x2 = np.array([0.2,0.25,30.0])
        x3 = np.array([0.4,-0.25,30.0])
        x4 = np.array([0.2,-0.25,30.0])
        x5 = np.array([0.2,0.25,30.5])
        x6 = np.array([0.2,0.25,29.5])
        x7 = np.array([0.2,-0.25,30.5])
        x8 = np.array([0.2,-0.25,29.5])

        # stringa = "../Chains_dV/Ax"+str(x_true[0])+"_Tx"+str(x_true[1])+"_Etch"+str(x_true[2])
        # plt.figure()
        # plt.plot(time, A(x_true), c=real_color, label='Real', linewidth=line_width)
        # plt.plot(time, A(x0), 'green', label='Pred', linewidth=line_width)
        # plt.plot(time, y_obs, '.b', label='Noisy', linewidth=line_width)
        # # Set axis labels
        # plt.xlabel('Time [ms]')
        # plt.ylabel(r'$\Delta C$ [fF/s]')
        # # Set title with relevant information
        # plt.title(
        #     f'Overetch = {X_test[i, 0]}μm; Offset = {X_test[i, 1]:.2f}μm; Thickness = {X_test[i, 2]:.2f}μm',
        #     fontsize=10, loc='center')
        # plt.legend(loc='upper right', fontsize=10)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.show()

        # Perform least squares optimization
        result = least_squares(objective_function,
                            x0,
                            args=([y_obs]), 
                            bounds=([0.1,
                                        -0.5,
                                        29.0],
                                    [0.5,
                                    0.5,
                                    31.0]), 
                            jac='3-point')

        optimized_params = result.x
        print(optimized_params)
        cov_matrix = np.linalg.inv(result.jac.T @ result.jac)

        x = Uniform(low=np.array([0.1,-0.5,29.0]),high=np.array([0.5,0.5,31.0]))
        y = Gaussian(mean=A(x), cov=noise*np.eye(len(time)))


        from cuqi.sampler import UGLA, Conjugate, ConjugateApprox, Gibbs

        # Define posterior distribution
        posterior = JointDistribution(x, y)(y=y_obs)
        
        MHsampler = MH(target=posterior,
                    proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                        cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                    x0=x0)
        samplesMH0 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        # MHsampler = MH(target=posterior,
        #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
        #                                  cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
        #                x0=x1)
        # samplesMH1 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        # MHsampler = MH(target=posterior,
        #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
        #                                  cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
        #                x0=x2)
        # samplesMH2 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        # MHsampler = MH(target=posterior,
        #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
        #                                  cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
        #                x0=x3)
        # samplesMH3 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        # MHsampler = MH(target=posterior,
        #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
        #                                  cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
        #                x0=x4)
        # samplesMH4 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        # print(samplesMH0.compute_rhat([samplesMH1,samplesMH2,samplesMH3,samplesMH4]))
        ci = samplesMH0.compute_ci()
        mean = samplesMH0.mean()
        data_list.append({'x_true': x_true, 'noise': noise_factor, 'mean': mean, 'ci': ci})
        print(data_list)

        # plt.figure()
        # plt.plot(time, y_test[i, :], c=real_color, label='Real', linewidth=line_width)
        # plt.plot(time, forward_model(samplesMH0.mean()), 'green', label='Pred', linewidth=line_width)
        # plt.plot(time, y_obs, '.b', label='Noisy', linewidth=line_width)
        # # Set axis labels
        # plt.xlabel('Time [ms]')
        # plt.ylabel(r'$\Delta C$ [fF/s]')
        # # Set title with relevant information
        # plt.title(
        #     f'Overetch = {samplesMH0.mean()[0]}μm; Offset = {samplesMH0.mean()[1]:.2f}μm; Thickness = {samplesMH0.mean()[2]:.2f}μm',
        #     fontsize=10, loc='center')
        # plt.legend(loc='upper right', fontsize=10)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.show()

        # samplesMH0.plot_trace()
        # samplesMH0.diagnostics()

        # from scipy import stats

        # for i in range(0,3):
        #     parameter_samples = samplesMH0.samples[i, :]
        #     plt.figure()

        #     # Plot the density of the samples
        #     kernel_density = stats.gaussian_kde(parameter_samples)
        #     x_range = np.linspace(np.min(parameter_samples), np.max(parameter_samples), 1000)
        #     plt.plot(x_range, kernel_density(x_range), label='Density', linewidth=2)

        #     # Plot vertical line at the exact parameter
        #     plt.axvline(x_true[i], color='red', label='Exact', linestyle='-', linewidth=2)

        #     # Calculate and plot mean and mode
        #     mean = np.mean(parameter_samples)
        #     mode = x_range[np.argmax(kernel_density(x_range))]
        #     plt.axvline(mean, color='green', label='Mean', linestyle='--', linewidth=2)
        #     plt.axvline(mode, color='blue', label='Mode', linestyle='--', linewidth=2)
        #     plt.axvline(optimized_params[i], color='black', label='LS', linestyle='--', linewidth=2)

        #     # Calculate and highlight the 95% credibility interval
        #     lower_bound, upper_bound = np.percentile(parameter_samples, [2.5, 97.5])
        #     plt.fill_between(x_range, 0, kernel_density(x_range), where=((x_range >= lower_bound) & (x_range <= upper_bound)), alpha=0.3, color='gray', label='95% C.I.')

        #     # Set axis labels and title
        #     plt.xlabel(['Overetch','Offset','Thickness'][i])
        #     plt.ylabel('Density')
        #     # Place the legend on the right side of the plot
        #     plt.legend()
        #     # Adjust the spacing between subplots
        #     plt.show()
        
    # Create DataFrame from the list of dictionaries
    data = pd.DataFrame(data_list)

    # At this point, data DataFrame contains all the inserted values
    print(data)
    data.to_csv('noise.csv', index=False)