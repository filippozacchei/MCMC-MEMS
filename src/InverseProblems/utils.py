import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde
import cuqi
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH
from cuqi.geometry import Continuous1D, Discrete
import time
import tensorflow as tf

# Adjust system path for local module imports
sys.path.append('../SurrogateModeling/')
sys.path.append('../utils/preprocessing')

from model import NN_Model
from preprocessing import preprocessing

def create_forward_model_function(data_processor, nn_model):
    """
    Creates a forward model function using the provided data processor and neural network model.
    """
    # Pre-compute static data manipulations
    num_time_points = len(data_processor.time)
    time_reshaped = data_processor.time.reshape(-1, 1)

    def forward_model(x):
        # If x is always the same size, consider optimizing these operations
        x_repeated = np.tile(x, (num_time_points, 1))
        
        # Combining time and x values
        x_combined = np.hstack((x_repeated,time_reshaped))

        # Scale and predict in one step
        scaled_data = data_processor.scaler.transform(x_combined)
        #output = nn_model.predict(scaled_data).flatten()
        output = nn_model.model(scaled_data).numpy().reshape(150,)
        return output
        
    return forward_model

def create_gradient_function(data_processor, nn_model):
    """
    Creates a function to compute the Jacobian of the output of a neural network model
    with respect to its input, using the given data processing parameters, and returns the result
    as a squeezed NumPy array.

    Args:
    data_processor: An object containing the scaler mean, scale, and time information.
    nn_model: A TensorFlow neural network model.

    Returns:
    A function that computes and returns the Jacobian of the neural network output with respect to its input as a squeezed NumPy array.
    """
    # Precompute constants and convert to TensorFlow tensors
    scaler_mean = tf.convert_to_tensor(data_processor.scaler.mean_, dtype=tf.float32)
    scaler_scale = tf.convert_to_tensor(data_processor.scaler.scale_, dtype=tf.float32)
    time_reshaped = tf.convert_to_tensor(data_processor.time.reshape(-1, 1), dtype=tf.float32)
    num_time_points = tf.shape(time_reshaped)[0]

    def compute_jacobian(x):
        """Computes the Jacobian of the neural network output with respect to the input x and returns a squeezed NumPy array."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_repeated = tf.tile(tf.expand_dims(x, 0), [num_time_points, 1])
        x_combined = tf.concat([x_repeated, time_reshaped], axis=-1)

        with tf.GradientTape() as tape:
            tape.watch(x_combined)
            scaled_data = (x_combined - scaler_mean) / scaler_scale
            output = nn_model.model(scaled_data)
            
        # Compute the Jacobian of the output with respect to x
        jacobian = tape.gradient(output, x_combined)

        # Squeeze to remove dimensions of size 1 and convert to NumPy array
        jacobian_squeezed = (jacobian.numpy())[:,:3]

        return jacobian_squeezed
    return compute_jacobian

def least_squares_optimization(y_observed, forward_model, start_point, bounds):
    """
    Performs least squares optimization to fit the model to the observed data.
    """
    result = least_squares(objective_function, start_point, args=([y_observed, forward_model]), bounds=bounds, jac='3-point')
    return result.x, np.linalg.inv(result.jac.T @ result.jac)

def objective_function(input, exact_outputs, forward_model):
    predicted_outputs = forward_model(input).reshape(exact_outputs.shape)
    residuals = predicted_outputs - exact_outputs
    return residuals

def plot_results(time, y_true, y_obs, forward_model, samplesMH, REAL_COLOR='red', LINE_WIDTH=1.5):
    plt.figure()
    plt.plot(1e3*time, y_true, c=REAL_COLOR, label='Real', linewidth=LINE_WIDTH)
    plt.plot(1e3*time, forward_model(samplesMH.mean()), 'green', label='Pred', linewidth=LINE_WIDTH)
    plt.plot(1e3*time, y_obs, '.-b', label='Noisy Signal', linewidth=LINE_WIDTH)
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\Delta C$ [fF]')
    plt.title(f'Overetch = {samplesMH.mean()[0]:.4f}μm; Offset = {samplesMH.mean()[1]:.4f}μm; Thickness = {samplesMH.mean()[2]:.4f}μm', fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_parameter_distribution(parameter_samples, x_true, parameter_name):
    plt.figure()
    kernel_density = gaussian_kde(parameter_samples)
    x_range = np.linspace(np.min(parameter_samples), np.max(parameter_samples), 1000)
    plt.plot(x_range, kernel_density(x_range), label='Density', linewidth=2)
    plt.axvline(x_true, color='red', label='Exact', linestyle='-', linewidth=2)
    mean, mode = np.mean(parameter_samples), x_range[np.argmax(kernel_density(x_range))]
    plt.axvline(mean, color='green', label='Mean', linestyle='--', linewidth=2)
    plt.axvline(mode, color='blue', label='Mode', linestyle='--', linewidth=2)
    lower_bound, upper_bound = np.percentile(parameter_samples, [2.5, 97.5])
    plt.fill_between(x_range, 0, kernel_density(x_range), where=((x_range >= lower_bound) & (x_range <= upper_bound)), alpha=0.3, color='gray', label='95% C.I.')
    plt.xlabel(parameter_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()