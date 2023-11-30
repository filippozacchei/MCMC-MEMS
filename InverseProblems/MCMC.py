import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cuqi
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH
from cuqi.geometry import Continuous1D, Discrete

from utils import *

# Constants
CONFIGURATION_FILE = '../SurrogateModeling/Config_files/config_VoltageSignal_temp.json'
TESTING_PATH = 'TESTING_PATH'
INITIAL_GUESS = np.array([0.3, 0.0, 30.0])
BOUNDS = ([0.1, -0.5, 29.0], [0.5, 0.5, 31.0])
NOISE_FACTORS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
N = int(6e5)
Nb = int(1e5)
Nt = 5
REAL_COLOR = 'red'
SURROGATE_COLOR = 'blue'
LINE_WIDTH = 1.5
OUTPUT_FILENAME = 'noise.csv'
PARAMETER_START_POINTS = [np.array([0.3, 0.0, 30.0]),  
                          np.array([0.4,0.25,30.0]), 
                          np.array([0.2,0.25,30.0]), 
                          np.array([0.4,-0.25,30.0]),
                          np.array([0.2,-0.25,30.0])]

# Adjust system path for local module imports
sys.path.append('../DATA/')
sys.path.append('../SurrogateModeling/')

from dataLoader import DataProcessor
from model import NN_Model

def main():
    print("Loading model and configuration...")
    nn_model, config = load_model_and_config(CONFIGURATION_FILE)
    data_processor = DataProcessor(CONFIGURATION_FILE)
    data_processor.process()
    forward_model = create_forward_model_function(data_processor, nn_model)

    # Create a CUQI model based on the forward model function
    cuqi_model = Model(forward=forward_model, range_geometry=Continuous1D(len(data_processor.time)), domain_geometry=Discrete(["Overetch", "Offset", "Thickness"]))

    # Load and process test data
    test_processor = DataProcessor(CONFIGURATION_FILE)
    test_processor.load_data(file_name=TESTING_PATH)
    output_columns = test_processor.df.columns[5:-1]
    X_values, y_values = test_processor.df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (test_processor.df[output_columns].values)

    # List for storing results
    data_list = []

    # Main processing loop
    for i in range(1, X_values.shape[0], 100):
        print(f"Processing sample {i}...")
        x_true, y_true = X_values[i,:], y_values[i,:]
        print(x_true)
        for noise_factor in NOISE_FACTORS:
            noise = (noise_factor * np.mean(y_true))**2
            y_observed = Gaussian(mean=y_true, cov=noise * np.eye(len(data_processor.time))).sample()
            
            # Perform least squares optimization
            print("  Performing least squares optimization...")
            optimized_params, covariance_matrix = least_squares_optimization(y_observed, forward_model, INITIAL_GUESS, BOUNDS)
            print(optimized_params)

            # Define distributions for the Bayesian inference
            x_distribution = Uniform(low=np.array([0.1, -0.5, 29.0]), high=np.array([0.5, 0.5, 31.0]))
            y_distribution = Gaussian(mean=cuqi_model(x_distribution), cov=noise * np.eye(len(data_processor.time)))
            posterior = JointDistribution(x_distribution, y_distribution)(y_distribution=y_observed)

            # Perform sampling using MH
            samples_mh = [setup_markov_chain_sampler(posterior, noise, start_point).sample_adapt(N, 0).burnthin(Nb, Nt) for start_point in PARAMETER_START_POINTS]

            # Plotting and data collection
            plot_results(data_processor.time, y_true, y_observed, forward_model, samples_mh[0], REAL_COLOR, LINE_WIDTH)
            samples_mh[0].plot_trace()
            for j in range(3):
                plot_parameter_distribution(samples_mh[0].samples[j, :], x_true[j], ['Overetch', 'Offset', 'Thickness'][j])

            # Computing diagnostics and collecting results
            print(samples_mh[0].compute_rhat(samples_mh[1:]))
            ci = samples_mh[0].compute_ci()
            mean = samples_mh[0].mean()
            data_list.append({'x_true_0': x_true[0],
                            'x_true_1': x_true[1], 
                            'x_true_2': x_true[2],
                            'noise': noise_factor, 
                            'mean_0': mean[0],
                            'mean_1': mean[1],
                            'mean_2': mean[2], 
                            'ci_lower_0': ci[0][0],
                            'ci_lower_1': ci[0][1],
                            'ci_lower_2': ci[0][2],
                            'ci_upper_0': ci[1][0],
                            'ci_upper_1': ci[1][1],
                            'ci_upper_2': ci[1][2]})
            print(data_list[-1])

    # Save and print the data
    data = pd.DataFrame(data_list)
    print(data)
    data.to_csv(OUTPUT_FILENAME, index=False)

if __name__ == "__main__":
    main()



