import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cuqi
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH,NUTS
from cuqi.geometry import Continuous1D, Discrete


from utils import *

# Constants
CONFIGURATION_FILE = '../SurrogateModeling/Config_files/config_VoltageSignal.json'
CONFIGURATION_FILE2 = '../SurrogateModeling/Config_files/config_VoltageSignal_st.json'
INITIAL_GUESS = np.array([0.3, 0.0, 30.0])
BOUNDS = ([0.1, -0.5, 29.0], [0.5, 0.5, 31.0])
NOISE_FACTORS = 1e-6*np.array([1000])
B = np.sqrt(200)
S = 5
N = int(6e5)
Nb = int(1e5)
Nt = 5
REAL_COLOR = 'red'
SURROGATE_COLOR = 'blue'
LINE_WIDTH = 1.5
OUTPUT_FILENAMEs = ["samples_"+str(i) for i in range(10)]

PARAMETER_START_POINTS = [np.array([0.3, 0.0, 30.0])]
                        #   np.array([0.4,0.25,30.0]), 
                        #   np.array([0.2,0.25,30.0]), 
                        #   np.array([0.4,-0.25,30.0]),
                        #   np.array([0.2,-0.25,30.0])]

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
    data_processor2 = DataProcessor(CONFIGURATION_FILE2)
    data_processor2.process()
    forward_model = create_forward_model_function(data_processor, nn_model)
    gradient_model = create_gradient_function(data_processor, nn_model)
    # Create a CUQI model based on the forward model function
    cuqi_model = Model(forward=forward_model, 
                       jacobian=gradient_model,
                       range_geometry=Continuous1D(len(data_processor.time)), 
                       domain_geometry=Discrete(["Overetch", "Offset", "Thickness"]))
    X_values, y_values = data_processor.X_test, data_processor.y_test

    # List for storing results
    data_list = []

    # Main processing loop
    for i in range(10,X_values.shape[0],10):
        print(f"Processing sample {i}...")
        x_true, y_true = X_values[i,:], y_values[i,:]
        print("True values: ", x_true)
        for noise_factor in NOISE_FACTORS:
            noise = (noise_factor*B*S)**2
            print("Noise: ", noise)
            y_observed = Gaussian(mean=y_true, cov=noise * np.eye(len(data_processor.time))).sample()
            
            # Perform least squares optimization
            print("Performing least squares optimization...")
            INITIAL_GUESSES = []

            for k in range(len(PARAMETER_START_POINTS)):
                optimized_params, covariance_matrix = least_squares_optimization(y_observed, forward_model, PARAMETER_START_POINTS[k], BOUNDS)
                INITIAL_GUESSES.append(optimized_params)
                print("Optimized Params: ", optimized_params)

            print(INITIAL_GUESSES)
            param = INITIAL_GUESSES[0]
            print("PARAM: ", param)
            # Define distributions for the Bayesian inference
            # x_distribution = Uniform(low=np.array([0.1, -0.5, 29.0]), high=np.array([0.5, 0.5, 31.0]))
            x_distribution = Gaussian(mean=np.array([0.3, 0.0, 30.0]), cov=np.array([0.01, 0.01, 0.5]))
            # param = INITIAL_GUESSES[0]
            # print("PARAM: ", param)
            y_distribution = Gaussian(mean=cuqi_model(x_distribution), cov=noise * np.eye(len(data_processor.time)))
            param = INITIAL_GUESSES[0]
            print("PARAM: ", param)
            posterior = JointDistribution(x_distribution, y_distribution)(y_distribution=y_observed)

            # Perform sampling using MH
            samples_mh = [setup_markov_chain_sampler(posterior, noise, param).sample_adapt(N, 0) for param in INITIAL_GUESSES]

            # NUTSsampler = NUTS(posterior,x0=param, max_depth=5)
            # NUTSsamples = NUTSsampler.sample_adapt(N,Nb)
            # samples_mh = [NUTSsamples]
            print("Effective sample_size: ", samples_mh[0].compute_ess())
            samples_array = [sample for sample in samples_mh]

            # Save the numpy array to a file
            np.save(OUTPUT_FILENAMEs[int(i/10)], samples_array[0].samples)  

            samples_mh[0].plot_trace()

            # Plotting and data collection
            # plot_results(data_processor.time, forward_model(x_true), y_observed, forward_model, samples_mh[0], REAL_COLOR, LINE_WIDTH)
            # for j in range(3):
            #     plot_parameter_distribution(samples_mh[0].samples[j, :], x_true[j], ['Overetch', 'Offset', 'Thickness'][j])

            # Computing diagnostics and collecting results
            # print("Rhat: ", samples_array[0].compute_rhat(samples_array[1:]))

if __name__ == "__main__":
    main()



