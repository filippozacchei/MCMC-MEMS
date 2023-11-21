import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import least_squares
from scipy.stats import norm, gaussian_kde
import cuqi
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH
from cuqi.geometry import Continuous1D, Discrete

# Adjust the system path to include specific directories
sys.path.append('../DATA/')
sys.path.append('../SurrogateModeling/')

# Import necessary modules
from dataLoader import DataProcessor
from model import NN_Model
from sklearn.model_selection import train_test_split

noise_factors = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
N=int(5e6)
Nb=int(2.5e4)
Nt=5

# CONFIGURATION FILE
CONFIGURATION_FILE = '../SurrogateModeling/Config_files/config_VoltageSignal_temp.json'

config = DataProcessor.parse_config(CONFIGURATION_FILE)  
model_path = '../SurrogateModeling/' + config['MODEL_PATH']
forward_model_tf = NN_Model()
forward_model_tf.load_model(model_path=model_path)

dataProcessor = DataProcessor(CONFIGURATION_FILE)
dataProcessor.process() 

def forward_model(x):
    x_ini = x.reshape((1,3))
    x_rep = np.repeat(x_ini, len(dataProcessor.time), axis=0)
    x_rep = np.column_stack((x_rep, np.tile(dataProcessor.time, len(x_rep) // len(dataProcessor.time))))
    x_scaled = dataProcessor.scaler.transform(x_rep)
    output = forward_model_tf.predict(x_scaled).flatten()
    return output

# Define the objective function for least squares
def objective_function(input,exact_outputs):
    predicted_outputs = forward_model(input).reshape(exact_outputs.shape)
    residuals = predicted_outputs - exact_outputs
    return residuals

A = Model(forward=forward_model,
          range_geometry=Continuous1D(len(dataProcessor.time)),
          domain_geometry=Discrete(["Overetch","Offset","Thickness"]))

# Define colors and line styles
real_color = 'red'
surrogate_color = 'blue'
line_width = 1.5

# Data preparation
test_processor = DataProcessor(CONFIGURATION_FILE)
test_processor.load_data(file_name='TESTING_PATH')
output_cols = test_processor.df.columns[5:-1]
X_val, y_val = test_processor.df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (test_processor.df[output_cols].values)

data_list=[]

for i in range(1,X_val.shape[0],100):
    for noise_factor in noise_factors:

        x_true, y_true = X_val[i,:],y_val[i,:]

        noise=(noise_factor*np.mean(y_true))**2
        y_obs = Gaussian(mean=y_true, cov=noise*(np.eye(len(dataProcessor.time)))).sample()

        x0 = np.array([0.3, 0.0, 30.0])
        x1 = np.array([0.4,0.25,30.0])
        x2 = np.array([0.2,0.25,30.0])
        x3 = np.array([0.4,-0.25,30.0])
        x4 = np.array([0.2,-0.25,30.0])
        x5 = np.array([0.2,0.25,30.5])
        x6 = np.array([0.2,0.25,29.5])
        x7 = np.array([0.2,-0.25,30.5])
        x8 = np.array([0.2,-0.25,29.5])

        print(x_true)

        # Perform least squares optimization
        result = least_squares(objective_function,
                            x0,
                            args=([y_obs]), 
                            bounds=([0.1, -0.5, 29.0],
                                    [0.5, 0.5, 31.0]), 
                            jac='3-point')

        optimized_params = result.x
        print(optimized_params)
        cov_matrix = np.linalg.inv(result.jac.T @ result.jac)

        x = Uniform(low=np.array([0.1,-0.5,29.0]),high=np.array([0.5,0.5,31.0]))
        y = Gaussian(mean=A(x), cov=noise*np.eye(len(dataProcessor.time)))

        # Define posterior distribution
        posterior = JointDistribution(x, y)(y=y_obs)
        
        MHsampler = MH(target=posterior,
                       proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                         cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                       x0=x0)
        samplesMH0 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        MHsampler = MH(target=posterior,
                       proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                         cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                       x0=x1)
        samplesMH1 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        MHsampler = MH(target=posterior,
                       proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                         cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                       x0=x2)
        samplesMH2 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        MHsampler = MH(target=posterior,
                       proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                         cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                       x0=x3)
        samplesMH3 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        MHsampler = MH(target=posterior,
                       proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),
                                         cov=np.array([noise*1e-4,noise*1e-4,1e-2])),
                       x0=x4)
        samplesMH4 = MHsampler.sample_adapt(N,0).burnthin(Nb,Nt)

        print(samplesMH0.compute_rhat([samplesMH1,samplesMH2,samplesMH3,samplesMH4]))
        ci = samplesMH0.compute_ci()
        mean = samplesMH0.mean()
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
        print(data_list)
        time=dataProcessor.time

        plt.figure()
        plt.plot(time, y_true, c=real_color, label='Real', linewidth=line_width)
        plt.plot(time, forward_model(samplesMH0.mean()), 'green', label='Pred', linewidth=line_width)
        plt.plot(time, y_obs, '.b', label='Noisy', linewidth=line_width)
        # Set axis labels
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\Delta C$ [fF/s]')
        # Set title with relevant information
        plt.title(
            f'Overetch = {samplesMH0.mean()[0]}μm; Offset = {samplesMH0.mean()[1]:.2f}μm; Thickness = {samplesMH0.mean()[2]:.2f}μm',
            fontsize=10, loc='center')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        samplesMH0.plot_trace()

        samplesMH0.diagnostics()

        from scipy import stats

        for i in range(0,3):
            parameter_samples = samplesMH0.samples[i, :]
            plt.figure()

            # Plot the density of the samples
            kernel_density = stats.gaussian_kde(parameter_samples)
            x_range = np.linspace(np.min(parameter_samples), np.max(parameter_samples), 1000)
            plt.plot(x_range, kernel_density(x_range), label='Density', linewidth=2)

            # Plot vertical line at the exact parameter
            plt.axvline(x_true[i], color='red', label='Exact', linestyle='-', linewidth=2)

            # Calculate and plot mean and mode
            mean = np.mean(parameter_samples)
            mode = x_range[np.argmax(kernel_density(x_range))]
            plt.axvline(mean, color='green', label='Mean', linestyle='--', linewidth=2)
            plt.axvline(mode, color='blue', label='Mode', linestyle='--', linewidth=2)
            plt.axvline(optimized_params[i], color='black', label='LS', linestyle='--', linewidth=2)

            # Calculate and highlight the 95% credibility interval
            lower_bound, upper_bound = np.percentile(parameter_samples, [2.5, 97.5])
            plt.fill_between(x_range, 0, kernel_density(x_range), where=((x_range >= lower_bound) & (x_range <= upper_bound)), alpha=0.3, color='gray', label='95% C.I.')

            # Set axis labels and title
            plt.xlabel(['Overetch','Offset','Thickness'][i])
            plt.ylabel('Density')
            # Place the legend on the right side of the plot
            plt.legend()
            # Adjust the spacing between subplots
            plt.show()
        
    # Create DataFrame from the list of dictionaries
    data = pd.DataFrame(data_list)

    # At this point, data DataFrame contains all the inserted values
    print(data)
    data.to_csv('noise.csv', index=False)