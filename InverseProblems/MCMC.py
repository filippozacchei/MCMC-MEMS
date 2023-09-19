import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the directory containing the dataLoader module to the sys.path
sys.path.append('./DATA/')

# Import the load_data function from the dataLoader modulec
from dataLoader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# Define Input Parameters
sampling_frequency = 1e-3 / 1e-5
period = 2e-3
n_periods = 2
n_epochs = 5000
batch_size = 100
n_nr = 32
time = np.arange(0, 2.5e-3, 1e-5)
lr = 0.01
N = 50000
Nb = 0
noise = 0.1**2

# Define the file paths for the datasets
C_dataset_filename = './DATA/CSV/CapacityDataset.csv'
U_dataset_filename = './DATA/CSV/UDataset.csv'
V_dataset_filename = './DATA/CSV/VDataset.csv'

# Import tensorflow for forwar model
import tensorflow as tf
model_path = './ML-Models/modelNo1_latin.h5'
forward_model_tf = tf.keras.models.load_model(model_path)

# Load the data using the load_data function and Obtain Capacity variation
C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=2.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=2.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=2.49ms'].values) / 1e-5

# Select the input and output columns
input_cols = ['AmplitudeX', 'AmplitudeY', 'AmplitudeZ', 'T_X', 'T_Y', 'T_Z', 'Overetch']
output_cols = dC_df.columns[7:-1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dC_df[input_cols].values, 
                                                    1e12 * (dC_df[output_cols].values), 
                                                    test_size=0.2, 
                                                    random_state=2)

# Flatten the output and concatenate time column
X_train_rep = np.repeat(X_train, len(output_cols), axis=0)
X_train_rep = np.column_stack((X_train_rep, np.tile(time, len(X_train_rep) // len(time))))
y_train_rep = y_train.flatten()

X_test_rep = np.repeat(X_test, len(output_cols), axis=0)
X_test_rep = np.column_stack((X_test_rep, np.tile(time, len(X_test_rep) // len(time))))
y_test_rep = y_test.flatten()

# Step 3: Preprocess the Data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_rep)
X_test_scaled = scaler.transform(X_test_rep)

def forward_model(x):
    x = tf.convert_to_tensor(x,dtype=tf.float32)
    output = forward_model_tensorflow(x).numpy().flatten()
    return output

def forward_model_tensorflow(x):
    # # Convert input_tensor to a TensorFlow tensor
    # Define constant values as TensorFlow tensors
    constant_values = tf.constant([50.0, 50.0, 1e-3, 1e-3], dtype=tf.float32)
    # Stack the tensors along axis 0 to achieve the desired operation
    concatenated_tensor = tf.stack([
        x[0],
        constant_values[0],
        constant_values[1],
        x[1],
        constant_values[2],
        constant_values[3],
        x[2],
    ])

    # Reshape and repeat to match your params_rep construction
    concatenated_tensor = tf.reshape(concatenated_tensor, (1, -1))
    concatenated_tensor = tf.tile(concatenated_tensor, (len(output_cols), 1))
    # Tile the time vector and stack it with params_rep
    time_tensor = tf.convert_to_tensor(time, dtype=tf.float32)
    time_tensor = tf.reshape(time_tensor,(len(output_cols),1))
    params_rep = tf.concat([concatenated_tensor, time_tensor], axis=1)
    # Use the scaler with TensorFlow operations
    # Manually scale concatenated_tensor using data_min and data_range from scaler
    data_min = tf.convert_to_tensor(scaler.data_min_, dtype=tf.float32)
    data_range = tf.convert_to_tensor(scaler.data_range_, dtype=tf.float32)
    scaled_concatenated_tensor = (params_rep - data_min) / data_range
    output = forward_model_tf(scaled_concatenated_tensor)
    return output

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
from cuqi.model import Model
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import MH, NUTS
from cuqi.geometry import Continuous1D, Discrete

A = Model(forward=forward_model,
          jacobian=compute_jacobian,
          range_geometry=Continuous1D(len(time)),
          domain_geometry=Discrete(['Ax','Tx','Etch']))

# Define Bayesian Problem


# print(compute_jacobian(x.sample()))
# Set figure size and resolution
plt.figure(figsize=(8,6))

# Define colors and line styles
real_color = 'red'
surrogate_color = 'blue'
line_width = 1.5

from scipy.optimize import least_squares, curve_fit
from scipy.stats import norm

for i in range(4,X_test.shape[0]):

    x_true = X_test[i,:]
    x = Uniform(low=np.array([0.0,0.5e-3,0.3]),high=np.array([100.0,1.5e-3,0.5]))
    y = Gaussian(mean=A(x),cov=noise)

    y_obs = y(x=np.array(x_true[[0,3,6]])).sample()
    print(y_obs.shape)

    # print(y_obs)
    # jac = compute_jacobian(x_true[[0,3,6]])
    # print(jac.shape)
    # stringa = "./Chains/Ax"+str(x_true[0])+"_Tx"+str(x_true[3])+"_Etch"+str(x_true[6])



    plt.plot(time, y_test[i, :], c=real_color, label='Real', linewidth=line_width)
    plt.plot(time, y_obs, c=surrogate_color, label='Surrogate', linewidth=line_width)
    # Set axis labels
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\Delta C$ [fF/s]')
    plt.title(
        'Ax = {}; Ay = {}; Az = {}; Tx = {:.2f}, Ty = {:.2f}, Tz = {:.2f}, Overetch = {:.2f}'.format(
                int(X_test[i, 0]),
                int(X_test[i, 1]), 
                int(X_test[i, 2]), 
                1e3*X_test[i, 3], 
                1e3*X_test[i, 4], 
                1e3*X_test[i, 5], 
                X_test[i, 6]),
                fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # # # Perform least squares optimization
    # # result = least_squares(objective_function,
    # #                        initial_params,
    # #                        args=([y_obs]), 
    # #                        bounds=([0.,
    # #                                 0.0005,
    # #                                 0.3],
    # #                                [100.,
    # #                                0.0015,
    # #                                0.5]), 
    # #                        jac='3-point')

    # # optimized_params = result.x

    # # print(optimized_params)

    # Define posterior distribution
    posterior = JointDistribution(x, y)(y=y_obs)
    x0 = np.array([100.0,1.5e-3,0.5])
    MHsampler = MH(target=posterior,
                   x0=x0,
                   proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1.0,1e-8,1e-4])),
                #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
                   dim=3
                   )
    samplesMH = MHsampler.sample_adapt(N,Nb)

    # np.savetxt(stringa, samplesMH.samples, fmt='%.4e')

    plt.figure()
    samplesMH.plot_chain(variable_indices=[0])
    plt.plot(range(N),x_true[0]*np.ones((N,1)), label='True Amplitude')
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    plt.figure()
    samplesMH.plot_chain(variable_indices=[1])
    plt.plot(range(N),x_true[3]*np.ones((N,1)), label='True Period')
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    plt.figure()
    samplesMH.plot_chain(variable_indices=[2])
    plt.plot(range(N),x_true[6]*np.ones((N,1)), label='True Overetch')
    plt.legend(loc='upper right', fontsize=10)    
    plt.show()
