import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the directory containing the dataLoader module to the sys.path
sys.path.append('../DATA/')

# Import the load_data function from the dataLoader modulec
from dataLoader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define Input Parameters
sampling_frequency = 1e-3 / 1e-5
period = 2e-3
n_periods = 2
n_epochs = 5000
batch_size = 100
n_nr = 32
time = np.arange(0, 1.5e-3, 1e-5)
lr = 0.1
N = 250
Nb = 100
noise = 10

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/C_training_HalfSine.csv'

# Import tensorflow for forwar model
import tensorflow as tf
model_path = '../ML-Models/modelNo1_dV_latin.h5'
forward_model_tf = tf.keras.models.load_model(model_path)

# Load the data using the load_data function and Obtain Capacity variation
C_dataset_filename = '../DATA/CSV/C_training_HalfSine.csv'
C_df = load_data(C_dataset_filename)
C_df = C_df.dropna()
print(C_df)

dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=1.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=1.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=1.49ms'].values) / 1e-5

# Select the input and output columns
input_cols = ['Overetch', 'Offset', 'Thickness']
output_cols = dC_df.columns[5:-1]

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

# Get the number of samples
num_samples = X_train_rep.shape[0]

# Generate a random permutation of indices
perm = np.random.permutation(num_samples)

X_train_rep = X_train_rep[perm]
y_train_rep = y_train_rep[perm]

# Step 3: Preprocess the Data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_rep)
X_test_scaled = scaler.transform(X_test_rep)

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/C_training_HalfSine.csv'
# Load the data using the load_data function and Obtain Capacity variation
C_df = load_data(C_dataset_filename)
C_df = C_df.dropna()
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=1.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=1.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=1.49ms'].values) / 1e-5

dC_test=dC_df 
X_test=dC_test[input_cols].values
y_test=1e12 * dC_test[output_cols].values

X_test_rep = np.repeat(X_test, len(output_cols), axis=0)
X_test_rep = np.column_stack((X_test_rep, np.tile(time, len(X_test_rep) // len(time))))
y_test_rep = y_test.flatten()

X_test_scaled = scaler.transform(X_test_rep)

def forward_model(x):
    x = tf.convert_to_tensor(np.array(x),dtype=tf.float32)
    output = forward_model_tensorflow(x).numpy().flatten()
    return output

def forward_model_tensorflow(x):
    # # Convert input_tensor to a TensorFlow tensor
    # Define constant values as TensorFlow tensors
    concatenated_tensor = tf.stack([
        x[0],
        x[1],
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

for i in range(X_test.shape[0]):

    x_true = X_test[i,:]
    print(x_true)
    # tx  = Gaussian(mean=x_true[3],cov=1e-8)
    # etch = Uniform(low=0.3,high=0.5)

    y_obs = Gaussian(mean=y_test[i,:], cov=noise*np.eye(len(time))).sample()

    # print(y_obs)
    # jac = compute_jacobian(x_true[[0,3,6]])
    # print(jac.shape)
    stringa = "../Chains_dV/Ax"+str(x_true[0])+"_Tx"+str(x_true[1])+"_Etch"+str(x_true[2])

    plt.figure()
    plt.plot(time, y_test[i, :], c=real_color, label='Real', linewidth=line_width)
    plt.plot(time, y_obs, '.b', label='Noisy', linewidth=line_width)
    # Set axis labels
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\Delta C$ [fF/s]')
    # Set title with relevant information
    plt.title(
        f'Overetch = {X_test[i, 0]}μm; Offset = {X_test[i, 1]:.2f}μm; Thickness = {X_test[i, 2]:.2f}μm',
        fontsize=10, loc='center')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    x0 = np.array([0.5,-0.5,31.0])
    x1 = np.array([0.5,0.5,31.0])
    x2 = np.array([0.1,0.5,29.0])
    x3 = np.array([0.3,-0.5,30.0])
    x4 = np.array([0.1,0.5,31.0])
    x5 = np.array([0.5,1/5,29.0])

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
    # x = Gaussian(mean=np.array([x_true[0],x_true[0],0.4]),cov=np.array([1e-2,1e-10,1e-4]))
    # acc = Gaussian(mean=x_true[0],cov=0.1**2)
    y = Gaussian(mean=A(x), cov=noise*np.eye(len(time)))


    from cuqi.sampler import UGLA, Conjugate, ConjugateApprox, Gibbs

    # Define posterior distribution
    posterior = JointDistribution(x, y)(y=y_obs)
    
    MHsampler = MH(target=posterior,
                   proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1e-4,1e-4,1e-4])),
                   x0=x0
                #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
                   )
    samplesMH = MHsampler.sample_adapt(N,Nb)

    plt.figure()
    samplesMH.plot_trace(variable_indices=[0])
    plt.show()

    plt.figure()
    samplesMH.plot_trace(variable_indices=[1])
    plt.show()

    plt.figure()
    samplesMH.plot_trace(variable_indices=[2])
    plt.show()

    print(samplesMH.compute_ci())
    print(samplesMH.mean())


    # MHsampler2 = MH(target=posterior,
    #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1.0,1e-8,1e-4])),
    #                x0=x2
    #             #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
    #                )
    # samplesMH2 = MHsampler2.sample_adapt(N,Nb)

    # MHsampler3 = MH(target=posterior,
    #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1.0,1e-8,1e-4])),
    #                x0=x3
    #             #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
    #                )
    # samplesMH3 = MHsampler3.sample_adapt(N,Nb)

    # MHsampler4 = MH(target=posterior,
    #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1.0,1e-8,1e-4])),
    #                x0=x4
    #             #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
    #                )
    # samplesMH4 = MHsampler4.sample_adapt(N,Nb)

    # MHsampler5 = MH(target=posterior,
    #                proposal=Gaussian(mean=np.array([0.0,0.0,0.0]).flatten(),cov=np.array([1.0,1e-8,1e-4])),
    #                x0=x5
    #             #    proposal=Gaussian(mean=np.array([0.0]).flatten(),cov=np.array([1.0])),
    #                )
    # samplesMH5 = MHsampler5.sample_adapt(N,Nb)
    # a = samplesMH.compute_rhat([samplesMH2,samplesMH3,samplesMH4,samplesMH5])
    # print(a)

    # np.savetxt(stringa+'_1', samplesMH.samples,  fmt='%.4e')
    # np.savetxt(stringa+'_2', samplesMH2.samples, fmt='%.4e')
    # np.savetxt(stringa+'_3', samplesMH3.samples, fmt='%.4e')
    # np.savetxt(stringa+'_4', samplesMH4.samples, fmt='%.4e')
    # np.savetxt(stringa+'_5', samplesMH5.samples, fmt='%.4e')
    # Specify the path to the text file
    # file_path = stringa

    # # Use np.loadtxt() to read the text file into a NumPy array
    # samples1 = np.loadtxt(file_path+'_1')
    # samplesMH.samples = samples1
    # samplesMH.ns = samples1.shape[1]
    # samplesMH.burnthin(5000)

    # samples2 = np.loadtxt(file_path+'_2')
    # samplesMH2.samples = samples2
    # samplesMH2.ns = samples2.shape[1]
    # samplesMH2.burnthin(5000)

    # samples3 = np.loadtxt(file_path+'_3')
    # samplesMH3.samples = samples3
    # samplesMH3.ns = samples3.shape[1]
    # samplesMH3.burnthin(5000)

    # samples4 = np.loadtxt(file_path+'_4')
    # samplesMH4.samples = samples4
    # samplesMH4.ns = samples4.shape[1]
    # samplesMH4.burnthin(5000)

    # samples5 = np.loadtxt(file_path+'_5')
    # samplesMH5.samples = samples5
    # samplesMH5.ns = samples5.shape[1]
    # samplesMH5.burnthin(5000)

    # a = samplesMH.compute_rhat([samplesMH2,samplesMH3,samplesMH4,samplesMH5])
    # print(a)

    # # samplesMH=samplesMH.burnthin(0,4)
    # Ni = samplesMH.Ns

 

    # # # import numpy as np
    # # # import matplotlib.pyplot as plt
    from scipy import stats

    # num_parameters = 3

    # # Calculate the cumulative mean along the rows (axis 0)
    # cumulative_mean1 = np.cumsum(samplesMH.samples, axis=1) / np.arange(1, samplesMH.samples.shape[1] + 1)
    # cumulative_mean2 = np.cumsum(samplesMH2.samples, axis=1) / np.arange(1, samplesMH.samples.shape[1] + 1)
    # cumulative_mean3 = np.cumsum(samplesMH3.samples, axis=1) / np.arange(1, samplesMH.samples.shape[1] + 1)
    # cumulative_mean4 = np.cumsum(samplesMH4.samples, axis=1) / np.arange(1, samplesMH.samples.shape[1] + 1)
    # cumulative_mean5 = np.cumsum(samplesMH5.samples, axis=1) / np.arange(1, samplesMH.samples.shape[1] + 1)

    # # Create an array for the x-axis (step numbers)
    # steps = np.arange(1, samples1.shape[1] + 1)

    # plt.figure()

    # # Plot the cumulative mean for each parameter
    # for i in range(0,num_parameters,2):
    #     plt.subplot(2, 1, int(i/2)+1)
    #     plt.plot(steps, cumulative_mean1[i, :], color='b')
    #     plt.plot(steps, cumulative_mean2[i, :], color='green')
    #     plt.plot(steps, cumulative_mean3[i, :], color='orange')
    #     plt.plot(steps, cumulative_mean4[i, :], color='grey')
    #     plt.plot(steps, cumulative_mean5[i, :], color='purple')
    #     plt.axhline(x_true[3*i], color='red', label='True', linestyle='-', linewidth=2)
    #     plt.axhline(optimized_params[i], color='black', label='LS', linestyle='--', linewidth=2)
    #     plt.xlabel('Step Number')
    #     plt.ylabel('Mean Value')
    #     plt.legend()

    # plt.tight_layout()
    # plt.show()



    for i in range(0,3):
        parameter_samples = samplesMH.samples[i, :]
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
    num_samples = samplesMH.samples.shape[1]

    # # for i in range(3):
    # #     parameter_samples = samplesMH.samples[i, :]
    # #     plt.figure()

    # #     # Plot the chain of samples
    # #     plt.plot(parameter_samples, label='Chain', linewidth=1)

    # #     # Calculate and highlight the 95% credibility interval
    # #     lower_bound, upper_bound = np.percentile(parameter_samples, [2.5, 97.5])
    # #     plt.fill_between(np.arange(num_samples), lower_bound, upper_bound, alpha=0.3, color='gray', label='95% C.I.')

    # #     # Plot horizontal lines for true and optimized parameters
    # #     plt.axhline(x_true[3*i], color='red', label='True', linestyle='-', linewidth=2)
    # #     plt.axhline(optimized_params[i], color='black', label='LS', linestyle='--', linewidth=2)

    # #     # Set axis labels and title
    # #     plt.xlabel('Sample')
    # #     plt.ylabel(['Amplitude', 'Period', 'Overetch'][i])

    # #     # Place the legend on the right side of the plot
    # #     plt.legend()

    # #     # Show the plot
    # #     plt.show()

    # # Define the parameter names (customize these as needed)
    # parameter_names = ['Amplitude', 'Period', 'Overetch']

    # # Create a grid of subplots for each pair of parameters
    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    # import numpy as np

    # Assuming you have the samples and parameter_names defined

    # num_parameters = len(parameter_names)
    # samples = samplesMH.samples
    # num_samples = samples.shape[1]
    # step_size = 100  # Number of points to add in each frame
    # frames = range(0, num_samples, step_size)

    # from matplotlib.animation import FuncAnimation

    # x_min, x_max = 0, 100
    # y_min, y_max = 0.3, 0.5
    # print(x_true)
    # def update(frame):
    #     plt.clf()
    #     plt.figure()
    #     i = 0;
    #     j = 2
    #         #    if i == j:
    #         #         # Diagonal plots: Histograms
    #         #         plt.hist(samples[:frame+1, i], bins=30, color='b', alpha=0.7)
    #         #         plt.xlabel(parameter_names[i])
    #         #         plt.ylabel('Frequency')
    #             # else:
    #                 # Contour plots for parameter pairs
    #     # Set the axis limits for X and Y
    #     plt.xlim(x_min, x_max)
    #     plt.ylim(y_min, y_max)

    #     x = samples[i, frame:frame+1000]
    #     y = samples[j, frame:frame+1000]    
    #     plt.scatter(x, y, s=10, alpha=0.1)

    #     # x2 = samplesMH2.samples[i, frame:frame+1000]
    #     # y2 = samplesMH2.samples[j, frame:frame+1000]    
    #     # plt.scatter(x2, y2, s=10, color='green', alpha=0.1) 

    #     # x3 = samplesMH3.samples[i, frame:frame+1000]
    #     # y3 = samplesMH3.samples[j, frame:frame+1000]    
    #     # plt.scatter(x3, y3, s=10, color='orange', alpha=0.1) 

    #     # x4 = samplesMH4.samples[i, frame:frame+1000]
    #     # y4 = samplesMH4.samples[j, frame:frame+1000]    
    #     # plt.scatter(x4, y4, s=10, color='grey', alpha=0.1) 

    #     # x5 = samplesMH5.samples[i, frame:frame+1000]
    #     # y5 = samplesMH5.samples[j, frame:frame+1000]    
    #     # plt.scatter(x5, y5, s=10, color='purple', alpha=0.1) 

    #     plt.scatter(x_true[0],x_true[6], marker='x', color='red', s=10, alpha=1)

    #     # Define the X-axis tick positions and labels
    #     x_ticks = np.arange(0, 10, 1)  # Create ticks from 1 to 10 (inclusive)
    #     x_tick_labels = [f"{x}g" for x in x_ticks]  # Format tick labels as "1g," "2g," etc.
    #     x_ticks = np.arange(0, 100, 10)  # Create ticks from 1 to 10 (inclusive)

    #     # Set the X-axis ticks and labels
    #     plt.xticks(x_ticks, x_tick_labels)

    #     plt.xlabel("Amplitude")
    #     plt.ylabel("Overetch (μm)")
    #     plt.tight_layout()

    # ani = FuncAnimation(plt.gcf(), update, frames=frames, repeat=False)

    # for it, frame in enumerate(frames):
    #     update(frame)
    #     plt.savefig(f'frames/frameG_{it:04d}.png', dpi=100)
    #     plt.clf()

    # # To combine the frames into a video using FFmpeg:
    # # ffmpeg -i frames/frame_%04d.png -c:v libx264 -vf "fps=30" output.mp4

    # plt.show()

    # plt.figure()
    # Ni=N
    # samplesMH.plot_chain(variable_indices=[0])
    # plt.plot(range(N),x_true[0]*np.ones((Ni,1)), label='True Amplitude')
    # plt.plot(range(N),optimized_params[0]*np.ones((Ni,1)), label='LS Amplitude') 
    # # plt.plot(range(N),samplesMH.mean()[0]*np.ones((Ni,1)), label='Samples mean')
    # plt.legend(loc='upper right', fontsize=10)
    # plt.show()

    # plt.figure()
    # samplesMH.plot_chain(variable_indices=[1])
    # plt.plot(range(N),x_true[3]*np.ones((Ni,1)), label='True Period')
    # plt.plot(range(N),optimized_params[1]*np.ones((Ni,1)), label='LS Period')
    # # plt.plot(range(N),samplesMH.mean()[1]*np.ones((Ni,1)), label='Samples mean')
    # plt.legend(loc='upper right', fontsize=10)
    # plt.show()

    # plt.figure()
    # samplesMH.plot_chain(variable_indices=[2])
    # plt.plot(range(N),x_true[6]*np.ones((Ni,1)), label='True Overetch')
    # plt.plot(range(N),optimized_params[2]*np.ones((Ni,1)), label='LS Overetch')
    # # plt.plot(range(N),samplesMH.mean()[2]*np.ones((Ni,1)), label='Samples mean')
    # plt.legend(loc='upper right', fontsize=10)    
    # plt.show()
        
    # plt.figure()
    # samplesMH.plot_violin(variable_indices=[0])
    # plt.show()
    # plt.figure()
    # samplesMH.plot_violin(variable_indices=[1])
    # plt.show()
    # plt.figure()
    # samplesMH.plot_violin(variable_indices=[2])
    # plt.show()

    # plt.figure()
    # samplesMH.plot_trace(variable_indices=[0], exact=x_true[0])
    # plt.show()
    # plt.figure()
    # samplesMH.plot_trace(variable_indices=[1], exact=x_true[3])
    # plt.show()
    # plt.figure()
    # samplesMH.plot_trace(variable_indices=[2], exact=x_true[6])
    # plt.show()
