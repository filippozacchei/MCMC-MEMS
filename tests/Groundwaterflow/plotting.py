import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import sys
sys.path.append('./solver')
from random_process_to_plot import * 


def plot_parameter_distributions(data, x_true, parameter_names, n_eig):
    fig = plt.figure(figsize=(12,40))
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(hspace=0.6, top=0.97)
    title = 'parameters samples'
    fig.suptitle(title, fontsize=14)
    for j in range(n_eig):
        parameter_samples = data[:,j]
        ax = fig.add_subplot(16,4, j+1)
        kernel_density = gaussian_kde(parameter_samples)
        x_range = np.linspace(np.min(parameter_samples), np.max(parameter_samples), 1000)
        plt.plot(x_range, kernel_density(x_range), label='Density', linewidth=2)
        plt.axvline(x_true[j], color='red', label='Exact', linestyle='-', linewidth=2)
        mean, mode = np.mean(parameter_samples), x_range[np.argmax(kernel_density(x_range))]
        plt.axvline(mean, color='green', label='Mean', linestyle='--', linewidth=2)
        plt.axvline(mode, color='blue', label='Mode', linestyle='--', linewidth=2)
        lower_bound, upper_bound = np.percentile(parameter_samples, [2.5, 97.5])
        plt.fill_between(x_range, 0, kernel_density(x_range), where=((x_range >= lower_bound) & (x_range <= upper_bound)), alpha=0.3, color='gray', label='95% C.I.')
        plt.ylabel('Density', fontsize=5)
        plt.legend(fontsize=5)
        ax.title.set_text(parameter_names[j])
    plt.show()



def plot_results( y_true, y_obs, model, samplesMH ,n_eig, x_true = None, REAL_COLOR='red', LINE_WIDTH=1.5):

    # Define the sampling points.
    x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # Grid to plot the basis 
    X,Y = np.meshgrid(x_data, y_data)

    fig = plt.figure(figsize=(12,4))
    plt.subplots_adjust(hspace=0.8)
    title = 'Comparison of true, noisy, and reconstructed output'
    fig.suptitle(title, fontsize=14)
    
    ax = fig.add_subplot(131 + 0)
    pcm = plt.pcolormesh(X, Y, y_true.reshape((5, 5)))
    ax.title.set_text('True signal sample: ')
    plt.colorbar(pcm, ax=ax)
    
    ax = fig.add_subplot(131 + 1)
    pcm = plt.pcolormesh(X, Y, y_obs.reshape((5, 5)) )
    ax.title.set_text('Noisy signal: ')
    plt.colorbar(pcm, ax=ax)
    
    ax = fig.add_subplot(131 + 2)
    estimated_parameters = np.mean(samplesMH, axis = 0)
    reconstructed_sample = model(estimated_parameters.reshape(1,n_eig))
    pcm = plt.pcolormesh(X, Y,reconstructed_sample.reshape((5, 5)))
    ax.title.set_text('Signal with estimated parameters' )
    plt.colorbar(pcm, ax=ax)

    plt.figure()
    plt.plot(range(25), y_true, c=REAL_COLOR, label='Real', linewidth=LINE_WIDTH)
    plt.plot(range(25), reconstructed_sample, 'green', label='Pred', linewidth=LINE_WIDTH)
    plt.plot(range(25), y_obs, '.-b', label='Noisy Signal', linewidth=LINE_WIDTH)
    if (x_true is not None):
        plt.plot(range(25), model(x_true),label='model+true param', linewidth=LINE_WIDTH)
    plt.xlabel('sensor number')
    plt.ylabel('Hydraulic pressure')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    plt.figure()
    plt.plot(range(25), y_obs-reconstructed_sample, c=REAL_COLOR, label='obs-reconst', linewidth=LINE_WIDTH)
    plt.plot(range(25), y_obs-model(x_true), '.-b', label='obs-model(true param)', linewidth=LINE_WIDTH)
    plt.xlabel('sensor number')
    plt.ylabel('Hydraulic pressure')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()



def plot_fields(x_true, x_estimated, n_eig, lognormal = False):
    field = RandomProcess(n_eig)
    field.compute_eigenpairs()
    field_reconstructed = RandomProcess(n_eig)
    field_reconstructed.compute_eigenpairs()

    fig = plt.figure(figsize=(10, 4))
    plt.subplots_adjust(hspace=0.5)
    title = 'True vs Reconstructed Random Field'
    fig.suptitle(title, fontsize=14)

    # Generate fields to find the global vmin and vmax
    field.generate(x_true)
    field_true = field.random_field
    field_reconstructed.generate(x_estimated)
    field_estimated = field_reconstructed.random_field

    vmin = min(field_true.min(), field_estimated.min())
    vmax = max(field_true.max(), field_estimated.max())

    ax = fig.add_subplot(121)
    field.plot(vmin=vmin, vmax=vmax, lognormal = lognormal)
    ax.title.set_text('True Transmissivity field: ')

    ax = fig.add_subplot(122)
    field_reconstructed.plot(vmin=vmin, vmax=vmax, lognormal = lognormal)
    ax.title.set_text('Reconstructed field: ')

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