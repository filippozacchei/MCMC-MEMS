import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import sys
sys.path.append('./solver')
from random_process import * 
from fenics import *


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


def plot_fields(x_true, x_estimated, n_eig, lamb=0.1 , resolution = [50,50] , mean = 1, std_dev = 1, lognormal = False):

    mesh = UnitSquareMesh(resolution[0], resolution[1])
    field = RandomProcess(mesh, n_eig, lamb )
    field.compute_eigenpairs()
    field_reconstructed = RandomProcess(mesh , n_eig,lamb)
    field_reconstructed.compute_eigenpairs()

    fig = plt.figure(figsize=(10, 4))
    plt.subplots_adjust(hspace=0.5)
    title = 'True vs Reconstructed Random Field'
    fig.suptitle(title, fontsize=14)

    # Generate fields to find the global vmin and vmax
    field.generate(x_true)
    field_reconstructed.generate(x_estimated)
    if lognormal == True:
        field_true = np.exp(mean + std_dev*field.random_field)
        field_estimated = np.exp(mean + std_dev*field_reconstructed.random_field)
    else:
        field_true = mean + std_dev*field.random_field
        field_estimated = mean + std_dev*field_reconstructed.random_field

    global_min = min(min(field_true), min(field_estimated))
    global_max = max(max(field_true), max(field_estimated))
    contour_levels_field = np.linspace(global_min, global_max, 100)

    x = mesh.coordinates()[:,0]; y = mesh.coordinates()[:,1]
    # Plot field and solution.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (24, 9))
    
    axes[0].set_title('Transmissivity Field', fontdict = {'fontsize': 24})
    axes[0].tick_params(labelsize=16)
    f1 = axes[0].tricontourf(x, 
                            y, 
                            field_true, 
                            levels = contour_levels_field, 
                            cmap = 'plasma');  
    fig.colorbar(f1, ax=axes[0])
        
    axes[1].set_title('Reconstruction', fontdict = {'fontsize': 24})
    axes[1].tick_params(labelsize=16)
    f2 = axes[1].tricontourf(x, 
                            y, 
                            field_estimated, 
                            levels = contour_levels_field, 
                            cmap = 'plasma');  
    fig.colorbar(f2, ax=axes[1])


    plt.show()