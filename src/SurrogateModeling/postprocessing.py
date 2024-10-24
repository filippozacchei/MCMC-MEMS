import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable

import itertools
from itertools import product

import numpy as np

def plot_dataset(train, test, features_labels=None, features_ticks=None, features_ticks_labels=None, digits='%.2f', projection='3d'):
    """
    Creates a 3D scatter plot of the training and test sets.

    Parameters:
    - X_train: Training features.
    - X_test: Test features.
    """
    # Create a scatter plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection=projection)
    ax.scatter(train[:, 0], train[:, 1], train[:, 2], c='red',  label='Training Set', alpha=0.5)
    ax.scatter(test[:, 0],  test[:, 1],  test[:, 2],  c='blue', label='Test Set',     alpha=0.5)

    if features_labels is not None:
        # Set labels and title
        ax.set_xlabel(features_labels[0], fontsize=14)
        ax.set_ylabel(features_labels[1], fontsize=14)
        ax.set_zlabel(features_labels[2], fontsize=14)
    
    if features_ticks is not None:
        # Format tick labels to two decimal places
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(digits))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(digits))
        ax.zaxis.set_major_formatter(ticker.FormatStrFormatter(digits))
        
        ax.set_xticks(features_ticks[0])
        ax.set_yticks(features_ticks[1])
        ax.set_zticks(features_ticks[2])

    if features_ticks_labels is not None:
        # Set tick values
        ax.set_xticklabels(features_ticks_labels[0], fontsize=10)
        ax.set_yticklabels(features_ticks_labels[1], fontsize=10)
        ax.set_zticklabels(features_ticks_labels[2], fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    # Display the plot
    plt.show()
    return 

def plot_sensitivity_dataset(X_train, X_test, y_train, y_test, indices, mean_sensitivity,
                             labels, ticks, ticks_labels, digits='%.2f', projection='3d'):
    # Prepare the dataset for plotting
    X_train_selected = X_train[:, indices[:-1]]
    X_test_selected = X_test[:, indices[:-1]]
    y_train_scaled = y_train / mean_sensitivity
    y_test_scaled = y_test / mean_sensitivity
    
    # Stack the selected features and the scaled sensitivity values
    train_data = np.hstack([X_train_selected, y_train_scaled[:, np.newaxis]])
    test_data = np.hstack([X_test_selected, y_test_scaled[:, np.newaxis]])
    
    # Select the labels, ticks, and tick labels based on the provided indices
    features_labels = [labels[i] for i in indices]
    features_ticks = [ticks[i] for i in indices]
    features_ticks_labels = [ticks_labels[i] for i in indices]
    
    # Plot the dataset
    plot_dataset(train_data, test_data,
                 features_labels=features_labels, 
                 features_ticks=features_ticks, 
                 features_ticks_labels=features_ticks_labels, 
                 digits=digits, projection=projection)

def plot_predictions(model, y_test, X_test, time_steps, max_plots=5):
    """
    Evaluates the model on test data and plots the predictions against the true values.

    Parameters:
    - model: The trained Keras model to evaluate.
    - y_test (array-like): True values for the test features.
    - time_steps (array-like): The time steps for each feature in X_test.
    - max_plots (int): The maximum number of plots to display.
    """
    y_pred = model.predict(X_test).reshape(y_test.shape)
    for i in range(min(y_pred.shape[0], max_plots)):
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.plot(time_steps, y_test[i, :], 'b-', label='Data', linewidth=2)
        plt.scatter(time_steps, y_pred[i, :], color='r', label='Prediction', edgecolors='w', s=50)  # Red scatter with white edge
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel(f'$\Delta$C [fF]', fontsize=14)
        plt.legend(fontsize=12)
        plt.title(f'Overetch: 0.0-{abs(X_test[i,0]-0.3)/0.4:.2f}$\sigma_O$, Offset: 0.0+{X_test[i,1]:.2f}$\sigma_U$, Thickness: 30.0+{(X_test[i,2]-30.0)/2:.2f}$\sigma_T$', fontsize=16)
        plt.grid(True)
        plt.show()

def plot_correlation_sensitivity(model, X_train, X_test, y_train, y_test):
    """
    Scatter plot representing the discrepancy between reference data and predictions from the
    surrogate model.
    """
    y_pred_train = model.predict(X_train).reshape(y_train.shape)
    y_pred_test = model.predict(X_test).reshape(y_test.shape)

    plt.figure(figsize=(8, 6))  # Set a larger figure size for better readability

    plt.scatter(y_train, y_pred_train, color='b', label='Training', edgecolor='w', s=50)
    plt.scatter(y_test, y_pred_test, color='r', label='Validation', edgecolor='w', s=50)
    plt.plot(y_train, y_train, 'k--', label='Data')

    plt.xlabel('S reference', fontsize=14)
    plt.ylabel('S predicted', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()