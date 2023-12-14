import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_train_test(X_train, X_test, feature_labels=None, features_ticks=None, digits='%.2f'):
    """
    Creates a 3D scatter plot of the training and test sets.

    Parameters:
    - X_train: Training features.
    - X_test: Test features.
    """
    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='red',  label='Training Set', alpha=0.5)
    ax.scatter(X_test[:, 0],  X_test[:, 1],  X_test[:, 2],  c='blue', label='Test Set',     alpha=0.5)

    if len(feature_labels)==3:
        # Set labels and title
        ax.set_xlabel(feature_labels[0], fontsize=12)
        ax.set_ylabel(feature_labels[1], fontsize=12)
        ax.set_zlabel(feature_labels[2], fontsize=12)
    
    if len(features_ticks)==3:
        # Format tick labels to two decimal places
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(digits))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(digits))
        ax.zaxis.set_major_formatter(ticker.FormatStrFormatter(digits))

        # Set tick values
        ax.set_xticks(features_ticks[0])
        ax.set_yticks(features_ticks[1])
        ax.set_zticks(features_ticks[2])

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    # Display the plot
    plt.show()
    return 

def plot_predictions(model, y_pred, y_test, X_test, time_steps, max_plots=5):
    """
    Evaluates the model on test data and plots the predictions against the true values.

    Parameters:
    - model: The trained Keras model to evaluate.
    - y_pred (array-like): Predicted values for the test features.
    - y_test (array-like): True values for the test features.
    - time_steps (array-like): The time steps for each feature in X_test.
    - max_plots (int): The maximum number of plots to display.
    """
    for i in range(min(y_pred.shape[0], max_plots)):
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.plot(time_steps, y_test[i, :], 'b-', label='Data', linewidth=2)
        plt.scatter(time_steps, y_pred[i, :], color='r', label='Prediction', edgecolors='w', s=50)  # Red scatter with white edge
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel(f'$\Delta$C [fF]', fontsize=14)
        plt.legend(fontsize=12)
        plt.title(f'Overetch: {X_test[i,0]:.2f}um, Offset: {X_test[i,1]:.2f}um, Thickness: {X_test[i,2]:.2f}um', fontsize=16)
        plt.grid(True)
        plt.show()

def plot_error_heatmap(errors, x_ticks=None, y_ticks=None, x_label='', y_label='', title='Relative Error', cmap='viridis', digits='.2f'):
    """Plots a heatmap for the given error matrix.
    
    Parameters:
        errors (np.ndarray): A 2D array of errors to plot in the heatmap.
        x_ticks (list): A list of tick values for the x-axis.
        y_ticks (list): A list of tick values for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title for the heatmap.
        cmap (str): Colormap to use for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(errors, cmap=cmap)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(x_ticks)) if x_ticks is not None else [])
    ax.set_yticks(np.arange(len(y_ticks)) if y_ticks is not None else [])
    ax.set_xticklabels([f'{tick:{digits}}' for tick in x_ticks] if x_ticks is not None else [])
    ax.set_yticklabels([f'{tick:{digits}}' for tick in y_ticks] if y_ticks is not None else [])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

def calculate_error_heatmap(y_pred, y_test, x_ticks, y_ticks, ord=np.inf):
    """Calculates and returns a matrix of errors between predictions and actual test values.
    
    Parameters:
        y_pred (np.ndarray): Predicted values.
        y_test (np.ndarray): Actual test values.
        x_ticks (list): List of tick values corresponding to columns of the error matrix.
        y_ticks (list): List of tick values corresponding to rows of the error matrix.

    Returns:
        np.ndarray: A 2D array of calculated errors.
    """
    errors = np.zeros((len(y_ticks), len(x_ticks)))
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        err = np.linalg.norm(pred - true, ord=ord) / np.linalg.norm(true, ord=ord)
        row, col = divmod(i, len(x_ticks))
        errors[row, col] = err
    return errors