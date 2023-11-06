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

def plot_predictions(model, X_test, y_test, time_steps, max_plots=5):
    """
    Evaluates the model on test data and plots the predictions against the true values.

    Parameters:
    - model: The trained Keras model to evaluate.
    - X_test (array-like): Test features.
    - y_test (array-like): True values for the test features.
    - time_steps (array-like): The time steps for each feature in X_test.
    - max_plots (int): The maximum number of plots to display.
    """
    y_pred = model.predict(X_test).reshape(y_test.shape)
    for i in range(min(len(X_test), max_plots)):
        plt.figure()
        plt.plot(time_steps, y_test[i, :], label='Actual')
        plt.plot(time_steps, y_pred[i, :], label='Predicted')
        plt.legend()
        plt.title(f'Test Sample {i+1}')
        plt.show()