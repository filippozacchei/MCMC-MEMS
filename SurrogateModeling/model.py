import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape, n_neurons=64, n_layers=6, activation_func='tanh', initializer='glorot_uniform'):
    """
    Constructs a sequential neural network model.

    Parameters:
    - input_shape (int): The shape of the input data.
    - n_neurons (int): Number of neurons in each layer.
    - n_layers (int): Number of hidden layers in the model.
    - activation_func (str): Activation function for the hidden layers.
    - initializer: Initializer to use for the weights.

    Returns:
    - model (Sequential): A compiled Keras sequential model.
    """
    model = Sequential()
    model.add(Dense(n_neurons, input_shape=(input_shape,), kernel_initializer=initializer))
    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer))
    model.add(Dense(1))  # Output layer
    return model


def plot_training_history(history):
    """
    Plots the training history of the model.

    Parameters:
    - history: A Keras History object returned by the fit function.
    """
    plt.figure()
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def save_model(model, model_path):
    """
    Saves the Keras model to a specified path.

    Parameters:
    - model: A Keras model object.
    - model_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)


