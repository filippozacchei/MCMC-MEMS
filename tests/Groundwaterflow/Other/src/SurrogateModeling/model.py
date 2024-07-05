import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import List, Optional, Callable, Tuple
import numpy as np
from tensorflow.keras import losses

# Retrieve the loss function object based on the string identifier
from scipy.optimize import minimize


class NN_Model:
    """
    A class to build, train, and manage a neural network model using TensorFlow and Keras.
    """

    def __init__(self):
        """
        Initializes the NN_Model class with an empty Sequential model and a None history.
        """
        self.model: Sequential = Sequential()
        self.history = None

    def load_model(self, model_path: str) -> None:
        """
        Loads a model saved at the specified path.

        :param model_path: The file path to the saved Keras model.
        """
        self.model = load_model(model_path)

    def build_model(self,  
                    input_shape: (int), 
                    n_neurons: List[int] = [64, 64, 64, 64, 64, 64, 64, 64], 
                    activation: str = 'tanh',
                    output_neurons: int = 1,
                    output_activation: str = 'linear',
                    initializer: str = 'glorot_uniform') -> None:
        """
        Constructs the neural network model layer by layer.

        :param input_shape: The shape of the input layer.
        :param n_neurons: A list containing the number of neurons for each layer.
        :param activations: A list of activation functions for each layer.
        :param output_neurons: The number of neurons in the output layer.
        :param output_activation: The activation function for the output layer.
        :param initializer: The initializer for the layer weights.

        :raises ValueError: If the length of `n_neurons` and `activations` lists do not match.
        """

        self.model.add(Dense(n_neurons[0], activation=activation, input_shape=(input_shape,), kernel_initializer=initializer, kernel_regularizer=l2(1e-9)))        
        for neurons in n_neurons[1:]:
            self.model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(1e-9)))
        self.model.add(Dense(output_neurons, activation=output_activation, kernel_regularizer=l2(1e-9)))

    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 100, 
                    verbose: int = 0,
                    lr_schedule: Optional[Callable[[int], float]] = None,
                    plot_loss: bool = False) -> None:
        """
        Trains the model on the provided dataset.

        :param X: Input data for training.
        :param y: Target data for training.
        :param X_val: Input data for validation.
        :param y_val: Target data for validation.
        :param learning_rate: Initial learning rate for training.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Number of samples per batch of computation.
        :param loss: Loss function to be used during training.
        :param validation_freq: Frequency (number of epochs) at which to evaluate the validation data.
        :param lr_schedule: Optional learning rate schedule function.

        :raises ValueError: If any input arrays are empty.
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0:
            raise ValueError("Input arrays must not be empty")

        self.model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))
        
        callbacks = []
        if lr_schedule is not None:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

        self.history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=(X_val, y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

        self.plot_training_history()

    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.

        Parameters:
        - save_path: Optional; file path to save the plot.

        This method should be called after training the model using `train_model`.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()

    def save_model(self, model_path: str) -> None:
        """
        Saves the current state of the model to a specified file path.

        :param model_path: The file path where the model should be saved.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model.

        :param X: Input data for making predictions.
        :return: The predicted values.
        :raises ValueError: If the input array is empty.
        """

        return self.model.predict(X, verbose=0)
