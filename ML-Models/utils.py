import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

sys.path.append('../DATA/')

def load_data(filename):
    # Read the CSV files
    df = pd.read_csv(filename, dtype=str, comment='#', float_precision='high')
    df = df.astype(np.float64)

    # Assign column labels
    column_labels = ['Overetch', 'Offset', 'Thickness', 'ts', 'tf']
    df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(151)]
    
    return df

# Load and process data
def load_and_process_data(config_file):
    df = load_data(config_file['DATASET_PATH'])
    d_df = df.copy()
    d_df.loc[:, config_file['TIME_COL_START']:config_file['TIME_COL_END']] = (
        df.loc[:, config_file['TIME_COL_SECOND']:config_file['TIME_COL_SECOND_END']].values -
        df.loc[:, config_file['TIME_COL_START']:config_file['TIME_COL_END']].values) / 1e-5
    return d_df

# Insert time steps to data
def flatten_and_expand_data(X, y, output_cols, time):
    X_rep = np.repeat(X, len(output_cols), axis=0)
    X_rep = np.column_stack((X_rep, np.tile(time, len(X_rep) // len(time))))
    y_rep = y.flatten()
    perm = np.random.permutation(len(X_rep))
    return X_rep[perm], y_rep[perm]

# Define the model structure
def build_model(input_shape, n_neurons=64, n_layers=6, activation_func='tanh'):
    model = Sequential([
        Dense(n_neurons, activation=activation_func, input_shape=(input_shape,)),
        *[Dense(n_neurons, activation=activation_func) for _ in range(n_layers)],
        Dense(1)
    ])
    return model

# Check that training and testing error are decreasing
def plot_training_history(history):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b', label='Training Error')
    plt.plot(range(1, len(train_losses) + 1), val_losses, 'r', label='Testing Error')
    plt.yscale('log')
    plt.legend()
    plt.show()