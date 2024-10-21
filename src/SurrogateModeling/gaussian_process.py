import json
import numpy as np
from model import NN_Model, LSTM_Model
import sys 

# Custom module imports
sys.path.append('../utils/')

from preprocessing import preprocessing, LSTMDataProcessor
from postprocessing import plot_predictions

import logging
from IPython.display import display
import GPy

from sklearn.cluster import KMeans

def get_batch(data, batch_index, batch_size):
    """
    Returns a minibatch of the data.

    Parameters:
    data (numpy.ndarray): The dataset from which to extract the batch.
    batch_index (int): The index of the batch to retrieve.
    batch_size (int): The size of each batch.

    Returns:
    numpy.ndarray: The selected minibatch.
    """
    start_index = batch_index * batch_size
    end_index = start_index + batch_size
    batch = data[start_index:end_index]
    return batch


config_file = './config_gp.json'

# Load and process data
logging.info("\nLoading and processing data.\n")
# data_processor = LSTMDataProcessor(config_file)
data_processor = preprocessing(config_file)
config = data_processor.config

n_samples, n_features = data_processor.X_train_scaled.shape
print(n_samples, n_features)

data_processor.y_train_scaled = data_processor.y_train_scaled.reshape(n_samples,1)

kernel = GPy.kern.RBF(input_dim=n_features, 
                      variance=4000, ARD=True, 
                      lengthscale=[0.5,0.5,0.5,0.05])
model = GPy.models.SparseGPRegression(data_processor.X_train_scaled, 
                                      data_processor.y_train_scaled, 
                                      kernel,
                                      num_inducing=int(0.01*n_samples))
model.optimize('bfgs', messages=True)  # Optimize with the current batch
display(model)
display(model.rbf.lengthscale)

y_pred, y_var = model.predict_noiseless(data_processor.X_train_scaled[0:(160*150),:],full_cov=False)
y_pred=y_pred.reshape(data_processor.y_train[0:160].shape)
y_var=y_var.reshape(data_processor.y_train[0:160].shape)

time = np.arange(0, 150e-3, 1e-3)

import time

# Start the timer
start_time = time.time()

y_pred, y_var = model.predict_noiseless(data_processor.X_test_scaled[0:(150*160),:],full_cov=False)
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(elapsed_time)

y_pred=y_pred.reshape(data_processor.y_test[0:160].shape)
y_var=y_var.reshape(data_processor.y_test[0:160].shape)

import matplotlib.pyplot as plt

time = np.arange(0, 150e-3, 1e-3)


for i in range(5):
    plt.figure
    plt.plot(time, data_processor.y_test[i], label="Numerical data")
    plt.plot(time,y_pred[i], label="Mean Prediction")
    plt.fill_between(
        time,
        y_pred[i] - 1.96 * y_var[i],
        y_pred[i] + 1.96 * y_var[i],
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.show()