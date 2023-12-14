import json
import numpy as np
from model import NN_Model, LSTM_Model
from dataLoader import DataProcessor, LSTMDataProcessor
from plotting import plot_predictions
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


config_file = './Config_files/config_VoltageSignal_temp.json'

# Load and process data
logging.info("\nLoading and processing data.\n")
config = DataProcessor.parse_config(config_file)
# data_processor = LSTMDataProcessor(config_file)
data_processor = DataProcessor(config_file)
data_processor.process()

n_samples, n_features = data_processor.X_train_scaled.shape
print(n_samples, n_features)

data_processor.y_train_scaled = data_processor.y_train_scaled.reshape(n_samples,1)

kernel = GPy.kern.RBF(input_dim=n_features, 
                      variance=4000, ARD=True, 
                      lengthscale=[0.5,0.5,0.5,0.05])
model = GPy.models.SparseGPRegression(data_processor.X_train_scaled, 
                                      data_processor.y_train_scaled, 
                                      kernel,
                                      num_inducing=int(0.1*n_samples))
# model.optimize('bfgs', messages=True)  # Optimize with the current batch
display(model)
display(model.rbf.lengthscale)

# model.save_model('./file')

y_pred, y_var = model.predict_noiseless(data_processor.X_train_scaled[0:1500,:],full_cov=False)
y_pred=y_pred.reshape(data_processor.y_train[0:10].shape)
y_var=y_var.reshape(data_processor.y_train[0:10].shape)

time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])

import matplotlib.pyplot as plt
for i in range(y_pred.shape[0]):
    plt.figure
    plt.plot(time,data_processor.y_train[i], label="Numerical data")
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

y_pred, y_var = model.predict_noiseless(data_processor.X_test_scaled,full_cov=False)
y_pred=y_pred.reshape(data_processor.y_test.shape)
y_var=y_var.reshape(data_processor.y_test.shape)

for i in range(y_pred.shape[0]):
    plt.figure
    plt.plot(time,data_processor.y_test[i], label="Numerical data")
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
