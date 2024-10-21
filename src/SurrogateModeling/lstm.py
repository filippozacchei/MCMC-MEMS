import json
import numpy as np
import tensorflow as tf
from model import NN_Model, LSTM_Model
import logging
import sys
import matplotlib.pyplot as plt

# Custom module imports
sys.path.append('../utils/')

from preprocessing import preprocessing, LSTMDataProcessor
from postprocessing import plot_predictions

# CONFIGURATION FILE
# CONFIGURATION_FILE = './Config_files/config_VoltageSignal_lstm.json'
CONFIGURATION_FILE = './config_I.json'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(config_file):
    try:
        # Load and process data
        logging.info("\nLoading and processing data.\n")
        data_processor = LSTMDataProcessor(config_file)
        # data_processor = preprocessing(config_file)
        config = data_processor.config

        # Build model
        logging.info("\nBuilding the model.\n")
        model = LSTM_Model()
        # model = NN_Model()

        n_samples, n_timeSteps, n_features = 800, 150, 5

        model.build_model(input_shape=(n_timeSteps,n_features))
        # model.build_model(input_shape=(data_processor.X_train_scaled.shape[1]))
        print(data_processor.X_train_scaled.shape)
        print(data_processor.X_test_scaled.shape)
        print(data_processor.y_train_scaled.shape)
        print(data_processor.y_test_scaled.shape)

        lr = config['LEARNING_RATE']
        decay_rate=config['DECAY_RATE']
        
        # Define learning rate scheduler
        def learning_rate_schedule(epoch):
            return lr / (1.0 + epoch * decay_rate)

        # Train the model
        logging.info("\nTraining the model.\n")
        model.train_model(X=data_processor.X_train_scaled,
                          y=data_processor.y_train_scaled,
                          X_val=data_processor.X_test_scaled,
                          y_val=data_processor.y_test_scaled,
                          epochs=config['N_EPOCHS'],
                          batch_size=config['BATCH_SIZE'],
                          lr_schedule=learning_rate_schedule,
                          verbose=2)

        # model.plot_training_history()

        # Plot predictions
        logging.info("\nPlotting predictions.\n")
        
        import time

        # Assuming X_test is defined and model is your model object

        # Start the timer
        start_time = time.time()

        # Perform the prediction operation
        y_pred = model.model(data_processor.X_test_scaled).numpy().reshape(data_processor.y_test.shape)

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(elapsed_time)
        
        time = np.arange(0, 150e-3, 1e-3)
        
        # for i in range(min(y_pred.shape[0], 5)):
        #     plt.figure()
        #     plt.plot(time, data_processor.y_test[i, :], label='Actual')
        #     plt.scatter(time, y_pred[i, :], label='Predicted')
        #     plt.legend()
        #     plt.show()

        # # Save model
        # logging.info("\nSaving the model.\n")
        # model.save_model(config['MODEL_PATH'])

        # return model

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

if __name__ == '__main__':
    logging.info("\nChecking GPU availability.\n")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train(CONFIGURATION_FILE)