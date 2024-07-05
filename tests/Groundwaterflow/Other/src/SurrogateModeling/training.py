import json
import logging
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom module imports
sys.path.append('../utils/preprocessing')

from model import NN_Model
from preprocessing import preprocessing


def check_gpu_availability():
    """Logs the number of available GPUs."""
    logging.info("Checking GPU availability.")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

def train(config_file):
    """
    Trains a neural network surrogate model based on the provided configuration, 
    which is parsed from a json file.

    The configuration file must include these parameters:
    - "INPUT_COLS": the input parameters parsed from csv file.
    - "DATASET_PATH": the path where the csv file is stored.
    - "CONFIGURATION" : two alternatives:
        - "I" refers to active voltage response,
        - "II" refers to sensitivity.
    - "MODEL_PATH": where to save the surrogate model.

    Parameters:
    - config_file: str, path to the JSON configuration file.    

    Raises:
    - Exception: Propagates any exceptions that occur during training.

    """
    
    check_gpu_availability()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load and process data
        logging.info("Loading and processing data.")

        data_processor = preprocessing(config_file)
        config = data_processor.config

        # Build and train model
        logging.info("Building the model.")
        model = NN_Model()
        model.build_model(input_shape=(data_processor.X_train_scaled.shape[1]),
                          n_neurons=[config["N_NEURONS"]] * config["N_LAYERS"], 
                          activation=config["ACTIVATION"])

        # Learning rate schedule
        def learning_rate_schedule(epoch):
            return config['LEARNING_RATE'] / (1.0 + epoch * config['DECAY_RATE'])

        logging.info("Training the model.")

        model.train_model(X=data_processor.X_train_scaled,
                          y=data_processor.y_train_scaled,
                          X_val=data_processor.X_test_scaled,
                          y_val=data_processor.y_test_scaled,
                          epochs=config['N_EPOCHS'],
                          batch_size=config['BATCH_SIZE'],
                          lr_schedule=learning_rate_schedule,
                          validation_freq=100,
                          verbose=2,
                          plot_loss=config['PLOT_TRAIN'])
        
        # Save, load, and plot model
        logging.info("Saving and reloading the model for inference.")

        model.save_model(config['MODEL_PATH'])

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


