import json
import numpy as np
import tensorflow as tf
from model import NN_Model
from dataLoader import DataProcessor
from plotting import plot_predictions
import logging

# CONFIGURATION FILE
CONFIGURATION_FILE = './Config_files/config_VoltageSignal_temp.json'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(config_file):
    try:
        # Load and process data
        logging.info("\nLoading and processing data.\n")
        config = DataProcessor.parse_config(config_file)
        data_processor = DataProcessor(config_file)
        data_processor.process()

        # Build model
        logging.info("\nBuilding the model.\n")
        model = NN_Model()
        model.build_model(data_processor.X_train_scaled.shape[1])

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
                          lr_schedule=learning_rate_schedule,
                          verbose=2)

        model.plot_training_history()

        # Plot predictions
        logging.info("\nPlotting predictions.\n")
        y_pred = model.predict(data_processor.X_test_scaled).reshape(data_processor.y_test.shape)
        time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
        plot_predictions(model, y_pred, data_processor.y_test, time, max_plots=5)

        # Save model
        logging.info("\nSaving the model.\n")
        model.save_model(config['MODEL_PATH'])

        return model

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

if __name__ == '__main__':
    logging.info("\nChecking GPU availability.\n")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train(CONFIGURATION_FILE)
