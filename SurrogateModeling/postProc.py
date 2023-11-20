import json
import numpy as np
import tensorflow as tf
import logging
from model import NN_Model
from dataLoader import DataProcessor
from plotting import plot_predictions, calculate_error_heatmap, plot_error_heatmap

# Constants
CONFIGURATION_FILE = './Config_files/config_VoltageSignal_temp.json'

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_feature_ticks(config):
    """Generates feature ticks for plotting based on the configuration."""
    return [
        np.linspace(config["MIN_OVERETCH"], config["MAX_OVERETCH"], 11),
        np.linspace(config["MIN_OFFSET"], config["MAX_OFFSET"], 11),
        np.linspace(config["MIN_THICKNESS"], config["MAX_THICKNESS"], 11)
    ]

def main():
    try:
        setup_logging()

        # Load and process configuration
        config = DataProcessor.parse_config(CONFIGURATION_FILE)
        data_processor = DataProcessor(CONFIGURATION_FILE)
        data_processor.process()

        # Model initialization and loading
        model = NN_Model()
        model.load_model(config['MODEL_PATH'])

        # Data preparation
        test_processor = DataProcessor(CONFIGURATION_FILE)
        test_processor.load_data(file_name='TESTING_PATH')
        output_cols = test_processor.df.columns[5:-1]
        X, y = test_processor.df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (test_processor.df[output_cols].values)
        X_scaled, y_scaled = data_processor.scale_new_data(X, y)

        # Model evaluation and plotting
        y_pred = model.predict(X_scaled).reshape(y.shape)
        plot_predictions(model, y_pred, y, X, data_processor.time, max_plots=5)

        # Define and apply filters for the data
        feature_ticks = get_feature_ticks(config)
        feature_labels = [r'Overetch [$\mu$m]', r'Offset [$\mu$m]', r'Thickness [$\mu$m]']
        filters = {"2": 30.0, "1": 0.0, "0": 0.3}
        for feature, value in filters.items():
            filter_mask = X[:, int(feature)] == value
            errors = calculate_error_heatmap(y_pred[filter_mask], y[filter_mask], feature_ticks[(int(feature) + 1) % 3], feature_ticks[(int(feature) + 2) % 3])
            plot_error_heatmap(errors, feature_ticks[(int(feature) + 1) % 3], feature_ticks[(int(feature) + 2) % 3], feature_labels[(int(feature) + 1) % 3], feature_labels[(int(feature) + 2) % 3], title=f'Error Heatmap for {feature} = {value}')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
    logging.info("Num GPUs Available: %s", len(tf.config.experimental.list_physical_devices('GPU')))
