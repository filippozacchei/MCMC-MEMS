import json
import numpy as np
import tensorflow as tf
from training import *
from plotting import *

# CONFIGURATION FILE
CONFIGURATION_FILE = './Config_files/config_VoltageSignal.json'

def main():
    try:
        config = parse_config(CONFIGURATION_FILE)  

        # Data preparation
        C_df, dC_df = load_data_derivative(config)
        X_train, X_test, y_train, y_test = split_data(C_df, config)

        # Flatten the output and concatenate time column
        time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
        X_train_rep, y_train_rep = stack_data(X_train, y_train, time)
        X_test_rep, y_test_rep = stack_data(X_test, y_test, time)

        # Data shuffling and scaling
        X_train_rep, y_train_rep = shuffle_data(X_train_rep, y_train_rep)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train_rep, X_test_rep)  

        feature_labels = [r'Overetch [$\mu$m]', r'Offset [$\mu$m]', r'Thickness [$\mu$m]']
        feature_ticks = [np.linspace(config["MIN_OVERETCH"], config["MAX_OVERETCH"], 11),
                         np.linspace(config["MIN_OFFSET"], config["MAX_OFFSET"], 11),
                         np.linspace(config["MIN_THICKNESS"], config["MAX_THICKNESS"], 11)]
        
        plot_train_test(X_train, X_test, feature_labels, feature_ticks)

        # Train or load the model
        model = train(config) if config['TRAINING'] else tf.keras.models.load_model(config["MODEL_PATH"])

        # Evaluate and Plot
        y_pred = model.predict(X_test_scaled).reshape(y_test.shape)
        plot_predictions(model, y_pred, y_test, time, max_plots=5)

        # Define filters for the data
        filters = {
            "2": 30.0,
            "1": 0.0,
            "0": 0.3
        }

        # Iterate through the filters and plot error heatmaps
        for feature, value in filters.items():
            filter_mask = X_test[:, int(feature)] == value
            y_test_filtered = y_test[filter_mask]
            y_pred_filtered = y_pred[filter_mask]

            # Calculate and plot the error heatmap
            errors = calculate_error_heatmap(y_pred_filtered, y_test_filtered, feature_ticks[0], feature_ticks[1])
            plot_error_heatmap(errors, feature_ticks[0], feature_ticks[1], feature_labels[0], feature_labels[1], title=f'Error Heatmap for {feature} = {value}')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
