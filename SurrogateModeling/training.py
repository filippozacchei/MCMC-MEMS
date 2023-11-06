import json
from model import *
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# CONFIGURATION FILE
CONFIGURATION_FILE = './Config_files/config_VoltageSignal.json'

def main():
    try:
        config = parse_config(CONFIGURATION_FILE)    
        
        # Data preparation
        C_df, dC_df = load_data_derivative(config)
        X_train, X_test, y_train, y_test = split_data(dC_df, config)

        # Flatten the output and concatenate time column
        time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
        X_train_rep, y_train_rep = stack_data(X_train, y_train, time)
        X_test_rep, y_test_rep = stack_data(X_test, y_test, time)

        # Data shuffling and scaling
        X_train_rep, y_train_rep = shuffle_data(X_train_rep, y_train_rep)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train_rep, X_test_rep)  

        # Model building and training
        model = build_model(X_train_scaled.shape[1],
                            n_neurons=config['N_NEURONS'],
                            n_layers=config['N_LAYERS'],
                            activation_func=config['ACTIVATION'])
    

        ## TRAIN THE MODEL
        lr = initial_lr=config['LEARNING_RATE']

        # Define learning rate scheduler within main for access to config
        def learning_rate_schedule(epoch, decay_rate=config['DECAY_RATE']):
            return lr / (1. + epoch * decay_rate)

        model.compile(loss='MeanSquaredError', optimizer=Adam(learning_rate=config['LEARNING_RATE']))

        history = model.fit(
            X_train_scaled, y_train_rep, epochs=config['N_EPOCHS'], batch_size=config['BATCH_SIZE'], verbose=2,
            validation_data=(X_test_scaled, y_test_rep),
            callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)]
        )
        plot_training_history(history) 

        # Save model
        save_model(model, config['MODEL_PATH'])

        # Model evaluation and plotting
        evaluate_and_plot(model, X_test_scaled, y_test, time)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, log the error to a file or system log


if __name__ == '__main__':
    main()
