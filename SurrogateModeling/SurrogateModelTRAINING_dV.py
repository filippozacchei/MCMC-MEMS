import json
from utils import *

# Constants
CONFIGURATION_FILE = './config_VoltageSignal.json'

# Load configuration from a JSON file
def load_config():
    with open(CONFIGURATION_FILE, 'r') as config_file:
        config = json.load(config_file)
    return config

# Load configuration
config = load_config()

def learning_rate_schedule(epoch):
    return config['LEARNING_RATE']/(1.+epoch*0.1)

# Main execution function
def main():
    C_df = load_and_process_data(config)
    output_cols = C_df.columns[5:-1]
    X, y = C_df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (C_df[output_cols].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['TEST_SIZE'], random_state=config['RANDOM_STATE'])
    
    # Flatten the output and concatenate time column
    time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
    X_train, y_train = flatten_and_expand_data(X_train, y_train, output_cols, time)
    X_test, y_test = flatten_and_expand_data(X_test, y_test, output_cols, time)
    
    # Preprocess the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train the model
    model = build_model(X_train_scaled.shape[1],
                        n_neurons=config['N_NEURONS'],
                        n_layers=config['N_LAYERS'],
                        activation_func=config['ACTIVATION'])

    model.compile(loss='MeanSquaredError', optimizer=Adam(learning_rate=config['LEARNING_RATE']))

    history = model.fit(
        X_train_scaled, y_train, epochs=config['N_EPOCHS'], batch_size=config['BATCH_SIZE'], verbose=2,
        validation_data=(X_test_scaled, y_test),
        callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)]
    )
    
    # Save the model
    model.save(config['MODEL_PATH'])
    plot_training_history(history)

if __name__ == '__main__':
    main()
