import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

class ColumnMismatchError(Exception):
    """Exception raised for a mismatch in expected and actual CSV columns."""
    def __init__(self, expected, actual):
        super().__init__(f"Column mismatch in the CSV file. Expected {expected} columns; found {actual} columns. "
                         "Ensure the CSV and config file columns match.")


def parse_config(config_path):
    """
    Parses and validates the configuration file.
    
    Parameters:
    - config_path (str): Path to the configuration JSON file.
    
    Returns:
    - config (dict): A dictionary containing configuration parameters.
    
    Raises:
    - FileNotFoundError: If the JSON config file cannot be found.
    - ValueError: If required keys are missing in the config.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    required_keys = {'DATASET_PATH', 'INPUT_COLS', 'TIME_COL_START', 'TIME_COL_END',
                     'TIME_COL_SECOND', 'TIME_COL_SECOND_END', 'Y_SCALING_FACTOR', 'TEST_SIZE', 'RANDOM_STATE'}
    if not required_keys.issubset(config.keys()):
        missing_keys = required_keys - set(config.keys())
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return config

def load_data(config_file):
    """
    Loads data from a CSV file and assigns new column labels.

    Parameters:
    - config_file: A dictionary containing configuration parameters, including:
                   - 'DATASET_PATH': Path to the CSV file.
                   - 'INPUT_COLS': List of names of the input columns.
    
    Returns:
    - A pandas DataFrame with the data and new column labels.
    
    Raises:
    - FileNotFoundError: If the CSV file cannot be found at the path provided.
    - ColumnMismatchError: If the actual and expected column counts do not match.
    """
    try:
        df = pd.read_csv(config_file['DATASET_PATH'], comment='#')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {config_file['DATASET_PATH']}")

    try:
        # Ensure the CSV has the correct number of columns
        input_cols = config_file['INPUT_COLS']
        time_step = df.iloc[0, len(input_cols)]
        time_final = df.iloc[0, len(input_cols)+1]
        num_time_steps = int(time_final / time_step) + 1
        time_columns = [f"Time={1e3*time_step * i:.2f}ms" for i in range(num_time_steps)]
        column_labels = input_cols + ['ts', 'tf'] + time_columns

        if len(df.columns) != len(column_labels):
            raise ColumnMismatchError(len(column_labels), len(df.columns))

        df.columns = column_labels
        return df
    except KeyError as e:
        raise KeyError(f"Missing column in CSV or config: {e}")


def load_data_derivative(config_file):
    """
    Loads data from a CSV file and calculates the derivative of specified columns.

    Parameters:
    - config_file: A dictionary containing configuration parameters.

    Returns:
    - A tuple containing the original DataFrame and a new DataFrame with derivative calculations.
    """
    df = load_data(config_file)
    # Ensure that you are returning the derivative DataFrame 'd_df', not 'df'.
    d_df = df.copy()
    d_df.loc[:, config_file['TIME_COL_START']:config_file['TIME_COL_END']] = (
        df.loc[:, config_file['TIME_COL_SECOND']:config_file['TIME_COL_SECOND_END']].values -
        df.loc[:, config_file['TIME_COL_START']:config_file['TIME_COL_END']].values) / df.at[0, 'ts']
    return df, d_df


def split_data(df, config):
    """
    Splits the dataset into training and test sets.

    Parameters:
    - df: The DataFrame to split.
    - config: A dictionary containing configuration parameters.

    Returns:
    - Four numpy arrays: X_train, X_test, y_train, y_test.
    """
    output_cols = df.columns[5:-1]
    X, y = df[config['INPUT_COLS']].values, config['Y_SCALING_FACTOR'] * (df[output_cols].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['TEST_SIZE'], random_state=config['RANDOM_STATE'])
    return X_train, X_test, y_train, y_test


def stack_data(X, y, time):
    """
    Prepares time series data by repeating X for each time step and appending the time to each feature set.

    Parameters:
    - X: The input features.
    - y: The output features.
    - time: The time steps to add to the features.

    Returns:
    - Two numpy arrays: Expanded X with time steps, and flattened y.
    """
    output_cols = y.shape[1]
    X_rep = np.repeat(X, output_cols, axis=0)
    X_rep = np.column_stack((X_rep, np.tile(time, len(X_rep) // len(time))))
    y_rep = y.flatten()
    return X_rep, y_rep


def shuffle_data(X, y):
    """
    Randomly shuffles the data.

    Parameters:
    - X: Input features.
    - y: Output labels.

    Returns:
    - Shuffled input features and corresponding labels.
    """
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    return X_shuffled, y_shuffled


def scale_data(X_train, X_test, scaler=None):
    """
    Scales the training and testing data using MinMax scaling.

    Parameters:
    - X_train: Training features to fit the scaler.
    - X_test: Testing features to transform based on the scaler.
    - scaler: An instance of a scaler (default: MinMaxScaler).

    Returns:
    - Scaled training and testing features, and the scaler.
    """
    if scaler is None:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
