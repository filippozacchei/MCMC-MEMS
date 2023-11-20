import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataLoadingError(Exception):
    """Base class for exceptions in this module."""
    pass

class FileNotFoundError(DataLoadingError):
    """Exception for when a file is not found."""
    pass

class ColumnMismatchError(DataLoadingError):
    """Exception for mismatch in the expected and actual number of DataFrame columns."""
    pass

class ConfigurationError(DataLoadingError):
    """Exception for issues in the configuration."""
    pass

class DataProcessor:
    def __init__(self, config_path):
        self.config = self.parse_config(config_path)
        self.df = None
        self.d_df = None
        self.time= None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_scaled = self.X_test_scaled = self.scaler = None

    def process(self):
        self.load_data()
        self.split_data()
        self.scale_data(scaling_strategy=self.config['SCALING_STRATEGY'])


    @staticmethod
    def parse_config(config_path):
        """
        Parses and Validates a Configuration File

        This function reads a JSON configuration file, verifies if all required keys are present, 
        and returns the configuration as a dictionary. It's designed to ensure that the configuration
        for a specific process or application is complete and correctly formatted before use.

        Parameters
        ----------
        config_path : str
            The file path to the JSON configuration file.

        Returns
        -------
        dict
            A dictionary containing the configuration parameters.

        Raises
        ------
        FileNotFoundError
            If the JSON configuration file cannot be found at the provided path.

        ValueError
            If there are missing required keys in the configuration file. The error message includes
            the list of missing keys.

        Examples
        --------
        To use this function, simply provide the path to your configuration file:

        >>> config = parse_config('path/to/config.json')
 
        {'DATASET_PATH': 'data/', 'INPUT_COLS': ['col1', 'col2'], ...}

        Notes
        -----
        The function expects the following keys in the configuration file:
        - 'DATASET_PATH'
        - 'INPUT_COLS'
        - 'TIME_COL_START'
        - 'TIME_COL_END'
        - 'TIME_COL_SECOND'
        - 'TIME_COL_SECOND_END'
        - 'Y_SCALING_FACTOR'
        - 'TEST_SIZE'
        - 'RANDOM_STATE'
        - 'MIN_OVERETCH'
        - 'MAX_OVERETCH'
        - 'MIN_OFFSET'
        - 'MAX_OFFSET'
        - 'MIN_THICKNESS'
        - 'MAX_THICKNESS'

        Ensure that your configuration file includes all these keys.
        """

        # Attempt to open and load the JSON file
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file could not be found at the specified path: '{config_path}'. Please ensure the path is correct and the file exists.")

        # Define the set of required keys
        required_keys = {
            'DATASET_PATH', 'INPUT_COLS', 'TIME_COL_START', 'TIME_COL_END',
            'TIME_COL_SECOND', 'TIME_COL_SECOND_END', 'Y_SCALING_FACTOR', 'TEST_SIZE',
            'RANDOM_STATE', 'MIN_OVERETCH', 'MAX_OVERETCH', 'MIN_OFFSET',
            'MAX_OFFSET', 'MIN_THICKNESS', 'MAX_THICKNESS'
        }

        # Check if all required keys are present in the config
        missing_keys = required_keys - config.keys()
        if not required_keys.issubset(config.keys()):
            missing_keys = required_keys - set(config.keys())
            raise ValueError(f"The configuration file is missing some required keys. Missing keys: {missing_keys}. Please ensure the configuration includes all necessary parameters.")
    

        return config

    def load_data(self, file_name='DATASET_PATH'):
        """
        Loads data from a CSV file and assigns new column labels.
        
        Returns:
        - A pandas DataFrame with the data and new column labels.
        
        Raises:
        - FileNotFoundError: If the CSV file cannot be found at the path provided.
        - ColumnMismatchError: If the actual and expected column counts do not match.
        """
        try:
            self.df = pd.read_csv(self.config[file_name], dtype=str, comment='#', float_precision='high', header=None)
            self.df = self.df.astype(np.float64)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to locate the data file. Expected file path from configuration: '{self.config[file_name]}'. Please verify the file path in the configuration.")

        try:
            input_cols = self.config['INPUT_COLS']
            
            # Extracting time step information directly into variables
            time_step = self.df.iloc[0, len(input_cols)]
            time_final = self.df.iloc[0, len(input_cols) + 1]
            num_time_steps = int(time_final/time_step) + 1

            self.time = np.arange(0,time_final,time_step)

            # Generating time column names in a compact manner
            time_columns = [f"Time={1e3 * time_step * i:.2f}ms" for i in range(num_time_steps)]
            column_labels = input_cols + ['ts', 'tf'] + time_columns

            # Check for column count mismatch
            if len(self.df.columns) != len(column_labels):
                raise ColumnMismatchError(f"The number of columns in the data file does not match the expected count. Expected {len(column_labels)} columns as per configuration, but found {len(df.columns)} columns in the file. Please check the data file and the 'INPUT_COLS' configuration.")      
            self.df.columns = column_labels    

            self.d_df = self.df.copy()
            self.d_df.loc[:, self.config['TIME_COL_START']:self.config['TIME_COL_END']] = (
                self.df.loc[:, self.config['TIME_COL_SECOND']:self.config['TIME_COL_SECOND_END']].values -
                self.df.loc[:, self.config['TIME_COL_START']:self.config['TIME_COL_END']].values) / self.df.at[0, 'ts']
        except KeyError as e:
            raise KeyError(f"A required column key is missing either in the CSV file or in the configuration. Missing key: '{e}'. Please ensure all necessary columns are defined in both the CSV and the configuration.")

    def split_data(self):
        """
        Splits the dataset into training and test sets.

        Parameters:
        - df: The DataFrame to split.
        - config: A dictionary containing configuration parameters.

        Returns:
        - Four numpy arrays: X_train, X_test, y_train, y_test.
        """
        output_cols = self.df.columns[5:-1]
        X, y = self.df[self.config['INPUT_COLS']].values, self.config['Y_SCALING_FACTOR'] * (self.df[output_cols].values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.config['TEST_SIZE'], random_state=self.config['RANDOM_STATE'])

    @staticmethod
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

    @staticmethod
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

    def scale_data(self, scaling_strategy=None):
        """
        Scales the training and testing data using MinMax scaling.

        Returns:
        - Scaled training and testing features, and the scaler.
        """
        if scaling_strategy is None or scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
         
        self.X_train, self.y_train = self.shuffle_data(self.X_train, self.y_train)
        X_train_rep, y_train_rep = self.stack_data(self.X_train, self.y_train, self.time)
        X_test_rep, y_test_rep = self.stack_data(self.X_test, self.y_test, self.time)
        self.X_train_scaled = self.scaler.fit_transform(X_train_rep)
        self.X_test_scaled = self.scaler.transform(X_test_rep)
        self.y_train_scaled = y_train_rep
        self.y_test_scaled = y_test_rep
    
    def scale_new_data(self, X, y):
        """
        Scales new data.

        Returns:
        - Scaled training and testing features, and the scaler.
        """
        X_rep, y_rep = self.stack_data(X, y, self.time)
        X_scaled = self.scaler.transform(X_rep)
        y_scaled = y_rep
        return X_scaled, y_scaled