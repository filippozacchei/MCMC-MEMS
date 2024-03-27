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
    
    """
    Class to handle data Processing.
    """

    
    def __init__(self, config_path):
        self.config = self.parse_config(config_path)
        self.data = None
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
        Parses and validates a JSON configuration file.

        Reads a JSON configuration file and ensures all required keys are present. Returns a dictionary with the configuration parameters, suitable for use in specific processes or applications.

        :param config_path: (str) Path to the JSON configuration file.
        :return: (dict) Configuration parameters.
        
        :raises FileNotFoundError: If the configuration file is not found at `config_path`.
        :raises ValueError: If required keys are missing in the configuration file. Lists missing keys.

        :examples: 
            >>> config = parse_config('path/to/config.json')
            {'DATASET_PATH': 'data/', 'INPUT_COLS': ['col1', 'col2'], ...}

        :notes:
            Required keys in the configuration file:
            - 'DATASET_PATH'
            - 'INPUT_COLS'
            - 'Y_SCALING_FACTOR'
            - 'TEST_SIZE'
            - 'RANDOM_STATE'
            - 'MIN_OVERETCH'
            - 'MAX_OVERETCH'
            - 'MIN_OFFSET'
            - 'MAX_OFFSET'
            - 'MIN_THICKNESS'
            - 'MAX_THICKNESS'
        """

        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file could not be found at the specified path: '{config_path}'.")

        required_keys = {
            'DATASET_PATH', 'INPUT_COLS', 'Y_SCALING_FACTOR', 'TEST_SIZE',
            'RANDOM_STATE', 'MIN_OVERETCH', 'MAX_OVERETCH', 'MIN_OFFSET',
            'MAX_OFFSET', 'MIN_THICKNESS', 'MAX_THICKNESS'
        }

        # Check if all required keys are present in the config
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"The configuration file is missing some required keys. Missing keys: {missing_keys}. Please ensure the configuration includes all necessary parameters.")

        return config

    def load_data(self):

        """

        Loads and processes data from a CSV file specified in the configuration.
        s
        This method reads a CSV file using a file path from the class configuration, processes it based on predefined rules, and stores the result in class attributes.

        :raises FileNotFoundError: If the CSV file cannot be found at the path provided in the configuration.
        :raises ColumnMismatchError: If the actual and expected column counts do not match.
        :raises KeyError: If a required configuration key is missing.

        """

        try:
            self.data = pd.read_csv(self.config['DATASET_PATH'], 
                                    dtype=str, 
                                    comment='#', 
                                    float_precision='high', 
                                    header=None).astype(np.float64)

            input_cols = self.config['INPUT_COLS']

            if self.config['CONFIGURATION'] == "I":

                time_step = self.data.iloc[0, len(input_cols)]
                time_final = self.data.iloc[0, len(input_cols) + 1]
                num_time_steps = int(time_final/time_step) + 1
                self.time = np.arange(0, time_final, time_step)

                time_columns = [f"Time={1e3 * time_step * i:.2f}ms" for i in range(num_time_steps)]
                self.data.columns = input_cols + ['ts', 'tf'] + time_columns
            
            else:
                self.data.columns = input_cols + ['sensitivity']

        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to locate the data file at '{self.config[file_name]}'. Verify file path in configuration.")

        except ColumnMismatchError:
            expected_cols = len(column_labels)
            found_cols = len(self.data.columns)
            raise ColumnMismatchError(f"Column count mismatch. Expected {expected_cols}, found {found_cols}. Check 'INPUT_COLS' in configuration.")
        

    def split_data(self):

        """

        Splits the dataset into training and test sets based on the configuration.

        This method uses class attributes for the dataset and configuration. It assumes that 'INPUT_COLS' and 'TEST_SIZE' are defined in the configuration.
        
        """

        if len(self.data.columns) > 5:
            output_cols = self.data.columns[5:-1]
        else:
            output_cols = self.data.columns[-1]

        X = self.data[self.config['INPUT_COLS']].values
        y = self.config['Y_SCALING_FACTOR'] * self.data[output_cols].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config['TEST_SIZE'], random_state=self.config['RANDOM_STATE']
        )

    @staticmethod
    def stack_data(X, y, time):

        """

        Prepares time series data by expanding the input features (X) and flattening the output features (y).
        
        This method repeats each row of X for the number of time steps, and appends the corresponding time step to each repeated row. The output array y is flattened.

        :param X: (numpy array) the input features with shape (n_samples, n_features).
        :param y: (numpy array) the output features with shape (n_samples, n_time_steps).
        :param time: (numpy array) the time steps with length equal to n_time_steps.

        :return X_rep: (numpy array) expanded X with time steps appended.
        :return y_rep: (numpy array) flattened y.

        """

        output_cols = y.shape[1]
        X_rep = np.repeat(X, output_cols, axis=0)
        time_repeated = np.tile(time, len(X_rep) // len(time))
        X_rep = np.column_stack((X_rep, time_repeated))
        y_rep = y.flatten()
        return X_rep, y_rep


    @staticmethod
    def shuffle_data(X, y):

        """

        Randomly shuffles the data.

        :param X: Input features.
        :param y: Output labels.

        :return: Shuffled input features and corresponding labels.

        """

        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        return X_shuffled, y_shuffled

    def scale_data(self, scaling_strategy=None):

        """

        Scales the training and testing data.

        It updates the values of :attribute X_train_scaled

        :param scaling_strategy: Optional; specifies the scaling strategy ('standard' for StandardScaler, 'minmax' for MinMaxScaler). Defaults to 'standard'.

        """

        self.scaler = self.select_scaler(scaling_strategy)
        self.X_train, self.y_train = self.shuffle_data(self.X_train, self.y_train)

        if self.time is not None:
            X_train_rep, y_train_rep = self.stack_data(self.X_train, self.y_train, self.time)
            X_test_rep, y_test_rep = self.stack_data(self.X_test, self.y_test, self.time)
            self.X_train_scaled = self.scaler.fit_transform(X_train_rep)
            self.X_test_scaled = self.scaler.transform(X_test_rep)
            self.y_train_scaled = y_train_rep
            self.y_test_scaled = y_test_rep
        else:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.y_train_scaled = self.y_train
            self.y_test_scaled = self.y_test

    # Helper method for selecting the scaler
    def select_scaler(self, scaling_strategy):
        if scaling_strategy == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()

    
    def scale_new_data(self, X, y=None):

        """

        Scales new data using the same scaler as the training data.

        :param X: Input features to be scaled.
        :param y: Optional; output features to be scaled. If provided, it will be processed similarly to X.

        :return X_scaled: Scaled input features.
        :return y_scaled: Scaled output features, if y is provided; otherwise, None.

        """

        if self.time is not None:
            X_rep, y_rep = self.stack_data(X, y, self.time)
            X_scaled = self.scaler.transform(X_rep)
            y_scaled = y_rep
        else:
            X_scaled = self.scaler.transform(X)
            y_scaled = None

        return X_scaled, y_scaled



# class LSTMDataProcessor(DataProcessor):
#     def __init__(self, config_path):
#         super().__init__(config_path)

#     @staticmethod
#     def VoltageProfile(t, amplitude_val=1.8, Tx_val=0.4e-3):
#         return 0.5 * amplitude_val * (1 + np.sin((t / Tx_val - 1 / 4) * 2 * np.pi)) * (t < 2 * Tx_val)

#     @staticmethod
#     def stack_data(X, y, time):
#         """
#         Reshapes data into sequences for LSTM training, including the VoltageProfile feature.

#         Parameters:
#         - X: The input features, expected shape [n_samples, n_features].
#         - y: The output features, expected shape [n_samples, output_features].
#         - time_steps: The number of time steps to be used in each sequence.

#         Returns:
#         - Two numpy arrays: X_seq and y_seq, reshaped for LSTM.
#         """
#         n_samples, n_features = X.shape
#         X_seq = np.zeros((n_samples, len(time), n_features + 2))
#         y_seq = np.zeros((n_samples, len(time), 1))

#         for i in range(n_samples):
#             for t in range(len(time)):
#                 X_seq[i, t, :-2] = X[i]
#                 X_seq[i, t, -2] = time[t]
#                 X_seq[i, t, -1] = LSTMDataProcessor.VoltageProfile(time[t])  # Assuming each time step is 1e-5
#                 y_seq[i, t, :] = y[i,t]
#         return X_seq, y_seq


#     def scale_data(self, scaling_strategy=None):
#         """
#         Scales 3D LSTM data and then reshapes it back to 3D.

#         Parameters:
#         - X: 3D data array of shape [n_samples, n_timesteps, n_features]
#         - scaler: An instance of a scaler, e.g., StandardScaler or MinMaxScaler
#         - n_timesteps: Number of timesteps in each sequence

#         Returns:
#         - Scaled and reshaped data
#         """

#         self.X_train, self.y_train = self.shuffle_data(self.X_train, self.y_train)
#         X_train_rep, y_train_rep = self.stack_data(self.X_train, self.y_train, self.time)
#         X_test_rep, y_test_rep = self.stack_data(self.X_test, self.y_test, self.time)

#         n_samples, n_timeSteps, n_features = X_train_rep.shape
#         n_samples_test = X_test_rep.shape[0]
        
#         # Flatten to 2D
#         X_train_rep = X_train_rep.reshape(-1, n_features)
#         X_test_rep = X_test_rep.reshape(-1, n_features)
#         y_train_rep = y_train_rep
#         y_test_rep = y_test_rep
          
#         if scaling_strategy is None or scaling_strategy == 'standard':
#             self.scaler = StandardScaler()
#         elif scaling_strategy == 'minmax':
#             self.scaler = MinMaxScaler()
         
#         self.X_train_scaled = self.scaler.fit_transform(X_train_rep).reshape(n_samples, n_timeSteps, n_features)
#         self.X_test_scaled = self.scaler.transform(X_test_rep).reshape(n_samples_test, n_timeSteps, n_features)
#         self.y_train_scaled =  y_train_rep
#         self.y_test_scaled =  y_test_rep
