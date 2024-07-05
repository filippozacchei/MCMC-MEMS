import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class preprocessing:
    
    """
    Class to handle data Processing.
    """

    
    def __init__(self, config_path):

        with open(config_path, 'r') as file:
            self.config = json.load(file)

        self.data = None
        self.time= None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_scaled = self.X_test_scaled = self.scaler = None
        self.process()

    def process(self):
        self.load_data()
        self.split_data()
        self.scale_data(scaling_strategy=self.config['SCALING_STRATEGY'])

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

            if self.config['CONFIGURATION'] == "I" :
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

