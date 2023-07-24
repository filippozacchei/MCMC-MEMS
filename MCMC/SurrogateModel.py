import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

sys.path.append('../DATA/')
from dataLoader import load_data

# Define Input Parameters
sampling_frequency = 1e-3 / 1e-5
period = 2e-3
n_periods = 2
n_epochs = 50000
batch_size = 100
n_nr = 32
time = np.arange(0, 2e-3, 1e-5)
lr = 0.01

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/CapacityDataset.csv'
U_dataset_filename = '../DATA/CSV/UDataset.csv'
V_dataset_filename = '../DATA/CSV/VDataset.csv'

# Load the data using the load_data function and Obtain Capacity variation
C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=2.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=2.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=2.49ms'].values) / 1e-5

dC_df = dC_df.drop_duplicates(subset=dC_df.columns[[0,3]])

# Select the input and output columns
input_cols = ['AmplitudeX', 'T_X', 'Overetch']
output_cols = dC_df.columns[7:-1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dC_df[input_cols].values, 1e12 * (dC_df[output_cols].values), test_size=0.2, random_state=42)

# Create a scatter plot
plt.scatter(dC_df['AmplitudeX'], dC_df['Overetch'], c='blue', label='Test Set', alpha=0.5)
plt.scatter(X_train[:, 0], X_train[:, 1], c='red', label='Training Set', alpha=0.7)

# Set labels and title
plt.xlabel('AmplitudeX', fontsize=12)
plt.ylabel('Overetch', fontsize=12)

# Adjust x-ticks and y-ticks
xtick_values = np.unique(np.concatenate([dC_df['AmplitudeX'], X_train[:, 0]]))
xtick_labels = [f'{i}g' for i in range(11)] # Generate the tick labels from 0g to 10g
plt.xticks(xtick_values, xtick_labels, fontsize=10)
plt.yticks(np.unique(np.concatenate([dC_df['Overetch'], X_train[:, 1]])), fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Display the plot
# plt.show()

plt.savefig('train_test_split_scatter.png', dpi=300)

# Print the shapes of the train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Get the number of samples
num_samples = X_train.shape[0]

# Generate a random permutation of indices
perm = np.random.permutation(num_samples)

X_train = X_train[perm]
y_train = y_train[perm]

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Build the Neural Network Surrogate Model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1]))  # Output layer with 1 unit for the target variable

# Define the learning rate schedule
def learning_rate_schedule(epoch):
    return lr / (1 + epoch * 0.1)  # Adjust the decay rate as needed

# Compile the model with Adam optimizer and the learning rate schedule
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

# Train the model with the learning rate schedule
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)])


# # Save the model
# model.save("model_200output.h5")

# Collect training and testing errors during training
train_errors = history.history['loss']
test_errors = history.history['val_loss']

# Plot the training and testing errors
epochs = range(1, len(train_errors) + 1)
plt.plot(epochs, train_errors, 'b', label='Training Error')
plt.plot(epochs, test_errors, 'r', label='Testing Error')
plt.title('Training and Testing Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.show()

# model = tf.keras.models.load_model("./model_200output.h5")

# Step 8: Predict with the Model
y_pred = model.predict(X_test_scaled)
y_train_p = model.predict(X_train_scaled)

# Plot the predicted values against the true values
for i in range(y_test.shape[0]):
    plt.plot(time, y_test[i,:], c='r', label='Real')
    plt.plot(time, y_pred[i,:], c='b', label='Surrogate')
    plt.xlabel('Time')
    plt.ylabel('{\Delta}C [pF/s]')
    plt.title('AmplitudeX={}; Overetch={}'.format(X_test[i,0], X_test[i,1]))
    plt.show()






