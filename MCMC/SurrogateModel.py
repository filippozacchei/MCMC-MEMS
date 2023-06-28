# Default 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the directory containing the dataLoader module to the sys.path
sys.path.append('../DATA/')

# Import the load_data function from the dataLoader modulec
from dataLoader import load_data

# for Data Handling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# for Surrogate model
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define Input Parameters
sampling_frequency = 1e-3/1e-5
period = 2e-3
n_periods = 2
n_epochs = 50000
batch_size = 10
time = np.arange(0,2e-3,1e-5)

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/CapacityDataset.csv'
U_dataset_filename = '../DATA/CSV/UDataset.csv'
V_dataset_filename = '../DATA/CSV/VDataset.csv'

# Load the data using the load_data function and Obtain Capacity variation
C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=1.99ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=2.00ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=1.99ms'].values) / 1e-5
# Drop the last column
dC_df = dC_df.drop(dC_df.columns[-1], axis=1)

# Step 1: Prepare the Data
y = 1e13*dC_df.loc[:, 'Time=0.00ms':'Time=1.99ms'].values  # target variables: capacity variation values
X = dC_df[['AmplitudeX','Overetch']].values  # Input features: acceleration amplitudes and overetch

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def mse_l1(y_true, y_pred):
    mse = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    l1_norm = tf.norm(y_true, ord=1) + 1e-16 
    return mse / l1_norm

# Step 3: Build the Neural Network Surrogate Model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1]))  # Output layer with 1 unit for the target variable

# Define Loss
# Step 5: Compile the Model
model.compile(loss=mse_l1, optimizer=Adam(learning_rate=0.0001))

# Step 6: Train the Model
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(X_test_scaled, y_test))

# Create subplots for each parameter
num_params = 2
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

# Initialize empty lists to store the training and validation losses
train_losses = []
val_losses = []

# Function to update the plot
def update_plot(epoch, train_loss, val_loss):

    # Step 8: Predict with the Model
    y_pred = model.predict(X_test_scaled[0:num_params,:])
    
    # Clear the subplots
    for i, ax in enumerate(axes):
        ax.clear()
        
        # Plot the real and predicted time response for parameter i
        ax.plot(time, y_test[i, :], c='r', label='Real')
        ax.plot(time, y_pred[i, :], c='b', label='Surrogate')
        
        # Set labels and title for each subplot
        ax.set_xlabel('Time')
        ax.set_ylabel('ΔC [pF/s]')
        ax.set_title('Parameter {}'.format(i+1))
        ax.legend()
    
    # Refresh the plot
    plt.tight_layout()
    plt.pause(0.01)


history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2, validation_data=(X_test_scaled, y_test))  

# Save the model
model.save("model.h5")

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






