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
n_epochs = 4000
batch_size = 100
n_nr = 32
time = np.arange(0, 2.5e-3, 1e-5)
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

# Select the input and output columns
input_cols = ['AmplitudeX', 'T_X', 'Overetch']
output_cols = dC_df.columns[7:-1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dC_df[input_cols].values, 1e12 * (dC_df[output_cols].values), test_size=0.2, random_state=2)

# Create a scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dC_df['AmplitudeX'], 1e3*dC_df['T_X'], dC_df['Overetch'], c='blue', label='Test Set', alpha=0.5)
ax.scatter(X_train[:, 0], 1e3*X_train[:, 1], X_train[:, 2], c='red', label='Training Set', alpha=0.7)

# Set labels and title
ax.set_xlabel('AmplitudeX', fontsize=12)
ax.set_ylabel('PeriodX [ms]', fontsize=12)
ax.set_zlabel(r'Overetch [$\mu$m]', fontsize=12)

xtick_values = [10*i for i in range(11)]
xtick_labels = [f'{i}g' for i in range(11)] # Generate the tick labels from 0g to 10g
plt.xticks(xtick_values, xtick_labels, fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Display the plot
plt.show()

# Flatten the output and concatenate time column
X_train_rep = np.repeat(X_train, len(output_cols), axis=0)
X_train_rep = np.column_stack((X_train_rep, np.tile(time, len(X_train_rep) // len(time))))
y_train_rep = y_train.flatten()

X_test_rep = np.repeat(X_test, len(output_cols), axis=0)
X_test_rep = np.column_stack((X_test_rep, np.tile(time, len(X_test_rep) // len(time))))
y_test_rep = y_test.flatten()

# Get the number of samples
num_samples = X_train_rep.shape[0]

# Generate a random permutation of indices
perm = np.random.permutation(num_samples)

X_train_rep = X_train_rep[perm]
y_train_rep = y_train_rep[perm]

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rep)
X_test_scaled = scaler.transform(X_test_rep)
print(y_test.shape)

# Step 3: Build the Neural Network Surrogate Model
model = Sequential()
model.add(Dense(n_nr, activation='tanh', input_shape=(X_train_scaled.shape[1],))) 
model.add(Dense(n_nr, activation='tanh'))
model.add(Dense(n_nr, activation='tanh'))
model.add(Dense(n_nr, activation='tanh'))
model.add(Dense(1))  # Output layer with 1 unit for the target variable

# Define the learning rate schedule
def learning_rate_schedule(epoch):
    return lr / (1 + epoch * 0.1)  # Adjust the decay rate as needed

# Compile the model with Adam optimizer and the learning rate schedule
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

# Train the model with the learning rate schedule
history = model.fit(X_train_scaled, y_train_rep, epochs=n_epochs, batch_size=batch_size, verbose=2,
                    validation_data=(X_test_scaled, y_test_rep),
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)])

train_losses = history.history['loss']
val_losses = history.history['val_loss']

# Save the model
model.save("modelNo1_latin.h5")

# Plot the training and testing errors
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b', label='Training Error')
plt.plot(epochs, val_losses, 'r', label='Testing Error')
plt.title('Training and Testing Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.show()


# # model = tf.keras.models.load_model('./modelNo0.h5')
# print(X_test_scaled)


# model = tf.keras.models.load_model('./modelNo1_latin.h5')
# Step 8: Predict with the Model
y_pred = model.predict(X_test_scaled).reshape(y_test.shape)

# Set figure size and resolution
plt.figure(figsize=(8, 6), dpi=300)

# Define colors and line styles
real_color = 'red'
surrogate_color = 'blue'
line_width = 1.5

# Iterate over test samples
for i in range(y_test.shape[0]):
    # Plot real values
    plt.plot(1e3 * time, y_test[i, :], c=real_color, label='Real', linewidth=line_width)

    # Plot surrogate values
    plt.plot(1e3 * time, y_pred[i, :], c=surrogate_color, label='Surrogate', linewidth=line_width)

    # Set axis labels
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\Delta C$ [fF/s]')

    # Set title and adjust font size
    plt.title('AmplitudeX = {}m/s; PeriodX = {} ms, Overetch = {:.2f} micron'.format(int(X_test[i, 0]), 1e3*X_test[i, 1], X_test[i, 2]), fontsize=12)

    # Set legend
    plt.legend(loc='upper right', fontsize=10)

    # Set grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the figure (optional)
    plt.savefig('predicted_vs_true.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()




