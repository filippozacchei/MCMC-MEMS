import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the directory containing the dataLoader module to the sys.path
sys.path.append('../DATA/')

# Import the load_data function from the dataLoader modulec
from dataLoader import load_data

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/CapacityDataset.csv'
U_dataset_filename = '../DATA/CSV/UDataset.csv'
V_dataset_filename = '../DATA/CSV/VDataset.csv'

# Load the data using the load_data function
C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)

# Print the loaded Capacity dataset
print("Capacitance Dataset:")
print(C_df)
print("\n\n")

# Perform the desired calculation on the selected columns
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=1.99ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=2.00ms'].values - C_df.loc[:, 'Time=0.00ms':'Time=1.99ms'].values) / 1e-5

# Drop the last column
dC_df = dC_df.drop(dC_df.columns[-1], axis=1)

# Print the modified dataset
print("Capacitance Variation Dataset:")
print(dC_df)
print("\n\n")

dC_df['C0'] = C_df['Time=0.00ms']
# Filter the dataset based on desired values
df_Ax = dC_df[dC_df['AmplitudeX'] > 5].copy()

# # Define the lines and columns to plot
# lines_to_plot = range(df_Ax.shape[0])
# columns_to_plot = df_Ax.columns[4:203]

# # Extract the values to plot
# values_to_plot = df_Ax.loc[lines_to_plot, columns_to_plot].values

# # Get unique overetch values
# overetch_values = df_Ax['Overetch'].unique()

# # Define a color palette for the overetch values
# color_palette = plt.cm.get_cmap('Set1', len(overetch_values))

# # Assign colors to each overetch value
# colors = [color_palette(i) for i in range(len(overetch_values))]

# # Create the plot
# plt.figure()
# for i, line in enumerate(lines_to_plot):
#     overetch = df_Ax.loc[line, 'Overetch']
#     plt.plot(values_to_plot[i], c=colors[overetch_values.tolist().index(overetch)])

# # Add labels and title to the plot
# plt.xlabel('Columns')
# plt.ylabel('Values')
# plt.title('C_df Values')

# # Add a legend to distinguish lines
# plt.legend()

# # Show the plot
# plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, AdamW
import tensorflow as tf

def max_absolute_error(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    max_error = tf.reduce_max(error)
    return max_error

# Step 1: Prepare the Data
X = df_Ax.loc[:, 'Time=0.00ms':'Time=1.99ms'].values  # Input features: capacity variation values
# X = np.hstack((X, df_Ax[['C0']].values))  # Insert 'AmplitudeX' column into X
y = df_Ax[['Overetch']].values  # Target variables: acceleration amplitudes and overetch

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build the Neural Network
model = Sequential()
model.add(Dense(32, activation='tanh', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1))  # Output layer with 1 unit for the target variable

# Step 5: Compile the Model
model.compile(loss='mean_squared_error', optimizer=AdamW(learning_rate=0.00001))

# Step 6: Train the Model
history = model.fit(X_train_scaled, y_train, epochs=10000, batch_size=10, verbose=1, validation_data=(X_test_scaled, y_test))

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

# Step 7: Evaluate the Model
mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Mean Squared Error:", mse)

# Step 8: Predict with the Model
y_pred = model.predict(X_test_scaled)
y_train_p = model.predict(X_train_scaled)
print(y_train.shape)
print(y_train_p.shape)

# Plot the predicted values against the true values
plt.scatter(y_train[:,0], y_train_p[:,0], c='r', label='Train')
plt.scatter(y_test[:,0], y_pred[:,0], c='b', label='Test')
plt.plot([min(y_test[:,0]), max(y_test[:,0])], [min(y_test[:,0]), max(y_test[:,0])], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Acceleration')
plt.show()

# Plot the predicted values against the true values
plt.scatter(y_train[:,1], y_train_p[:,1], c='r', label='Train')
plt.scatter(y_test[:,1], y_pred[:,1], c='b', label='Test')
plt.plot([min(y_test[:,1]), max(y_test[:,1])], [min(y_test[:,1]), max(y_test[:,1])], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Overetch')
plt.show()

# Step 9: Validate the Model
# Compare predicted_values with ground truth values

# Step 10: Fine-tune and Iterate as necessary
# Adjust the model architecture, hyperparameters, etc.

