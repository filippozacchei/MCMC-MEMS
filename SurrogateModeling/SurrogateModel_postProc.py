import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

sys.path.append('../DATA/')

from dataLoader import load_data

# Define Input Parameters
time = np.arange(0, 1.5e-3, 1e-5)

# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/C_training_HalfSine.csv'

# Load the data using the load_data function and Obtain Capacity variation
C_df = load_data(C_dataset_filename)
C_df = C_df.dropna()
print(C_df)
dC_df = C_df.copy()
dC_df.loc[:, 'Time=0.00ms':'Time=1.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=1.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=1.49ms'].values) / 1e-5

# Select the input and output columns
input_cols = ['Overetch', 'Offset', 'Thickness']
output_cols = dC_df.columns[5:-1]

# Split the data into train and test sets
num_folds=5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(dC_df[input_cols].values,
                                                    1e12 * (dC_df[output_cols].values),     
                                                    test_size=0.2, 
                                                    random_state=2)




# # Create a scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='red',  label='Training Set', alpha=0.5)
# ax.scatter(X_test[:, 0],  X_test[:, 1],  X_test[:, 2],  c='blue', label='Test Set',     alpha=0.5)

# # Set labels and title
# ax.set_xlabel('AmplitudeX', fontsize=12)
# ax.set_ylabel('PeriodX [ms]', fontsize=12)
# ax.set_zlabel(r'Overetch [$\mu$m]', fontsize=12)

# xtick_values = [10*i for i in range(11)]
# xtick_labels = [f'{i}g' for i in range(11)] # Generate the tick labels from 0g to 10g
# plt.xticks(xtick_values, xtick_labels, fontsize=10)

# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(fontsize=10)

# # Display the plot
# plt.show()

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
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_rep)
X_test_scaled = scaler.transform(X_test_rep)


# Define the file paths for the datasets
C_dataset_filename = '../DATA/CSV/C_testing_HalfSine.csv'

# Load the data using the load_data function and Obtain Capacity variation
C_df = load_data(C_dataset_filename)
C_df = C_df.dropna()
dC_test = C_df.copy()
dC_test.loc[:, 'Time=0.00ms':'Time=1.49ms'] = (C_df.loc[:, 'Time=0.01ms':'Time=1.50ms'].values \
                                           - C_df.loc[:, 'Time=0.00ms':'Time=1.49ms'].values) / 1e-5


X_test=dC_test[input_cols].values
y_test=1e12 * dC_test[output_cols].values

X_test_rep = np.repeat(X_test, len(output_cols), axis=0)
X_test_rep = np.column_stack((X_test_rep, np.tile(time, len(X_test_rep) // len(time))))
y_test_rep = y_test.flatten()

X_test_scaled = scaler.transform(X_test_rep)

import tensorflow as tf
model_path = '../ML-Models/modelNo1_dV_latin.h5'
model = tf.keras.models.load_model(model_path)
y_pred = model.predict(X_test_scaled).reshape(y_test.shape)

# Set figure size and resolution
plt.figure(figsize=(8, 6), dpi=300)

# Define colors and line styles
real_color = 'red'
surrogate_color = 'blue'
line_width = 1.5

etch = np.linspace(0.1,0.4,5)
offset = np.linspace(-0.4,0.4,5)
thickness = np.linspace(29.0,31.0,4)

errors = np.zeros((5,5))

from scipy.stats import pearsonr

# # Iterate over test samples
# for i in range(y_test.shape[0]):
#     # Plot real values
#     plt.plot(1e3 * time, 1e-6*y_test[i, :], c=real_color, label='Real', linewidth=line_width)

#     # Plot surrogate values
#     plt.plot(1e3 * time, y_pred[i, :], c=surrogate_color, label='Surrogate', linewidth=line_width)

#     # Set axis labels
#     plt.xlabel('Time [ms]')
#     plt.ylabel(r'$\Delta C$ [fF/s]')

#     # Set title and adjust font size
#     plt.title(
#         'Overetch = {:.2f}; Offset = {:.2f}, Thickness = {:.2f}'.format(
#                 X_test[i, 0],
#                 X_test[i, 1],  
#                 X_test[i, 2]),
#                 fontsize=12)

#     # Set legend
#     plt.legend(loc='upper right', fontsize=10)

#     # Set grid lines
#     plt.grid(True, linestyle='--', alpha=0.5)

#     # Save the figure (optional)
#     plt.savefig('predicted_vs_true.png', dpi=300, bbox_inches='tight')

#     # Show the plot
#     plt.show()

# Calculate and store errors
for i in range(25):
    out = y_pred[i, :]
    exa = 1e-6*y_test[i, :]
    err = np.max(np.linalg.norm(out - exa, 1)) / np.linalg.norm(exa, 1)
    row = int(i / 5)
    col = i % 5
    print(row,col)
    errors[row, col] = err

print(errors)
# print(np.min(np.min(errors)))

# Create a heatmap of the errors
plt.figure(figsize=(10, 6))
cax = plt.matshow(errors, cmap='viridis')  # You can choose a different colormap if needed
plt.colorbar(cax)
plt.xticks(np.arange(5), [f'{e:.2f}um' for e in etch])
plt.yticks(np.arange(5), [f'{a:.2f}um' for a in np.flip(offset)])
# plt.yticks(np.arange(11), [f'{t * 1e3:.2f} ms' for t in np.flip(Tx)])
# plt.xlabel('Overetch (µm)')
plt.ylabel(r'Offset [${\mu}$m]')
# plt.ylabel('Acceleraion Amplitude')
plt.xlabel(r'Overetch [${\mu}$m]')
plt.title('Relative Error')
plt.gca().xaxis.tick_bottom()  # Move x-axis ticks to the bottom
plt.gca().yaxis.tick_left()  # Move y-axis ticks to the left
plt.gca().xaxis.set_label_position('bottom')  # Set x-axis label position
plt.gca().yaxis.set_label_position('left')  # Set y-axis label position

# # Save the figure as a high-quality image for presentation
# plt.savefig('relative_l1_error_heatmap.png', dpi=300, bbox_inches='tight')

# # Display the plot
plt.show()


plt.figure(figsize=(12, 8))  # Increase size for better visibility

# Use a clear colormap
cax = plt.matshow(errors, cmap='plasma')  # 'plasma' colormap offers clear color differences and is perceptually uniform

cb = plt.colorbar(cax, shrink=0.8, aspect=5)  # Adjust colorbar size
cb.set_label('Relative Error', rotation=270, labelpad=20, fontsize=14)  # Label colorbar

# Adjust font size and labels for clarity
plt.xticks(np.arange(len(etch)), [f'{e:.2f}µm' for e in etch], fontsize=12)
plt.yticks(np.arange(len(offset)), [f'{a:.2f}µm' for a in np.flip(offset)], fontsize=12)

plt.xlabel(r'Overetch [${\mu}$m]', fontsize=16)
plt.ylabel(r'Offset [${\mu}$m]', fontsize=16)
plt.title('Relative Error vs Overetch and Offset', fontsize=18, pad=20)  # Increase padding to ensure title doesn't overlap with plot

# Ensure ticks and labels are on the outside and positioned appropriately
plt.tick_params(direction='out', length=6, width=2, colors='k', which='major', grid_color='k', grid_alpha=0.5)
plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()
plt.gca().xaxis.set_label_position('bottom')
plt.gca().yaxis.set_label_position('left')

# Add grid lines for better visibility
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()  # Ensure all elements fit within the figure area
plt.show()


# # Iterate over test samples
# for i in range(y_test.shape[0]):
# # Create a subplot for each sample (adjust the layout as needed)
#     # plt.subplot(3, 3, i + 1)
#     T = X_test[i,3]
#     etch = X_test[i,6]

#     # Plot real values
#     plt.plot(1e3 * time, y_test[i, :], c=real_color, label='Real', linewidth=line_width)

#     # Plot surrogate values
#     plt.plot(1e3 * time, y_pred[i, :], c=surrogate_color, label='Surrogate', linewidth=line_width)

#     # Set axis labels
#     plt.xlabel('Time [ms]', fontsize=12)
#     plt.ylabel(r'$\Delta C$ [fF/s]', fontsize=12)

#     # Set title with relevant information
#     plt.title(
#         f'Ax = {int(X_test[i, 0]/9.81)}g; Tx = {1e3 * X_test[i, 3]:.2f}ms; Overetch = {X_test[i, 6]:.2f}μm',
#         fontsize=10, loc='center')

#     # Set legend
#     plt.legend(loc='upper right', fontsize=10)

#     # Set grid lines
#     plt.grid(True, linestyle='--', alpha=0.5)

#     # Adjust the layout for subplots (spacing between plots)
#     plt.tight_layout()

#     # Save the figure (optional)
#     plt.savefig('predicted_vs_true_subplot.png', dpi=300, bbox_inches='tight')

#     # Show the plot
    # plt.show()




