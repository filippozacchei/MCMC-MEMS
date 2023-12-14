import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
from matplotlib import patheffects


sys.path.append('../DATA/')
sys.path.append('../SurrogateModeling/')

from dataLoader import DataProcessor
from model import NN_Model

true_values = [3.3062500e-01, 6.0800000e-03, 3.0515042e+01]
# Load samples
samples = np.load('./samples.npy')[0]

# Number of parameters
num_params = samples.shape[0]
strings = ['Overetch', 'Offset', 'Thickness']

# Set global figure style
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['font.size'] = 12

# Create histograms for each parameter
for i in range(num_params):
    plt.figure()
    
    # Histogram
    n, bins, patches = plt.hist(samples[i, :], bins=50, density=True, color='skyblue', edgecolor='black', label=None)
    
    # Define text properties for bold and black borders
    text_properties = {'fontsize': 16, 'color': 'darkred'}

    # Vertical line for the true value
    plt.axvline(true_values[i], color='darkred', linestyle='dashed', linewidth=3, label='True')
    
    # Define text properties for bold and black borders
    text_properties = {'fontsize': 16, 'color': 'blue'}

    # Vertical line for the mean
    mean_value = np.mean(samples[i, :])
    plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=3, label='Mean')

    # Highlight the 90% confidence interval (Light Gray)
    ci_lower, ci_upper = np.percentile(samples[i, :], [5, 95])
    plt.fill_betweenx([0, max(n)], ci_lower, ci_upper, color='gray', alpha=0.35, label='95% CI')
    
    plt.legend()
    # Plot labels and title
    plt.xlabel(strings[i], fontsize=14)
    plt.ylabel('Density', fontsize=14)
    # plt.title(f'Histogram of {strings[i]}', fontsize=16)
    plt.grid(True)
    plt.show()

# Scatter plots
alpha_val = 0.05
for i in range(num_params):
    for j in range(i + 1, num_params):
        plt.figure()
        plt.scatter(samples[i, :], samples[j, :], alpha=alpha_val, color='steelblue')
        plt.xlabel(strings[i], fontsize=14)
        plt.ylabel(strings[j], fontsize=14)
        # plt.title(f'Scatter Plot: {strings[i]} vs {strings[j]}', fontsize=16)
        plt.grid(True)
        plt.show()

# Load model and process data
model = NN_Model()
model.load_model('../SurrogateModeling/Saved_models/model_sensitivity.h5')
data_processor = DataProcessor('../SurrogateModeling/Config_files/config_sensitivity.json')
data_processor.process()
samples_transposed = samples.T
samples_transposed, _ = data_processor.scale_new_data(samples_transposed)

# Make predictions and calculate mean
predicted_sensitivities = model.predict(samples_transposed)
mean_sensitivity = np.mean(predicted_sensitivities)

true_value = 4.515042

# Plot histogram of predicted sensitivities
plt.figure()
n, bins, patches = plt.hist(predicted_sensitivities.flatten(), bins=30, density=True, color='coral', edgecolor='black')
# plt.title('Histogram of Predicted Sensitivities', fontsize=16)
plt.xlabel('Sensitivity', fontsize=14)
plt.ylabel('Density', fontsize=14)56
plt.axvline(mean_sensitivity, color='darkred', linestyle='dashed', linewidth=2)
plt.text(mean_sensitivity, plt.gca().get_ylim()[1] * 0.95, f'Mean: {mean_sensitivity:.2f}', color='darkred', fontsize=12, ha='right')

# Vertical line for the true value
plt.axvline(true_value, color='blue', linestyle='dashed', linewidth=3)
plt.text(true_value, max(n) * 0.9, f'True: {true_value:.4f}', color='blue', fontsize=12, ha='right')

plt.grid(True)


# # Define text properties for bold and black borders
# text_properties = {'fontsize': 16, 'fontweight': 'bold', 'color': 'orange',
#                     'path_effects': [patheffects.withStroke(linewidth=2, foreground='black')]}

plt.show()
