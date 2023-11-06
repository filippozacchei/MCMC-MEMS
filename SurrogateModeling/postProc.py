import json
import tensorflow as tf
from training import *
from plotting import *

# CONFIGURATION FILE
CONFIGURATION_FILE = './Config_files/config_VoltageSignal.json'

def main():
    try:
        config = parse_config(CONFIGURATION_FILE)  

        # Data preparation
        C_df, dC_df = load_data_derivative(config)
        X_train, X_test, y_train, y_test = split_data(C_df, config)

        # Flatten the output and concatenate time column
        time = np.arange(0, config['TIME_FINAL'], config['TIME_INTERVAL'])
        X_train_rep, y_train_rep = stack_data(X_train, y_train, time)
        X_test_rep, y_test_rep = stack_data(X_test, y_test, time)

        # Data shuffling and scaling
        X_train_rep, y_train_rep = shuffle_data(X_train_rep, y_train_rep)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train_rep, X_test_rep)  

        # Initialize feature_labels and feature_ticks as lists
        feature_labels=[None, None, None]
        feature_ticks = [None, None, None]
        feature_labels[0] = r'Overetch [$\mu$m]'
        feature_labels[1] = r'Offset [$\mu$m]'
        feature_labels[2] = r'Thickness [$\mu$m]'
        feature_ticks[0] = np.arange(0.1, 0.5, 0.1)
        feature_ticks[1] = np.arange(-0.5, 0.5, 0.25)
        feature_ticks[2] = np.arange(29.0, 31.0, 1.0)
        plot_train_test(X_train, X_test,feature_labels,feature_ticks)
        print(X_train)

        model = None
        # Train Model if needed
        if config['TRAINING']:
            model = train(CONFIGURATION_FILE)
        else:
            model = tf.keras.models.load_model(config["MODEL_PATH"])

        # Evaluate and Plot
        plot_predictions(model, X_test_scaled, y_test, time, max_plots=20)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
# # Set figure size and resolution
# plt.figure(figsize=(8, 6), dpi=300)

# # Define colors and line styles
# real_color = 'red'
# surrogate_color = 'blue'
# line_width = 1.5

# etch = np.linspace(0.1,0.4,5)
# offset = np.linspace(-0.4,0.4,5)
# thickness = np.linspace(29.0,31.0,4)

# errors = np.zeros((5,5))

# from scipy.stats import pearsonr

# # Calculate and store errors
# for i in range(25):
#     out = y_pred[i, :]
#     exa = 1e-6*y_test[i, :]
#     err = np.max(np.linalg.norm(out - exa, 1)) / np.linalg.norm(exa, 1)
#     row = int(i / 5)
#     col = i % 5
#     print(row,col)
#     errors[row, col] = err

# print(errors)
# # print(np.min(np.min(errors)))

# # Create a heatmap of the errors
# plt.figure(figsize=(10, 6))
# cax = plt.matshow(errors, cmap='viridis')  # You can choose a different colormap if needed
# plt.colorbar(cax)
# plt.xticks(np.arange(5), [f'{e:.2f}um' for e in etch])
# plt.yticks(np.arange(5), [f'{a:.2f}um' for a in np.flip(offset)])
# # plt.yticks(np.arange(11), [f'{t * 1e3:.2f} ms' for t in np.flip(Tx)])
# # plt.xlabel('Overetch (µm)')
# plt.ylabel(r'Offset [${\mu}$m]')
# # plt.ylabel('Acceleraion Amplitude')
# plt.xlabel(r'Overetch [${\mu}$m]')
# plt.title('Relative Error')
# plt.gca().xaxis.tick_bottom()  # Move x-axis ticks to the bottom
# plt.gca().yaxis.tick_left()  # Move y-axis ticks to the left
# plt.gca().xaxis.set_label_position('bottom')  # Set x-axis label position
# plt.gca().yaxis.set_label_position('left')  # Set y-axis label position

# # # Save the figure as a high-quality image for presentation
# # plt.savefig('relative_l1_error_heatmap.png', dpi=300, bbox_inches='tight')

# # # Display the plot
# plt.show()


# plt.figure(figsize=(12, 8))  # Increase size for better visibility

# # Use a clear colormap
# cax = plt.matshow(errors, cmap='plasma')  # 'plasma' colormap offers clear color differences and is perceptually uniform

# cb = plt.colorbar(cax, shrink=0.8, aspect=5)  # Adjust colorbar size
# cb.set_label('Relative Error', rotation=270, labelpad=20, fontsize=14)  # Label colorbar

# # Adjust font size and labels for clarity
# plt.xticks(np.arange(len(etch)), [f'{e:.2f}µm' for e in etch], fontsize=12)
# plt.yticks(np.arange(len(offset)), [f'{a:.2f}µm' for a in np.flip(offset)], fontsize=12)

# plt.xlabel(r'Overetch [${\mu}$m]', fontsize=16)
# plt.ylabel(r'Offset [${\mu}$m]', fontsize=16)
# plt.title('Relative Error vs Overetch and Offset', fontsize=18, pad=20)  # Increase padding to ensure title doesn't overlap with plot

# # Ensure ticks and labels are on the outside and positioned appropriately
# plt.tick_params(direction='out', length=6, width=2, colors='k', which='major', grid_color='k', grid_alpha=0.5)
# plt.gca().xaxis.tick_bottom()
# plt.gca().yaxis.tick_left()
# plt.gca().xaxis.set_label_position('bottom')
# plt.gca().yaxis.set_label_position('left')

# # Add grid lines for better visibility
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.tight_layout()  # Ensure all elements fit within the figure area
# plt.show()


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
#     plt.show()




