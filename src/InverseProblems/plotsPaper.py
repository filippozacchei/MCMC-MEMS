import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import patheffects, gridspec, ticker
from scipy.stats import gaussian_kde
from tensorflow.keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D

# Ensure your custom modules are in the path
sys.path.append('../utils/')
sys.path.append('../SurrogateModeling/')

from preprocessing import preprocessing
from model import NN_Model

def load_samples():
    sample_names = ["_"+str(i) for i in range(10)]
    samples = []
    for name in sample_names:
        try:
            A = np.load(f'./samples{name}.npy')
            samples.append(A[:,0:-1:20])
        except FileNotFoundError:
            print(f"Sample {name} not found.")
    return samples

def plot_histograms(samples, true_params):

    num_params = samples.shape[0]
    param_names = ['Overetch', 'Offset', 'Thickness']

    sigma = [0.4,1,2]
    avgi = [0.3,0.0,30.0]
    sim_avg = ['Oavg','Uavg','Tavg']
    
    for i in range(num_params):
        plt.figure()
        n, bins, _ = plt.hist(samples[i, :], bins=50, density=True, color='skyblue', edgecolor='black')
        mean_value = np.mean(samples[i, :])
        ci_lower, ci_upper = np.percentile(samples[i, :], [5, 95])

        plt.axvline(true_params[i], color='darkred', linestyle='dashed', linewidth=3, label='True')
        plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=3, label='Mean')
        plt.fill_betweenx([0, max(n)], ci_lower, ci_upper, color='gray', alpha=0.35, label='95% CI')

        ratio_lower = (ci_lower-avgi[i]) / sigma[i]
        ratio_upper = (ci_upper-avgi[i]) / sigma[i]
        sym_lower = ""
        sym_upper = ""
        if ratio_lower>0:
            sym_lower = "+"
        if ratio_upper>0:
            sym_upper = "+"
        s = ['O','U','T']
        plt.xticks([ci_lower,ci_upper],[sim_avg[i]+sym_lower+f'{ratio_lower:.4f} $\sigma_'+s[i]+'$',sim_avg[i]+sym_upper+f'{ratio_upper:.4f} $\sigma_'+s[i]+'$'])

        plt.xlabel(param_names[i], fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_scatter(sample):
    print("ciao")
    num_params = sample.shape[0]
    param_names = ['Overetch', 'Offset', 'Thickness']
    sigma=[0.4,1,2]
    avgi = [0.3,0.0,30.0]
    sim_avg = [r'$O_{avg}+$',r'$U_{avg}+$',r'$T_{avg}+$']
    for i in range(num_params):
        for j in range(i + 1, num_params):
            plt.figure(figsize=(12,6))
            plt.scatter(sample[i, :], sample[j, :], alpha=0.05, color='steelblue')
            plt.xlabel(param_names[i], fontsize=14)
            plt.ylabel(param_names[j], fontsize=14)
            s = ['O','U','T']
            ci_lower, ci_upper = np.percentile(sample[i, :], [5, 95])
            ratio_lower = abs(ci_lower-avgi[i]) / sigma[i]
            ratio_upper = abs(ci_upper-avgi[i]) / sigma[i]
            plt.xticks([ci_lower,ci_upper],[sim_avg[i]+f'{ratio_lower:.4f} $\sigma_'+s[i]+'$',sim_avg[i]+f'{ratio_upper:.4f} $\sigma_'+s[i]+'$'])    
            ci_lower, ci_upper = np.percentile(sample[j, :], [5, 95])
            ratio_lower = abs(ci_lower-avgi[j]) / sigma[j]
            ratio_upper = abs(ci_upper-avgi[j]) / sigma[j]
            plt.yticks([ci_lower,ci_upper],[sim_avg[j]+f'{ratio_lower:.4f} $\sigma_'+s[j]+'$',sim_avg[j]+f'{ratio_upper:.4f} $\sigma_'+s[j]+'$'])
            plt.grid(True)
            plt.show()

def plot_density_scatter(sample, true_values, sigma_values):
    x, y = sample[0, :], sample[1, :]
    z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(4, 4)

    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_scatter.scatter(x, y, c=z, alpha=0.5)
    ax_scatter.scatter(true_values[0],true_values[1],marker='*',s=200,color='white',edgecolors='black')

    ax_scatter.set_xlabel('Overetch (µm)')
    ax_scatter.set_ylabel('Offset (µm)')

    # Set custom tick locator and formatter
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax_scatter.xaxis.set_major_formatter(formatter)
    ax_scatter.yaxis.set_major_formatter(formatter)

    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
    ax_histx.hist(x, bins=50, density=True, alpha=0.5)
    ax_histy.hist(y, bins=50, orientation='horizontal', density=True, alpha=0.5)

        # Remove ticks
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])

    # Add a line and text for width
    y_min, y_max = ax_scatter.get_ylim()
    line_y = y_min - (y_max - y_min) / 100  # Position the line at 10% of the y-axis height
    ax_scatter.hlines(line_y, x.min(), x.max(), colors='black', linestyles='dashed', lw=2)
    text_x = x.min() + (x.max() - x.min()) / 2  # Center the text
    ratio = (x.max() - x.min())/0.4
    ax_scatter.text(text_x, line_y, f'Width: {ratio:.4f} $\sigma_O$', ha='center', va='bottom')

    # Add a line and text for width
    x_min, x_max = ax_scatter.get_xlim()
    line_x = x_min + (x_max - x_min) / 1000  # Position the line at 10% of the y-axis height
    ax_scatter.vlines(line_x*0.9994, y.min(), y.max(), colors='black', linestyles='dashed', lw=2)
    text_y = y.min() + (y.max() - y.min()) / 2  # Center the text
    ratio = (y.max() - y.min())/1
    ax_scatter.text(line_x, text_y, f'Width: {ratio:.4f} $\sigma_U$', ha='left', va='center', rotation=90)

    ax_histx.axis('off')
    ax_histy.axis('off')

    plt.show()

def plot_sensitivity_boxplot(samples, true_values, model_path, config_file):
    model = NN_Model()
    model.load_model(model_path)

    data_processor = preprocessing(config_file)
    data_processor.process()

    S_avg = model.predict(data_processor.scaler.transform([[0.3, 0.0, 30.0]]))

    sensitivities = []
    median_sensitivities = []
    for sample in samples:
        sample_transposed = sample.T
        sample_transposed = data_processor.scaler.transform(sample_transposed)
        predicted_sensitivities = model.predict(sample_transposed).flatten() / S_avg[0, 0]

        sensitivities.append(predicted_sensitivities)
        median_sensitivities.append(np.mean(predicted_sensitivities))

    # Sort the data by true values for better visualization
    sorted_indices = np.argsort(true_values)
    sorted_true_values = np.array(true_values/ S_avg[0, 0])[sorted_indices]
    sorted_median_sensitivities = np.array(median_sensitivities)[sorted_indices]

    # Create a combined scatter and boxplot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Define colors and styles
    scatter_color = 'dodgerblue'
    box_colors = {'boxes': 'darkblue', 'whiskers': 'black', 'medians': 'red', 'caps': 'black'}
    line_style = {'linestyle': '--', 'linewidth': 1.5, 'color': 'gray'}

    # Scatter plot for median sensitivities
    ax1.scatter(sorted_true_values, sorted_median_sensitivities, color=scatter_color, edgecolor='k', zorder=2, label='Mean Predicted Sensitivity')

    # Adding boxplots for each point
    # Adding boxplots for each point
    sample_labels = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'L']
    for i, true_value in enumerate(true_values):
        ax1.boxplot(sensitivities[i], positions=[true_value]/ S_avg[0, 0], widths=0.02, patch_artist=True, showfliers=False, zorder=1, boxprops=dict(facecolor='lightblue', color=box_colors['boxes']), medianprops=dict(color=box_colors['medians']), whiskerprops=dict(color=box_colors['whiskers']), capprops=dict(color=box_colors['caps']))
        ax1.text(true_value/ S_avg[0, 0]-(-1)**i*0.02, np.mean(sensitivities[i])+0.001, sample_labels[i], horizontalalignment='center', verticalalignment='bottom', fontsize=12)
    # Line of equality
    # Line of equality
    min_val = min(min(true_values), min(median_sensitivities))
    max_val = max(max(true_values), max(median_sensitivities))
    ax1.plot([min_val, max_val], [min_val, max_val], **line_style, label='Line of Equality')

    # Enhancements for publication quality
    ax1.set_xlabel('Reference S', fontsize=12)
    ax1.set_ylabel('Predicted S', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=10)

    # Ticks and labels
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xticks([])

    plt.tight_layout()
    plt.show()


def plot_sensitivity_histogram(sample, true_value, model, scaler, title=None):

    S_avg = 4.44
    sample_transposed = sample.T
    sample_transposed = scaler.transform(sample_transposed)
    predicted_sensitivities = model.predict(sample_transposed).flatten()/S_avg

    mean_sensitivity = np.mean(predicted_sensitivities)
    ci_lower, ci_upper = np.percentile(predicted_sensitivities, [5, 95])

    plt.figure()
    n, bins, _ = plt.hist(predicted_sensitivities, bins=50, density=True, color='coral', edgecolor='black')
    plt.axvline(mean_sensitivity, color='darkred', linestyle='dashed', linewidth=2, label = 'Mean')
    plt.axvline(true_value/S_avg, color='blue', linestyle='dashed', linewidth=3, label = 'True')
    plt.fill_betweenx([0, max(n)], ci_lower, ci_upper, color='gray', alpha=0.35, label='95% CI')
    plt.xlabel('Sensitivity',fontsize=14)
    plt.ylabel('Density',fontsize=14)
    plt.legend()
    plt.grid(True,axis='y')
    plt.show()

def plot_costs():
    # Improved style settings
    plt.style.use('seaborn-whitegrid')  # Use a clean and elegant style
    plt.rcParams.update({'font.size': 12, 'axes.labelweight': 'bold', 'figure.figsize': (8, 6)})

    # Define the data
    categories = ['Offline Cost', 'Online Cost']
    FEM_costs = [0, 100000]  # Adjusted to 0.1 for log scale
    ANN_costs = [860, 120]

    x = np.arange(len(categories))  # the label locations
    width = 0.4  # the width of the bars for better clarity

    fig, ax = plt.subplots()

    # Elegant color palette
    colors = ['#1f77b4', '#ff7f0e']

    # Plotting with refined colors
    rects1 = ax.bar(x - width/2, FEM_costs, width, label='FEM Model', color=colors[0], edgecolor='black')
    rects2 = ax.bar(x + width/2, ANN_costs, width, label='ANN Model', color=colors[1], edgecolor='black')

    # Labeling
    ax.set_ylabel('Cost (minutes)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend(frameon=True, shadow=True, borderpad=1)

    # Logarithmic scale and grid
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)  # Add grid lines for both major and minor ticks

    # Adding a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 6),  # 6 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


# Main Function
def main():
    # Load data
    CONFIGURATION_FILE = '../SurrogateModeling/Config_files/config_VoltageSignal.json'
    data_processor = preprocessing(CONFIGURATION_FILE)
    data_processor.process()
    CONFIGURATION_FILE2 = '../SurrogateModeling/Config_files/config_VoltageSignal_st.json'
    data_processor2 = preprocessing(CONFIGURATION_FILE2)
    data_processor2.process()
    X_values, y_values = data_processor.X_test, data_processor.y_test

    true_values = X_values[0:100:10]

    samples = load_samples()
    sample=samples[0]
    plot_histograms(samples, true_values)
    plot_scatter(samples, true_values)

    for sample, true_value in zip(samples, true_values):
        plot_density_scatter(sample, true_value, sigma_values=(0.2, 0.5))  # Adjust sigma values as needed

    SENSITIVITY_MODEL_PATH = '../SurrogateModeling/Saved_models/model_sensitivity.h5'
    SENSITIVITY_CONFIG_FILE = '../SurrogateModeling/Config_files/config_sensitivity.json'
    
    data_processor = preprocessing(SENSITIVITY_CONFIG_FILE)
    data_processor.process()

    CONFIGURATION_FILE3 = '../SurrogateModeling/Config_files/config_sensitivity_st_reference.json'
    data_processor3 = preprocessing(CONFIGURATION_FILE3)
    data_processor3.process()

    model = NN_Model()
    model.load_model(SENSITIVITY_MODEL_PATH)

    true_values = data_processor.y_test[0:100:10]
    x_values = data_processor.X_test
    # Columns to be swapped
    i, j = 1,2
    x_values[:, [i, j]] = x_values[:, [j, i]]


    plot_sensitivity_histogram(samples, num_values, SENSITIVITY_MODEL_PATH, SENSITIVITY_CONFIG_FILE, x_values, title='Coventor')

    plot_sensitivity_histogram(samples, true_values, SENSITIVITY_MODEL_PATH, SENSITIVITY_CONFIG_FILE, x_values, title='ST reference')

    plot_sensitivity_boxplot(samples, true_values, SENSITIVITY_MODEL_PATH, SENSITIVITY_CONFIG_FILE)



if __name__ == "__main__":
    main()
