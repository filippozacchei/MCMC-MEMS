import matplotlib.pyplot as plt
import numpy as np

# Sample data
time = np.linspace(0, 2.5, 100)  # Replace with your actual time values
real_data = np.sin(4 * np.pi * time)  # Replace with your actual real_data values
surrogate_data = real_data + np.random.normal(0, 0.2, len(time))  # Replace with your actual surrogate_data values

# Plot parameters
amplitudeX = 70  # example value, replace with your actual data
periodX = 1.46  # example value, replace with your actual data
overetch = 0.49  # example value, replace with your actual data

# Create the plot
plt.figure(figsize=(8, 6))  # Set the figure size as desired
plt.plot(time, real_data, 'r-', label='Real')  # Plot the real data
plt.plot(time, surrogate_data, 'b-', label='Surrogate')  # Plot the surrogate data

# Labeling the plot
plt.title(f'AmplitudeX = {amplitudeX}m/s; PeriodX = {periodX} ms, Overetch = {overetch} micron')
plt.xlabel('Time [ms]')
plt.ylabel('ΔC [fF/s]')

# Show the legend
plt.legend()

# Display the plot
plt.show()
