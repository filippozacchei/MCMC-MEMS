import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the data from the CSV file
file_path = './noise.csv'  # Update with your file path
df = pd.read_csv(file_path)
print(df)
plt.figure(figsize=(10, 6))

# Plot mean values
plt.errorbar(df['noise'], df['mean'], 
             yerr=[df['mean'] - df['ci_lower'], df['ci_upper'] - df['mean']], 
             fmt='o', ecolor='orange', capsize=5, label='CI')

# Highlight true value
plt.axhline(y=df['x_true'][0], color='r', linestyle='--', label='Real')
plt.axhline(y=df['x_true'][6], color='b', linestyle='--', label='Real')

plt.xlabel('Noise Level')
plt.ylabel('Overetch [micron]')
plt.title('Estimated Means with Confidence Intervals at Different Noise Levels')
plt.legend()
plt.show()
