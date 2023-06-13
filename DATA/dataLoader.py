import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('CapacityDataset.csv', header=None, dtype=np.float64, comment='#')

# Assign column labels
column_labels = ['AmplitudeX', 'AmplitudeY', 'AmplitudeZ', 'Overetch']
df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(201)]

# Display the DataFrame
print(df)
