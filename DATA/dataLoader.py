import pandas as pd
import numpy as np

def load_data(c_filename, u_filename, v_filename):
    # Read the CSV files
    C_df = pd.read_csv(c_filename, header=None, dtype=np.float64, comment='#')
    U_df = pd.read_csv(u_filename, header=None, dtype=np.float64, comment='#')
    V_df = pd.read_csv(v_filename, header=None, dtype=np.float64, comment='#')

    # Assign column labels
    column_labels = ['AmplitudeX', 'AmplitudeY', 'AmplitudeZ', 'Overetch']
    C_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(201)]
    U_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(201)]
    V_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(201)]

    return C_df, U_df, V_df