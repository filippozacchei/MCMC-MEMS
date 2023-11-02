import pandas as pd
import numpy as np

def load_data(c_filename):
    # Read the CSV files
    C_df = pd.read_csv(c_filename, dtype=str, comment='#', float_precision='high')
    # U_df = pd.read_csv(u_filename, dtype=str, comment='#', float_precision='high')
    # V_df = pd.read_csv(v_filename, dtype=str, comment='#', float_precision='high')

    C_df = C_df.astype(np.float64)
    # U_df = U_df.astype(np.float64)
    # V_df = V_df.astype(np.float64)

    # Assign column labels
    column_labels = ['Overetch', 'Offset', 'Thickness', 'ts', 'tf']
    C_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(151)]
    # U_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(251)]
    # V_df.columns = column_labels + [f'Time={0+1E-2*i:.2f}ms' for i in range(251)]
    
    return C_df