import sys
sys.path.append('../DATA/')

import numpy as np
import pandas as pd
from dataLoader import *

C_dataset_filename = '../DATA/CSV/CapacityDataset.csv'
U_dataset_filename = '../DATA/CSV/UDataset.csv'
V_dataset_filename = '../DATA/CSV/VDataset.csv'

C_df, U_df, V_df = load_data(C_dataset_filename, U_dataset_filename, V_dataset_filename)

print(C_df)