import os
import pandas as pd
import numpy as np


def algo_baseline(csv_clean_name, data_clean_dir, dataset_name):
    csv_clean_name = os.path.join(data_clean_dir, dataset_name, csv_clean_name)
    df_clean = pd.read_csv(csv_clean_name)
    return np.array([[record, np.array([1, 0, 0])] for record in df_clean["record_id"].values])
