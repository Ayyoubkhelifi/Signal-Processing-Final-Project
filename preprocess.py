import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(data_dir):
    all_data = []
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file_name)
                    try:
                        data = pd.read_csv(file_path)
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return pd.concat(all_data, ignore_index=True)

def clean_data(df):
    # Implement any data cleaning (e.g., remove NaN, etc.)
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# Example of usage
# data = load_data('data/room_3')
# cleaned_data = clean_data(data)
# normalized_data = normalize_data(cleaned_data)
