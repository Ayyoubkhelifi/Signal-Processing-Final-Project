import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def load_data(data_dir, n_components=100):
    all_data = []
    data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Load and combine data in chunks
    chunk_size = 1000
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            data_file = os.path.join(subfolder_path, 'data.csv')
            if os.path.exists(data_file):
                try:
                    print(f"Loading data from {data_file}")
                    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
                        all_data.append(chunk)
                except Exception as e:
                    print(f"Error reading {data_file}: {e}")
    
    if not all_data:
        raise ValueError("No data files were loaded")
    
    # Combine all data    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape before preprocessing: {combined_data.shape}")
    
    # Handle missing values using SimpleImputer
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(combined_data)
    print(f"Data shape after imputation: {imputed_data.shape}")
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(imputed_data)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
    print(f"Data shape after PCA: {reduced_data.shape}")
    
    return reduced_data

def clean_data(df):
    print("Starting data cleaning...")
    print(f"Initial shape: {df.shape}")
    
    # Convert to numpy array if it's a DataFrame
    if isinstance(df, pd.DataFrame):
        df = df.values
    
    # Replace infinities with large finite numbers
    df = np.nan_to_num(df, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"Shape after cleaning: {df.shape}")
    return df

def normalize_data(df):
    if len(df) == 0:
        raise ValueError("Empty array provided for normalization")
    
    print(f"Normalizing data with shape: {df.shape}")
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df)
    print(f"Normalized data shape: {normalized.shape}")
    return normalized
