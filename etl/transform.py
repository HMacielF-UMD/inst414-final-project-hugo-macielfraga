"""
This module cleans, scales, and prepares audio feature data
for machine learning or analysis.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def add_scaled_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a scaled version of the RMS (energy) column using StandardScaler.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'rms_mean' column

    Returns:
    - pd.DataFrame: Updated DataFrame with 'rms_scaled' column
    """
    if 'rms_mean' in df.columns:
        scaler = StandardScaler()
        df['rms_scaled'] = scaler.fit_transform(df[['rms_mean']].fillna(0))
    return df

def clean_and_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and tidies the audio features DataFrame.

    Parameters:
    - df (pd.DataFrame): Raw DataFrame with audio features

    Returns:
    - pd.DataFrame: Cleaned DataFrame with missing values handled and rounded
    """
    df = df.copy()

    if 'file_name' in df.columns:
        df = df.drop_duplicates(subset='file_name')

    # Fill missing numeric values
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Round for readability
    df[num_cols] = df[num_cols].round(4)

    return df

def transform_features(input_csv="data/extracted/tracks_features.csv",
                       output_csv="data/processed/transformed_data.csv") -> pd.DataFrame:
    """
    Loads raw audio features CSV, tidies the data, adds derived features,
    and exports to a clean transformed dataset.

    Parameters:
    - input_csv (str): Path to the raw extracted audio feature file
    - output_csv (str): Path to save the cleaned/transformed file

    Returns:
    - pd.DataFrame: Transformed DataFrame with additional columns
    """
    if not os.path.exists(input_csv):
        print(f"Input CSV not found: {input_csv}")
        return None

    df = pd.read_csv(input_csv)
    df = clean_and_tidy(df)
    df = add_scaled_energy(df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Transformed dataset saved to: {output_csv}")
    return df

if __name__ == "__main__":
    transform_features()
