"""
Performs unsupervised clustering on song features using KMeans.
"""

import pandas as pd
from sklearn.cluster import KMeans
import os
def run_kmeans_clustering(
    input_csv="data/processed/transformed_data.csv",
    labels_csv="data/provided/mood_relation.csv",
    output_csv="data/outputs/clustered_tracks.csv",
    n_clusters=2
):
    """
    Runs KMeans clustering and appends true mood labels (if available).

    Parameters:
    - input_csv (str): Path to feature CSV
    - labels_csv (str): Path to mood-labeled CSV
    - output_csv (str): Path to save clustered data
    - n_clusters (int): Number of clusters

    Returns:
    - pd.DataFrame: DataFrame with cluster and mood
    """
    if not os.path.exists(input_csv) or not os.path.exists(labels_csv):
        print("Missing input or label file.")
        return None

    df = pd.read_csv(input_csv)
    labels_df = pd.read_csv(labels_csv)

    # Cluster on numeric features
    features = df.select_dtypes(include='number')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)

    # Merge mood labels (optional)
    df = pd.merge(df, labels_df, on=["name", "artists"], how="left")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Clustered data with mood labels saved to {output_csv}")
    return df


if __name__ == "__main__":
    run_kmeans_clustering()
