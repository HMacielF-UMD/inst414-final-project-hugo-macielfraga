"""
evaluate.py

This module evaluates the performance of both supervised and unsupervised models.
It loads prediction results from CSV files and prints evaluation metrics or distributions.

Functions:
- evaluate_supervised(): Prints precision, recall, and f1-score for predicted vs actual moods.
- evaluate_unsupervised(): Prints cluster distribution by mood for unsupervised clustering.

Usage:
Run this script directly to evaluate both models using files in the data/outputs directory.
"""

import pandas as pd
from sklearn.metrics import classification_report
import os

def evaluate_supervised(predictions_csv: str):
    """
    Evaluates the performance of the supervised model using classification metrics.

    Parameters:
    - predictions_csv (str): Path to the CSV file containing actual and predicted mood labels

    Returns:
    - None
    """
    if not os.path.exists(predictions_csv):
        print(f"File not found: {predictions_csv}")
        return

    df = pd.read_csv(predictions_csv)

    if 'mood' not in df.columns or 'predicted_mood' not in df.columns:
        print("Required columns not found in predictions CSV.")
        return

    print("Supervised Model Evaluation:")
    print(classification_report(df['mood'], df['predicted_mood'], digits=2))


def evaluate_unsupervised(clustering_csv: str):
    """
    Evaluates the clustering results by comparing actual mood labels to predicted clusters.

    Parameters:
    - clustering_csv (str): Path to the CSV file containing actual mood and cluster assignments

    Returns:
    - None
    """
    if not os.path.exists(clustering_csv):
        print(f"File not found: {clustering_csv}")
        return

    df = pd.read_csv(clustering_csv)

    if 'mood' not in df.columns or 'cluster' not in df.columns:
        print("Required columns not found in clustering CSV.")
        return

    print("Unsupervised Clustering Evaluation (Cluster Distribution by Mood):")
    print(df.groupby(['mood', 'cluster']).size().unstack(fill_value=0))


if __name__ == "__main__":
    evaluate_supervised("data/outputs/predicted_moods.csv")
    evaluate_unsupervised("data/outputs/clustered_tracks.csv")
