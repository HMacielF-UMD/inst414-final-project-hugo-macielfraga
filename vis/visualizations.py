"""
This module visualizes outputs from the supervised and unsupervised mood classification analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

# Ensure output directory exists
VIS_DIR = "data/vis"
os.makedirs(VIS_DIR, exist_ok=True)

def visualize_supervised(predictions_csv: str):
    """
    Visualizes supervised model results: confusion matrix and mood prediction counts.

    Parameters:
    - predictions_csv (str): Path to CSV with actual and predicted moods

    Returns:
    - None
    """
    df = pd.read_csv(predictions_csv)
    if 'mood' not in df.columns or 'predicted_mood' not in df.columns:
        print("Missing required columns in predictions CSV.")
        return

    # Confusion Matrix
    cm = confusion_matrix(df['mood'], df['predicted_mood'], labels=['Happy', 'Sad'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Happy', 'Sad'], yticklabels=['Happy', 'Sad'])
    plt.title("Supervised: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "supervised_confusion_matrix.png"))
    plt.close()

    # Count of Predictions
    plt.figure(figsize=(6, 4))
    sns.countplot(x='predicted_mood', data=df)
    plt.title("Supervised: Predicted Mood Counts")
    plt.xlabel("Predicted Mood")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "supervised_prediction_counts.png"))
    plt.close()

def visualize_unsupervised(cluster_csv: str):
    """
    Visualizes unsupervised clustering results: heatmap of clusters by mood and PCA plot.

    Parameters:
    - cluster_csv (str): Path to CSV with cluster labels and actual moods

    Returns:
    - None
    """
    df = pd.read_csv(cluster_csv)
    if 'cluster' not in df.columns or 'mood' not in df.columns:
        print("Missing required columns in cluster CSV.")
        return

    # Cluster Distribution Heatmap
    cluster_counts = df.groupby(['mood', 'cluster']).size().unstack(fill_value=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cluster_counts, annot=True, cmap='Purples')
    plt.title("Unsupervised: Cluster Distribution by Mood")
    plt.xlabel("Cluster")
    plt.ylabel("Mood")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "unsupervised_cluster_heatmap.png"))
    plt.close()

    # PCA 2D Scatter (if enough numeric features)
    features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['cluster'], errors='ignore')
    if features.shape[1] >= 2:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)
        df_pca = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df['cluster']

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='deep')
        plt.title("Unsupervised: PCA of Tracks by Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "unsupervised_pca_scatter.png"))
        plt.close()

def visualize_feature_correlations(features_csv: str):
    """
    Creates a heatmap showing correlations between numeric audio features.

    Parameters:
    - features_csv (str): Path to CSV file containing extracted audio features

    Returns:
    - None
    """
    df = pd.read_csv(features_csv)

    # Select numeric features only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.shape[1] < 2:
        print("Not enough numeric features for correlation heatmap.")
        return

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Audio Features")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "feature_correlation_heatmap.png"))
    plt.close()

def visualize_feature_distributions(features_csv: str):
    """
    Creates multiple plots comparing audio features by mood:
    - MFCC Mean vs Tempo
    - Spectral Centroid vs Tempo
    - Spectral Centroid vs RMS Scaled
    - Chroma Mean vs MFCC Mean
    - ZCR Mean vs RMS Scaled
    - Boxplot of MFCC Mean by Mood
    - Histogram of Tempo by Mood

    Parameters:
    - features_csv (str): Path to CSV with audio features and mood labels

    Returns:
    - None
    """
    df = pd.read_csv(features_csv)

    # 1. MFCC Mean vs Tempo
    if all(col in df.columns for col in ['tempo', 'mfcc_mean', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='tempo', y='mfcc_mean', hue='mood', data=df, alpha=0.7)
        plt.title("MFCC Mean vs Tempo by Mood")
        plt.xlabel("Tempo")
        plt.ylabel("MFCC Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "mfcc_vs_tempo_by_mood.png"))
        plt.close()

    # 2. Spectral Centroid vs Tempo
    if all(col in df.columns for col in ['tempo', 'spectral_centroid_mean', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='tempo', y='spectral_centroid_mean', hue='mood', data=df, alpha=0.7)
        plt.title("Spectral Centroid vs Tempo by Mood")
        plt.xlabel("Tempo")
        plt.ylabel("Spectral Centroid Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "spectral_centroid_vs_tempo_by_mood.png"))
        plt.close()

    # 3. Spectral Centroid vs RMS Scaled
    if all(col in df.columns for col in ['spectral_centroid_mean', 'rms_scaled', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='spectral_centroid_mean', y='rms_scaled', hue='mood', data=df, alpha=0.7)
        plt.title("Spectral Centroid vs RMS Scaled by Mood")
        plt.xlabel("Spectral Centroid Mean")
        plt.ylabel("RMS Scaled")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "spectral_centroid_vs_rms_scaled_by_mood.png"))
        plt.close()

    # 4. Chroma Mean vs MFCC Mean
    if all(col in df.columns for col in ['chroma_mean', 'mfcc_mean', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='chroma_mean', y='mfcc_mean', hue='mood', data=df, alpha=0.7)
        plt.title("Chroma Mean vs MFCC Mean by Mood")
        plt.xlabel("Chroma Mean")
        plt.ylabel("MFCC Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "chroma_vs_mfcc_by_mood.png"))
        plt.close()

    # 5. ZCR Mean vs RMS Scaled
    if all(col in df.columns for col in ['zcr_mean', 'rms_scaled', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='zcr_mean', y='rms_scaled', hue='mood', data=df, alpha=0.7)
        plt.title("ZCR Mean vs RMS Scaled by Mood")
        plt.xlabel("ZCR Mean")
        plt.ylabel("RMS Scaled")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "zcr_vs_rms_scaled_by_mood.png"))
        plt.close()

    # 6. Boxplot: MFCC Mean by Mood
    if all(col in df.columns for col in ['mfcc_mean', 'mood']):
        plt.figure(figsize=(6, 5))
        sns.boxplot(x='mood', y='mfcc_mean', data=df)
        plt.title("MFCC Mean Distribution by Mood")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "boxplot_mfcc_by_mood.png"))
        plt.close()

    # 7. Histogram: Tempo by Mood
    if all(col in df.columns for col in ['tempo', 'mood']):
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='tempo', hue='mood', kde=True, bins=20, element="step")
        plt.title("Tempo Distribution by Mood")
        plt.xlabel("Tempo")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "histogram_tempo_by_mood.png"))
        plt.close()


if __name__ == "__main__":
    visualize_supervised("data/outputs/predicted_moods.csv")
    visualize_unsupervised("data/outputs/clustered_tracks.csv")
    visualize_feature_correlations("data/outputs/predicted_moods.csv")
    visualize_feature_distributions("data/outputs/predicted_moods.csv")

