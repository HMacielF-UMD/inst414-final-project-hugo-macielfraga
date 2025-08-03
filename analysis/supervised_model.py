"""
Trains a supervised classification model to predict song mood.
Uses labeled mood data and extracted audio features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train_classifier(
    features_csv="data/processed/transformed_data.csv",
    labels_csv="data/provided/mood_relation.csv",
    output_csv="data/outputs/predicted_moods.csv"
):
    """
    Trains a Random Forest model to predict mood and saves predictions.

    Parameters:
    - features_csv (str): Path to CSV with audio features
    - labels_csv (str): Path to CSV with song name, artist, and mood
    - output_csv (str): Path to save predictions and mood comparison

    Returns:
    - pd.DataFrame: DataFrame with predictions and actual labels
    """
    if not os.path.exists(features_csv) or not os.path.exists(labels_csv):
        print("Missing input files.")
        return None

    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv)

    # Merge on name and artists
    df = pd.merge(features_df, labels_df, on=["name", "artists"], how="inner")

    if "mood" not in df.columns:
        print("Mood label column missing after merge.")
        return None

    # Define X and y
    X = df.select_dtypes(include="number")
    y = df["mood"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    df["predicted_mood"] = clf.predict(X)

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("Model training complete. Predictions saved.")
    print(classification_report(y_test, clf.predict(X_test)))

    return df

if __name__ == "__main__":
    train_classifier()
