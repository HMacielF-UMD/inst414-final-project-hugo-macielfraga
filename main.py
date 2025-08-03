"""
Main script to run the full music mood classification pipeline.
"""

import etl.extract as extract
import etl.transform as transform
import etl.spotify as spotify
import etl.youtube as youtube
import analysis.supervised_model as supervised_model
import analysis.unsupervised_model as unsupervised_model
import analysis.evaluate as evaluate
import vis.visualizations as visualizations
import pandas as pd

def main():
    print("=== STEP A: Fetching data from Spotify ===")
    spotify.get_playlist_tracks()

    print("=== STEP B: Downloading songs from YouTube ===")
    df = youtube.download_tracks_from_user_library()

    print("=== STEP C: Extracting audio features ===")
    extract.append_audio_features_to_df(df)

    print("=== STEP C.1: Transforming and tidying audio features ===")
    transform.transform_features()

    print("=== STEP D: Running supervised model ===")
    supervised_model.train_classifier()

    print("=== STEP D.1: Running unsupervised model ===")
    unsupervised_model.run_kmeans_clustering()

    print("=== STEP E: Evaluating models ===")
    evaluate.evaluate_supervised("data/outputs/predicted_moods.csv")
    evaluate.evaluate_unsupervised("data/outputs/clustered_tracks.csv")

    print("=== STEP F: Visualizing outputs ===")
    visualizations.visualize_feature_correlations("data/outputs/audio_features.csv")

if __name__ == "__main__":
    main()
