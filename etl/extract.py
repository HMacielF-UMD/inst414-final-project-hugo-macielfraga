"""
This module extracts audio features from MP3 files using Librosa and
appends them to a DataFrame for further analysis.
"""

import librosa
import os
import pandas as pd

def extract_audio_features(file_path: str) -> dict:
    """
    Extracts basic audio features from a single MP3 file using Librosa.

    Parameters:
    - file_path (str): Path to the MP3 file

    Returns:
    - dict: Dictionary of audio features (tempo, MFCC mean, chroma mean, etc.)
    """
    try:
        y, sr = librosa.load(file_path, sr=None)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        return {
            'tempo': tempo,
            'mfcc_mean': mfcc.mean(),
            'chroma_mean': chroma.mean(),
            'zcr_mean': zcr.mean(),
            'spectral_centroid_mean': spectral_centroid.mean(),
            'rms_mean': rms.mean()
        }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return {
            'tempo': None,
            'mfcc_mean': None,
            'chroma_mean': None,
            'zcr_mean': None,
            'spectral_centroid_mean': None,
            'rms_mean': None
        }

def append_audio_features_to_df(df: pd.DataFrame,
                                mp3_dir="data/audio",
                                output_path="data/extracted/tracks_features.csv") -> pd.DataFrame:
    """
    Appends audio features to a DataFrame and saves it as a CSV.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'file_name' column
    - mp3_dir (str): Directory where MP3 files are stored
    - output_path (str): Path to save the resulting CSV

    Returns:
    - pd.DataFrame: Updated DataFrame with audio features
    """
    feature_columns = ['tempo', 'mfcc_mean', 'chroma_mean', 'zcr_mean', 'spectral_centroid_mean', 'rms_mean']
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        file_path = os.path.join(mp3_dir, f"{row['file_name']}.mp3")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        features = extract_audio_features(file_path)
        for key, value in features.items():
            df.at[idx, key] = value
        print(f"Processed:  {row['file_name']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Audio features saved to: {output_path}")
    return df

if __name__ == "__main__":
    # Example usage
    input_csv = "data/extracted/youtube_tracks.csv"
    df = pd.read_csv(input_csv)
    updated_df = append_audio_features_to_df(df)
    print(updated_df.head())
