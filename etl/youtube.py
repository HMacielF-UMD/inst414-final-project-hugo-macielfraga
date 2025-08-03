"""
This module fetches tracks from Spotify and downloads them from YouTube
using yt-dlp for later audio analysis.
"""

import yt_dlp
from . import spotify
import os
import re
import pandas as pd

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a filename by removing characters that are not allowed in file names.

    Parameters:
    - name (str): The original track + artist name

    Returns:
    - str: A safe file name string
    """
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

def get_tracks_from_user_library() -> pd.DataFrame:
    """
    Fetches the user's saved tracks from Spotify and adds a 'file_name' column.

    Returns:
    - pd.DataFrame: DataFrame with track name, artists, album, uri, and file_name
    """
    df = spotify.get_playlist_tracks()

    # Add a sanitized filename column
    df['file_name'] = df.apply(
        lambda row: sanitize_filename(f"{row['name']} - {row['artists']}"),
        axis=1
    )

    return df

def download_track_from_youtube(filename, search_query, output_dir="data/audio") -> bool:
    """
    Downloads a track from YouTube using yt-dlp.

    Parameters:
    - filename (str): Sanitized file name for saving
    - search_query (str): YouTube search string
    - output_dir (str): Where to save audio

    Returns:
    - bool: True if downloaded, False if skipped or failed
    """
    file_path = os.path.join(output_dir, f"{filename}.mp3")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(file_path):
        print(f"Skipping: {filename} (already exists)")
        return False

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'default_search': 'ytsearch1',
        'outtmpl': f'{output_dir}/{filename}.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])
        print(f"Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def download_tracks_from_user_library(output_dir="data/audio") -> pd.DataFrame:
    """
    Downloads tracks listed in a DataFrame and updates a 'downloaded' column.

    Parameters:
    - df (pd.DataFrame): Track DataFrame with a 'file_name' column
    - output_dir (str): Directory to save downloaded MP3s

    Returns:
    - pd.DataFrame: Updated DataFrame with a 'downloaded' boolean column
    """
    df = get_tracks_from_user_library()
    df['downloaded'] = False

    for idx, row in df.iterrows():
        query = f"{row['name']} {row['artists']} audio"
        success = download_track_from_youtube(row['file_name'], query, output_dir)
        df.at[idx, 'downloaded'] = success

        # Save the updated DataFrame with download status
        df.to_csv("data/extracted/youtube_tracks.csv", index=False)

    print(f"Downloaded {df['downloaded'].sum()} out of {len(df)} tracks.")
    return df

if __name__ == "__main__":
    tracks_df = get_tracks_from_user_library()
    updated_df = download_tracks_from_user_library(tracks_df)
    updated_df.to_csv("data/extracted/youtube_tracks.csv", index=False)
    print("Final track list saved to data/extracted/downloaded_tracks.csv")
