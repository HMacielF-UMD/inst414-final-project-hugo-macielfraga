"""
Spotify API integration for fetching saved tracks, playlists, and playlist tracks.
Uses the Spotipy library to interact with the Spotify API.
Automatically saves fetched data as CSV files in the data directory.
"""

# Necessary imports
from dotenv import load_dotenv
import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Initialize Spotify client
sp = Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    redirect_uri=os.getenv('REDIRECT_URI'),
    scope=os.getenv('SCOPE')
))

def get_saved_tracks(output_path='data/extracted/spotify_saved_tracks.csv') -> pd.DataFrame:
    """
    Fetches the user's saved tracks and exports them as a CSV file.

    Parameters:
    - output_path (str): Path to save the resulting CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing columns ['name', 'artists', 'uri']
    """
    results = sp.current_user_saved_tracks()
    print("What are the results 1? : " , results)

    tracks = []

    for item in results['items']:
        track = item['track']
        tracks.append({
            'name': track['name'],
            'artists': ', '.join([artist['name'] for artist in track['artists']]),
            'uri': track['uri']
        })

    df = pd.DataFrame(tracks)
    df.to_csv(output_path, index=False)
    print(f"Saved tracks exported to {output_path}")
    return df

def get_users_playlists(output_path='data/extracted/spotify_playlists.csv') -> pd.DataFrame:
    """
    Fetches the user's playlists and exports them as a CSV file.

    Parameters:
    - output_path (str): Path to save the resulting CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing columns ['name', 'uri']
    """
    results = sp.current_user_playlists()
    playlists = []
    print("What are the results 2 ? : " , results)

    for item in results['items']:
        playlists.append({
            'name': item['name'],
            'uri': item['uri']
        })

    df = pd.DataFrame(playlists)
    df.to_csv(output_path, index=False)
    print(f"Playlists exported to {output_path}")
    return df

def get_playlist_tracks(playlist_id='spotify:playlist:0T8NKdyKavL6k9Sr9xdFQB',
                        output_path='data/extracted/spotify_playlist_tracks.csv') -> pd.DataFrame:
    """
    Fetches all tracks from a specific Spotify playlist and exports them as a CSV file.

    Parameters:
    - playlist_id (str): Spotify playlist URI or ID.
    - output_path (str): Path to save the resulting CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing columns ['name', 'artists', 'album', 'uri']
    """
    results = sp.playlist_tracks(playlist_id)

    tracks = []

    for item in results['items']:
        track = item['track']
        if track:
            tracks.append({
                'name': track['name'],
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'uri': track['uri']
            })

    df = pd.DataFrame(tracks)
    df.to_csv(output_path, index=False)
    print(f"Playlist tracks exported to {output_path}")
    return df

if __name__ == "__main__":
    # Example usage
    get_playlist_tracks()
    get_saved_tracks()
    get_users_playlists()


# To-do:
# - Add function to create the new Playlists.