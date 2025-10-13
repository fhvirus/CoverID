from collections.abc import Callable
import os

import pydub
from tqdm import tqdm
import numpy as np
from algorithms import chroma_features, compare_features

def load_covers80(datadir: str):
    """
    Load covers80 dataset from given directory. Paths are now hardcoded.
    """

    LIST1_PATH = 'list1.list'
    LIST2_PATH = 'list2.list'

    try:
        with open(os.path.join(datadir, LIST1_PATH), 'r') as f:
            list1 = f.read().splitlines()
        with open(os.path.join(datadir, LIST2_PATH), 'r') as f:
            list2 = f.read().splitlines()
    except Exception as e:
        print(f'Cannot open song lists: {e}')
        exit(1)

    name_list = [ s.split('/')[0] for s in list1 ]
    assert name_list == [ s.split('/')[0] for s in list2 ], 'Song lists are not identical'

    def load_songs_from_list(song_list: list[str]):
        songs = {}
        print('Loading song list...')
        for song in tqdm(song_list):
            name = song.split('/')[0]
            data = pydub.AudioSegment.from_file(os.path.join(datadir, song) + '.mp3')
            songs.update({name: data})
        return songs

    ver_A = load_songs_from_list(list1)
    ver_B = load_songs_from_list(list2)

    return name_list, ver_A, ver_B


def match_one_song(database: dict[str, pydub.AudioSegment],
                   song: pydub.AudioSegment,
                   compare: Callable[[pydub.AudioSegment, pydub.AudioSegment], float]
                   ) -> tuple[float, str]:
    """choose the song with highest similarity in database"""
    results = [ (compare(data, song), name) for name, data in database.items() ]
    return max(results)

def match_one_song_features(features_list,
                   song: pydub.AudioSegment
                   ) -> tuple[float, str]:
    """choose the song with highest similarity in database"""
    #results = [ (compare_features(features, song), name) for features, name in features_list ]
    results = []
    for features, name in features_list:
        #print(name)  
        song = song.set_channels(1) if song.channels > 1 else song
        # Convert pydub AudioSegment to numpy array
        song_samples = np.array(song.get_array_of_samples()).astype(np.float32)
        # Normalize audio samples to the range [-1, 1]
        song_samples /= np.iinfo(song.array_type).max
        # Get the sample rates
        sr_song = song.frame_rate
        # Compute chroma features for both audio signals
        song_chroma = chroma_features(song_samples, sr_song, hop_time=100, n_fft=1024, variation="none")
        score = compare_features(features, song_chroma)
        results.append((score, name))
    print(results)
    best_match = max(results, key=lambda x: x[0])
    print(best_match)
    score_match, name_match = best_match
    return (score_match, name_match)

def match_all_songs_features(database: dict[str, pydub.AudioSegment],
                    covers: dict[str, pydub.AudioSegment]
                    ) -> tuple[list[str], list[str]]:
    truth_list = []
    matched_list = []
    features_list = []
    for name, data in database.items():
        data = data.set_channels(1) if data.channels > 1 else data
        data_samples = np.array(data.get_array_of_samples()).astype(np.float32)
        data_samples /= np.iinfo(data.array_type).max
        sr = data.frame_rate
        features_list.append((chroma_features(data_samples, sr, hop_time=10, n_fft=1024, variation="none"),name))
    i = 0
    for name, song in covers.items():
        if i >= 5:
            break
        _, matched_name = match_one_song_features(features_list, song)
        truth_list.append(name)
        matched_list.append(matched_name)
        i += 1
        print(f"Cover song: {name}. Matched song: {matched_name}")

    return truth_list, matched_list

def match_all_songs(database: dict[str, pydub.AudioSegment],
                    covers: dict[str, pydub.AudioSegment],
                    compare: Callable[[pydub.AudioSegment, pydub.AudioSegment], float]
                    ) -> tuple[list[str], list[str]]:
    truth_list = []
    matched_list = []
    for name, song in covers.items():
        _, matched_name = match_one_song(database, song, compare)
        truth_list.append(name)
        matched_list.append(matched_name)
    return truth_list, matched_list
