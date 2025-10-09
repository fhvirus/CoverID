from collections.abc import Callable
import os

import pydub
from tqdm import tqdm

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
