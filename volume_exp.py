import numpy as np
from pydub.utils import make_chunks
from utils import load_covers80

def get_volume(song):
    volumes = np.maximum([ chunk.dBFS for chunk in make_chunks(song, 10) ], -100)
    return np.average(volumes)

DATADIR = './coversongs/covers32k/'
name_list, ver_A, ver_B = load_covers80(DATADIR)

songs = sorted([ (get_volume(song), name) for name, song in ver_A.items() ])

for vol, name in songs:
    print(f'{name},{vol:.02f}')
