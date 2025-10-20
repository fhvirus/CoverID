import numpy as np
from tqdm import tqdm
import librosa

from utils import load_covers80

def chroma_features(sample):
    sample_rate, sample_width = sample.frame_rate, sample.sample_width * 8
    y = np.array(sample.get_array_of_samples()).astype(np.float32) / (2 ** (sample_width - 1))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate, units='samples')
    hop_length = int((60 / tempo[0]) / 2 * sample_rate)
    feat = librosa.feature.chroma_cens(y=y, sr=sample_rate, hop_length=hop_length)
    return np.array(feat).T

def thresh_and_scale(S, rho, delta):
    n, m = S.shape
    row_thresholds = np.partition(S, int((1 - rho) * m), axis=1)[:, int((1 - rho) * m)]
    col_thresholds = np.partition(S, int((1 - rho) * n), axis=0)[int((1 - rho) * n), :]
    mask = (S >= row_thresholds[:, None]) & (S >= col_thresholds[None, :])
    S[~mask] = delta
    return S

def D_matrix(S, penalty):
    D = np.zeros((S.shape[0]+1, S.shape[1]+1))
    for i in range(1, S.shape[0]+1):
        for j in range(1, S.shape[1]+1):
            D[i, j] = max(0, D[i-1, j-1] + S[i-1, j-1],
                          D[i-1, j]+ S[i-1, j-1] - penalty,
                          D[i, j-1] + S[i-1, j-1] - penalty)
    return D[1:, 1:]

def match_two_chroma(x, y):
    S = np.dot(x, y.T)
    # S[S < 0.75] = -2
    S = thresh_and_scale(S, 0.2, -2)
    D = D_matrix(S, 0.5)
    score = np.max(D)
    return score

def shift_to_match_key(x, y):
    kx = np.sum(x, axis=0)
    ky = np.sum(y, axis=0)
    score = [ np.dot(np.roll(kx, i), ky) for i in range(12) ]
    shift = np.argmax(score)
    return np.roll(x, shift, axis=1)

DATADIR = './coversongs/covers32k/'
name_list, ver_A, ver_B = load_covers80(DATADIR)

chroma_A = {}
chroma_B = {}
for name, song in tqdm(ver_A.items()):
    chroma_A[name] = chroma_features(song)
for name, song in tqdm(ver_B.items()):
    chroma_B[name] = chroma_features(song)

# chroma_A['noise'] = chroma_B['noise'] = np.ones((2000, 12)) / 12

correct_rank = { i: 0 for i in range(len(name_list)) }

for name_a, feat_a in chroma_A.items():
    print(f'{"="*30} Matching {name_a}... {"="*30}')
    matches = []
    for name_b, feat_b in chroma_B.items():
        feat_b = shift_to_match_key(feat_b, feat_a)
        score = match_two_chroma(feat_a, feat_b)
        print(f'{name_b:<50}:{score}')
        matches.append((score, name_b))
    matches = sorted(matches)
    for i in range(len(matches)):
        if matches[i][1] == name_a:
            correct_rank[i] = correct_rank[i] + 1
    # print(f'==> Best match for {name_a}: {best[1]} with score {best[0]:.02f}')

print(correct_rank)
