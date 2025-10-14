import pydub
import numpy as np
import librosa
from numba import njit

def dummy_compare(a: pydub.AudioSegment,
                  b: pydub.AudioSegment) -> float:
    """Given two audio, returns their similarity"""

    num_a = a.frame_count()
    num_b = b.frame_count()

    return abs(num_a - num_b) / num_a

def compare(a: pydub.AudioSegment,
            b: pydub.AudioSegment) -> float:
    """Given two audio, returns their similarity"""

    # Convert pydub AudioSegment to numpy array
    a_samples = np.array(a.get_array_of_samples()).astype(np.float32)
    b_samples = np.array(b.get_array_of_samples()).astype(np.float32)

    # Normalize audio samples to the range [-1, 1]
    a_samples /= np.iinfo(a.array_type).max
    b_samples /= np.iinfo(b.array_type).max

    # Get the sample rates
    sr_a = a.frame_rate
    sr_b = b.frame_rate

    # Compute chroma features for both audio signals
    a_chroma = chroma_features(a_samples, sr_a, hop_time=10, n_fft=2048, variation="none")
    b_chroma = chroma_features(b_samples, sr_b, hop_time=10, n_fft=2048, variation="none")
    print(f"get chroma features: {a_chroma.shape}, {b_chroma.shape}")

    # Compute similarity matrix
    S = similarity_matrix(a_chroma, b_chroma)
    D = D_matrix(S)

    # The similarity score can be defined as the maximum value in the similarity matrix
    similarity_score = np.max(D)

    return similarity_score

def compare_features(og_features,
            cover_features) -> float:
    """Given two audio, returns their similarity"""

    # Compute similarity matrix
    S = similarity_matrix(og_features, cover_features)
    D = D_matrix(S)

    # The similarity score can be defined as the maximum value in the similarity matrix
    similarity_score = np.max(D)
    #print(f"Similarity score: {similarity_score}")

    return similarity_score

from typing import Literal
def chroma_features(x: np.ndarray, sr: int, hop_time:int=10, n_fft:int=2048, variation: Literal["none", "norm", "cens"] = "none", l: int = 1, d: int = 1):
    """
    Compute chroma features from an audio signal. Parameters:
    x : np.ndarray. Audio time series (samples).
    sr : int. Sampling rate of the audio signal.
    hop_time : int. ms.
    n_fft : int. Length of the FFT window.
    variation : {"none", "norm", "cens"}, optional. Type of chroma features to compute:
        - "none": raw chroma,
        - "norm": normalized chroma,
        - "cens": CENS features (requires parameters `l` and `d`).
        Default is "none".
    l : int, optional. Consecutive vectors window size. Used for CENS.
    d : int, optional. Downsampling factor. Used for CENS.
    
    Returns:
    features: np.ndarray. Computed chroma feature matrix.
    """
    hop_length = int(sr * hop_time / 1000)  # Convert hop time from ms to samples
    if variation == "none": # Implement for "simple" version (normal chroma features done in Lab 3)
        # todo
        features = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length, n_fft=n_fft) # Revise and change parameters
    elif variation == "norm":
        # todo
        features = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length, n_fft=n_fft) # Revise and change parameters
    elif variation == "cens":
        # todo
        features_no_dw = librosa.feature.chroma_cens(y=x, sr=sr, hop_length=hop_length, win_len_smooth = l) #, l=l, d=d) # Revise and change parameters
        # Downsampling as librosa does not implement it:
        features = features_no_dw[:, ::d]    
    else:
        raise ValueError(f"Invalid variation '{variation}'. Must be one of ['none', 'norm', 'cens']")
    return features

def similarity_matrix(x_chroma, y_chroma) -> np.ndarray:
    """
    Compute a similarity matrix from feature sequences. P
    x_chroma : np.ndarray. Chroma feature matrix for audio x (shape: (n_features, n_frames_x)).
    y_chroma : np.ndarray. Chroma feature matrix for audio y (shape: (n_features, n_frames_y)).
    Returns:
    D : np.ndarray. Similarity matrix (shape: (n_frames_x, n_frames_y)).
    """
    S = np.dot(x_chroma.T, y_chroma)  # Compute the dot product between feature vectors
    return S

@njit(cache=True, fastmath=True)  
def D_matrix(S):
    D = np.zeros((S.shape[0]+1, S.shape[1]+1))  # D matrix (start with idx 1)
    for i in range(1, S.shape[0]+1):
        for j in range(1, S.shape[1]+1):
            D[i, j] = max(0, D[i-1, j-1] + S[i-1, j-1], D[i-1, j]+ S[i-1, j-1], D[i, j-1] + S[i-1, j-1])
    
    return D[1:, 1:]  # Return the similarity matrix without the extra row and column


def D_matrix_optimal(S):
    n, m = S.shape
    D = np.zeros((n, m), dtype=S.dtype)

    for i in range(1, n):
        diag = D[i-1, :-1] + S[i, 1:]
        up   = D[i-1, :]   + S[i, :]
        left = D[i, :-1]   + S[i, 1:]

        D[i, 1:] = np.maximum(0, np.maximum.reduce([
            diag,
            up[1:],   # D(n-1,m)
            left
        ]))
    return D