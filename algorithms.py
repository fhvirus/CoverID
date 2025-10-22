import pydub
import numpy as np
import librosa
from numba import njit
from scipy.signal import correlate2d


# Major , minor
krum_schm_profiles = [[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]]
tones = ['C', 'C#', 'D', 'D#', 'E', 'F','F#', 'G', 'G#', 'A', 'A#', 'B']

alpha=[[[1,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0]],
        [[1,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0]]]

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

    # Compute chroma features for both audio signals (optimized for speed)
    a_chroma = chroma_features(a_samples, sr_a, hop_time=50, n_fft=1024, variation="norm")
    b_chroma = chroma_features(b_samples, sr_b, hop_time=50, n_fft=1024, variation="norm")
    print(f"get chroma features: {a_chroma.shape}, {b_chroma.shape}")

    # Compute similarity matrix
    S = similarity_matrix(a_chroma, b_chroma, norm=True)
    D = smith_waterman(S, a=0, b=0.75, k=1, l=1,penalty="affine")

    # The similarity score can be defined as the maximum value in the similarity matrix
    similarity_score = np.max(D)

    return similarity_score

def compare_features(og_features,
            cover_features) -> float:
    """Given two audio, returns their similarity"""

    # Compute similarity matrix
    song_chroma, S, D, similarity_score = chroma_shifting(og_features, cover_features)
    #S = similarity_matrix(og_features, cover_features, norm=True)
    #D = smith_waterman(S, a=0, b=0.5, k=1, l=1,penalty="affine")

    # The similarity score can be defined as the maximum value in the similarity matrix
    #similarity_score = np.max(D)
    #print(f"Similarity score: {similarity_score}")

    return similarity_score

def compare_beat_sync(a: pydub.AudioSegment,
                      b: pydub.AudioSegment,
                      beats_per_frame: int = 4) -> float:
    """
    Compare two audio segments using beat-synchronous chroma features.
    
    Parameters:
    a, b: pydub.AudioSegment objects to compare
    beats_per_frame: number of beats per chroma frame (4 = quarter beats, 12 = 12th beats)
    
    Returns:
    similarity_score: float similarity score
    """
    from beat_chroma import extract_beat_chroma
    
    # Convert pydub AudioSegment to numpy array
    a_samples = np.array(a.get_array_of_samples()).astype(np.float32)
    b_samples = np.array(b.get_array_of_samples()).astype(np.float32)

    # Normalize audio samples to the range [-1, 1]
    a_samples /= np.iinfo(a.array_type).max
    b_samples /= np.iinfo(b.array_type).max

    # Get the sample rates
    sr_a = a.frame_rate
    sr_b = b.frame_rate

    # Extract beat-synchronous chroma features
    a_chroma = extract_beat_chroma(a_samples, sr_a, beats_per_frame=beats_per_frame)
    b_chroma = extract_beat_chroma(b_samples, sr_b, beats_per_frame=beats_per_frame)
    
    print(f"Beat-sync chroma features: {a_chroma.shape}, {b_chroma.shape}")

    # Compute similarity matrix
    S = similarity_matrix(a_chroma, b_chroma, norm=True)
    D = smith_waterman(S, a=0, b=0.75, k=1, l=1,penalty="affine")

    # The similarity score can be defined as the maximum value in the similarity matrix
    similarity_score = np.max(D)

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
        # chroma_cqt doesn't use n_fft parameter
        features = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length) # Revise and change parameters
    elif variation == "cens":
        chroma = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length)
        chroma = chroma / np.max(chroma)
        features = compute_cens(chroma, l=11, d=3)    
    else:
        raise ValueError(f"Invalid variation '{variation}'. Must be one of ['none', 'norm', 'cens']")
    return features

def chroma_features_fast(x: np.ndarray, sr: int, hop_time:int=100, n_fft:int=512):
    """
    Fast version of chroma features with reduced resolution.
    Uses larger hop_time and smaller FFT for speed.
    """
    hop_length = int(sr * hop_time / 1000)  # Convert hop time from ms to samples
    features = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length, n_fft=n_fft)
    return features

def normalize_columns(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    # Evitem divisió per zero:
    norms[norms == 0] = 1
    return mat / norms

def thresh_and_scale(S, rho, delta):
    n, m = S.shape
    row_thresholds = np.partition(S, int((1 - rho) * m), axis=1)[:, int((1 - rho) * m)]
    col_thresholds = np.partition(S, int((1 - rho) * n), axis=0)[int((1 - rho) * n), :]
    mask = (S >= row_thresholds[:, None]) & (S >= col_thresholds[None, :])
    S[~mask] = delta
    return S

def similarity_matrix(x_chroma, y_chroma, norm:True) -> np.ndarray:
    """
    Compute a similarity matrix from feature sequences. P
    x_chroma : np.ndarray. Chroma feature matrix for audio x (shape: (n_features, n_frames_x)).
    y_chroma : np.ndarray. Chroma feature matrix for audio y (shape: (n_features, n_frames_y)).
    Returns:
    D : np.ndarray. Similarity matrix (shape: (n_frames_x, n_frames_y)).
    """
    if norm:
        x_chroma = normalize_columns(x_chroma)
        y_chroma = normalize_columns(y_chroma)
    
    S = np.dot(x_chroma.T, y_chroma)  # Compute the dot product between feature vectors
    #S_smooth = librosa.segment.path_enhance(S, 51, window='hann', n_filters=7) # rev, trying to see somthing
    return thresh_and_scale(S, 0.2, -2)

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

@njit(cache=True, fastmath=True)
def smith_waterman(S, a=1, b=0.5, k=1, l=1,penalty="constant", th=0):
    D = np.zeros((S.shape[0]+1, S.shape[1]+1))  # D matrix (start with idx 1)
    if penalty == "constant":
        wk = b*k
        wl = b*l
    elif penalty == "affine":
        wk = a+b*k
        wl = a+b*l
    elif penalty == "double_affine":
        wk = a+min(k,th)*b+max(0,k-th)*b
        wl = a+min(l,th)*b+max(0,l-th)*b
    else:
         raise ValueError(f"Invalid gap penalty mode: '{penalty}'. Expected 'constant' or 'affine'.")
    for i in range(1, S.shape[0]+1):
        for j in range(1, S.shape[1]+1):
            D[i, j] = max(0,
              D[i-1, j-1] + S[i-1, j-1], # sense el -1??
              D[i-k, j] - wk,
              D[i, j-l] - wl)
    return D[1:, 1:]  # Return the similarity matrix without the extra row and column



def generate_key_profiles(alpha, ks_profiles, s=0.6, n_harmonics=4): # Based on paper "Tonal Description of Polyphonic Audio for Music Content Processing"
    profiles = []

    for j, alpha_matrix in enumerate(alpha):  
        base_profiles = []
        for i in range(12):  
            harmonic_profile = np.zeros(12)
            for j_note in range(12):  
                if alpha_matrix[i][j_note] == 1:
                    for h in range(1, n_harmonics + 1):
                        pitch_class = (j_note * h) % 12
                        contribution = s ** (h - 1)
                        harmonic_profile[pitch_class] += contribution
            final_profile = harmonic_profile * ks_profiles[j]
            base_profiles.append(final_profile)
        profiles.append(base_profiles)

    return profiles

def generate_key_profiles_paper(): # Based on paper "Understanding the Algorithm Behind Audio Key Detection"
    """
    Generate normalized key profiles for all 24 keys (12 major + 12 minor).
    
    The algorithm creates templates for each key by:
    1. Shifting the Krumhansl-Schmuckler profiles for each chromatic step
    2. Normalizing each profile to unit length
    
    Returns:
    profiles: list of 24 normalized key profiles [C major, C# major, ..., B major, C minor, ..., B minor]
    """
    global krum_schm_profiles
    
    profiles = []
    
    # Major key profiles (0-11)
    major_template = np.array(krum_schm_profiles[0])  # Krumhansl-Schmuckler major profile
    for shift in range(12):  # For each chromatic step (C, C#, D, ...)
        # Shift the template to create profile for this key
        shifted_profile = np.roll(major_template, shift)
        # Normalize to unit length
        norm = np.linalg.norm(shifted_profile)
        if norm > 0:
            normalized_profile = shifted_profile / norm
        else:
            normalized_profile = shifted_profile
        profiles.append(normalized_profile)
    
    # Minor key profiles (12-23)  
    minor_template = np.array(krum_schm_profiles[1])  # Krumhansl-Schmuckler minor profile
    for shift in range(12):  # For each chromatic step (Cm, C#m, Dm, ...)
        # Shift the template to create profile for this key
        shifted_profile = np.roll(minor_template, shift)
        # Normalize to unit length
        norm = np.linalg.norm(shifted_profile)
        if norm > 0:
            normalized_profile = shifted_profile / norm
        else:
            normalized_profile = shifted_profile
        profiles.append(normalized_profile)
    
    return profiles

def detect_key_algorithm(chroma_features):
    """
    Key detection algorithm based on "Understanding the Algorithm Behind Audio Key Detection".
    
    Algorithm steps:
    1. Compute average chroma distribution from input features
    2. Normalize the chroma distribution
    3. Generate all 24 key profiles (12 major + 12 minor)
    4. Compute correlation between chroma and each key profile
    5. Return the key with highest correlation
    
    Parameters:
    chroma_features: chroma feature matrix (12 x n_frames)
    
    Returns:
    key_index: detected key index (0-11 for major, 12-23 for minor)
    correlation: correlation coefficient of best match
    """
    # Step 1: Compute average chroma distribution
    chroma_sum = np.mean(chroma_features, axis=1)
    
    # Step 2: Normalize chroma distribution to unit length
    chroma_norm = np.linalg.norm(chroma_sum)
    if chroma_norm > 0:
        chroma_normalized = chroma_sum / chroma_norm
    else:
        chroma_normalized = chroma_sum
    
    # Step 3: Generate all key profiles
    key_profiles = generate_key_profiles_paper()
    
    # Step 4: Find best matching key profile
    max_correlation = -1
    best_key = 0
    
    for key_idx, profile in enumerate(key_profiles):
        # Compute correlation between normalized chroma and key profile
        correlation = np.dot(chroma_normalized, profile)
        
        if correlation > max_correlation:
            max_correlation = correlation
            best_key = key_idx
    
    return best_key, max_correlation

def detect_key(chroma_features, krum_schm=True, Algo=True):
    """
    Main key detection function with multiple algorithm options.
    
    Parameters:
    chroma_features: chroma feature matrix
    krum_schm: use original Krumhansl-Schmuckler method  
    Algo: use algorithm from "Understanding the Algorithm Behind Audio Key Detection"
    
    Returns:
    key_index: detected key index
    """
    global krum_schm_profiles
    global tones
    global alpha

    if Algo:
        # Use the algorithm from "Understanding the Algorithm Behind Audio Key Detection"
        key_index, correlation = detect_key_algorithm(chroma_features)
        print(f"Key detection: index={key_index}, correlation={correlation:.3f}")
        return key_index
        
    elif krum_schm:
        # Original implementation with alpha matrices
        chroma_sum = np.mean(chroma_features, axis=1) 
        max_r = 0
        key = 0
        new_profiles = generate_key_profiles(alpha, krum_schm_profiles) 
        for j, elem in enumerate(new_profiles):
            for i, tonic in enumerate(tones):
                profile = np.roll(elem, i)
                r = np.corrcoef(chroma_sum, profile)[0, 1]
                if not np.isnan(r) and r > max_r:
                    max_r = r
                    key = i # + j*12 # Assign: C=1, C#=2, D=3.... Cm=13, C#m=14, Dm=15 .... Bm=24
        return key
    else:
        # Simple method: return index of maximum chroma bin
        chroma_sum = np.mean(chroma_features, axis=1)
        key = np.argmax(chroma_sum)
        return key

def get_key_name_from_index(key_index):
    """
    Convert key index to human-readable key name.
    
    Parameters:
    key_index: 0-23 (0-11 for major keys, 12-23 for minor keys)
    
    Returns:
    key_name: string like "C major", "F# minor", etc.
    """
    global tones
    
    if key_index < 12:
        # Major key
        root = tones[key_index]
        return f"{root} major"
    else:
        # Minor key
        root = tones[key_index - 12]
        return f"{root} minor"


def transpose_chroma(chroma_features, new_key: str = "C", use_algorithm=True): 
    """
    Transpose chroma features to a target key.
    
    Parameters:
    chroma_features: input chroma feature matrix
    new_key: target key name (e.g., "C", "F#", "Am", "Bbm")
    use_algorithm: whether to use the paper's algorithm for key detection
    
    Returns:
    transposed: transposed chroma features
    """
    global tones
    
    # Detect original key
    if use_algorithm:
        original_key_index = detect_key(chroma_features, krum_schm=False, Algo=True)
        original_key_name = get_key_name_from_index(original_key_index)
        print(f"Detected original key: {original_key_name}")
    else:
        original_key_index = detect_key(chroma_features, krum_schm=True, Algo=False)
    
    # Parse target key
    new_key = new_key.strip()
    is_minor = False
    
    # Handle minor key notation
    if new_key.endswith('m') or new_key.endswith('min') or new_key.endswith('minor'):
        is_minor = True
        # Remove minor indicators
        new_key = new_key.replace('minor', '').replace('min', '').replace('m', '').strip()
    
    # Handle alternative notations for sharps/flats
    key_alternatives = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }
    if new_key in key_alternatives:
        new_key = key_alternatives[new_key]
    
    # Error if new_key selected is not valid
    if new_key not in tones:
        raise ValueError(f"Unknown key selected: {new_key}. Valid keys: {tones}")
    
    new_key_index = tones.index(new_key)
    if is_minor:
        new_key_index += 12  # Offset for minor keys
    
    # Calculate the shift needed
    original_root = original_key_index % 12  # Get root note (ignore major/minor)
    target_root = new_key_index % 12         # Get target root note
    shift = (target_root - original_root) % 12
    
    # Apply circular rotation to transpose
    transposed = np.roll(chroma_features, shift, axis=0)
    
    if use_algorithm:
        target_key_name = get_key_name_from_index(new_key_index)
        print(f"Transposed to: {target_key_name} (shift: {shift} semitones)")
    
    return transposed

def shifting(x, y):
    kx = np.sum(x, axis=1)
    ky = np.sum(y, axis=1)
    score = [ np.dot(np.roll(ky, i), kx) for i in range(12) ]
    shift = np.argmax(score)
    return np.roll(y,shift,axis=1)

def chroma_shifting(original_features, cover_features):
    max_score = -np.inf
    best_shift = 0
    best_matrix = []
    best_s = []
    best_shifted_cover = None

    for shift in range(12):
        shifted = np.roll(cover_features, shift, axis=0)
        s = similarity_matrix(original_features, shifted, norm=True)
        d = smith_waterman(s,b=0.75, k=1)
        score = np.max(d)
        if score > max_score:
            max_score = score
            best_shift = shift
            best_shifted_cover = shifted
            best_matrix = d
            best_s = s

    return best_shifted_cover, best_s, best_matrix, max_score

## Chroma extracting functions:
def quantize_chroma(chroma):
    """
    Quantization based on Müller's book
    """
    q = np.zeros_like(chroma, dtype=int)

    thresholds = [0.05, 0.1, 0.2, 0.4]  # Log-like thresholds
    for i, thresh in enumerate(thresholds, start=1):
        q[chroma >= thresh] = i

    return q

def smooth_downsample_normalize(q_chroma, l, d):
    """
    Suavitza cada fila (component chroma) amb una finestra de longitud l (Hann).
    """
    # Smoothing:
    window = np.hanning(l)
    window /= window.sum()  # normalitzar

    smoothed = np.zeros_like(q_chroma, dtype=float)
    for i in range(12):
        smoothed[i] = np.convolve(q_chroma[i], window, mode='same')
    # Downsampling
    downsampled = smoothed[:, ::d]
    # Normalization
    norms = np.linalg.norm(downsampled, axis=0, keepdims=True)
    norms[norms == 0] = 1  # evitar divisions per zero
    normalized = downsampled / norms
    return normalized

def compute_cens(chroma, l=41, d=10):
    """
    Compute CENS features from chroma features
    
    Parameters:
        chroma: np.ndarray (12, T)
        l: int, smoothing window size (in frames)
        d: int, downsampling factor

    Returns:
        cens_features: np.ndarray (12, T//d)
    """
    q = quantize_chroma(chroma)
    cens_features = smooth_downsample_normalize(q,l,d)
    return cens_features
