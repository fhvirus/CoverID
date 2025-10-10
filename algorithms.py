import pydub

def dummy_compare(a: pydub.AudioSegment,
                  b: pydub.AudioSegment) -> float:
    """Given two audio, returns their similarity"""

    num_a = a.frame_count()
    num_b = b.frame_count()

    return abs(num_a - num_b) / num_a


from typing import Literal
def chroma_features(x: np.ndarray, sr: int, variation: Literal["none", "norm", "cens"] = "none", l: int = 1, d: int = 1):
   """
    Compute chroma features from an audio signal. Parameters:
    x : np.ndarray. Audio time series (samples).
    sr : int. Sampling rate of the audio signal.
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
    if variation == "none": # Implement for "simple" version (normal chroma features done in Lab 3)
        # TODO
        features = librosa.feature.chroma_stft(y=x, sr=sr, tuning=0, norm=2,hop_length=hop_length, n_fft=n_fft) # Revise and change parameters
    elif variation == "norm":
        # TODO
    elif variation == "cens":
        # TODO
        features = librosa.feature.chroma_cens(x, sr=sr, hop_length=hop_length)
    else:
        raise ValueError(f"Invalid variation '{variation}'. Must be one of ['none', 'norm', 'cens']")

    return features

