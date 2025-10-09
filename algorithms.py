import pydub

def dummy_compare(a: pydub.AudioSegment,
                  b: pydub.AudioSegment) -> float:
    """Given two audio, returns their similarity"""

    num_a = a.frame_count()
    num_b = b.frame_count()

    return abs(num_a - num_b) / num_a
