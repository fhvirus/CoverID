"""
Beat-synchronous chroma feature extraction and matching functions.

This module contains functions for extracting beat-synchronized chroma features
and performing cover song detection using these features. Beat-synchronous features
can provide better temporal alignment and reduce computational complexity compared
to frame-synchronous approaches.
"""

import pydub
from tqdm import tqdm
import numpy as np
import librosa
from algorithms import compare_features

def extract_beat_chroma(audio_samples, sr, beats_per_frame=4, use_cqt=False):
    """
    Extract beat-synchronized chroma features.
    
    Parameters:
    audio_samples: numpy array of audio samples
    sr: sample rate
    beats_per_frame: number of beats per chroma frame (1/4 beat = 4, 1/8 beat = 8, 1/12 beat = 12)
    use_cqt: whether to use CQT-based chroma (better frequency resolution) or STFT-based
    
    Returns:
    chroma_beats: beat-synchronized chroma features
    """
    # Detect beat positions
    tempo, beats = librosa.beat.beat_track(y=audio_samples, sr=sr, units='time')
    # Convert tempo to scalar if it's an array
    tempo_value = float(tempo) if hasattr(tempo, '__len__') else tempo
    print(f"Detected tempo: {tempo_value:.1f} BPM, {len(beats)} beats")
    
    # Calculate subdivision positions (e.g., 1/4 beats, 1/8 beats, etc.)
    beat_subdivisions = []
    for i in range(len(beats) - 1):
        beat_start = beats[i]
        beat_end = beats[i + 1]
        beat_duration = beat_end - beat_start
        subdivision_duration = beat_duration / beats_per_frame
        
        for j in range(beats_per_frame):
            subdivision_time = beat_start + j * subdivision_duration
            beat_subdivisions.append(subdivision_time)
    
    # Add final subdivisions for the last beat (estimate based on average beat duration)
    if len(beats) > 1:
        avg_beat_duration = np.mean(np.diff(beats))
        final_beat_start = beats[-1]
        for j in range(beats_per_frame):
            subdivision_time = final_beat_start + j * avg_beat_duration / beats_per_frame
            beat_subdivisions.append(subdivision_time)
    
    # Convert times to sample indices
    beat_frames = librosa.time_to_frames(beat_subdivisions, sr=sr, hop_length=512)
    
    # Extract chroma at beat positions using either CQT or STFT
    if use_cqt:
        # CQT provides better frequency resolution but is slower
        chroma = librosa.feature.chroma_cqt(y=audio_samples, sr=sr, hop_length=512)
        print("Using CQT-based chroma features")
    else:
        # STFT is faster but with lower frequency resolution
        chroma = librosa.feature.chroma_stft(y=audio_samples, sr=sr, hop_length=512)
        print("Using STFT-based chroma features")
    
    # Synchronize chroma to beats
    chroma_beats = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    
    expected_frames = int(tempo_value * beats_per_frame * len(audio_samples) / sr / 60)
    actual_frames = chroma_beats.shape[1]
    print(f"Beat-sync chroma shape: {chroma_beats.shape}, expected ~{expected_frames} frames, got {actual_frames}")
    
    # Optional: apply additional processing for better cover song detection
    # Normalize each chroma vector to unit length
    chroma_beats = librosa.util.normalize(chroma_beats, axis=0, norm=2)
    
    return chroma_beats

def match_one_song_features_beat_sync(features_list,
                                      song: pydub.AudioSegment,
                                      beats_per_frame: int = 4
                                      ) -> tuple[float, str]:
    """
    Choose the song with highest similarity using beat-synchronous chroma features.
    
    Parameters:
    features_list: list of (beat_chroma_features, name) tuples from database
    song: pydub.AudioSegment to match
    beats_per_frame: number of beats per chroma frame (4 = quarter beats)
    
    Returns:
    (score, name): tuple of best match score and name
    """
    results = []
    for features, name in features_list:
        song = song.set_channels(1) if song.channels > 1 else song
        # Convert pydub AudioSegment to numpy array
        song_samples = np.array(song.get_array_of_samples()).astype(np.float32)
        # Normalize audio samples to the range [-1, 1]
        song_samples /= np.iinfo(song.array_type).max
        # Get the sample rates
        sr_song = song.frame_rate
        
        # Extract beat-synchronous chroma features for the query song
        song_chroma = extract_beat_chroma(song_samples, sr_song, beats_per_frame=beats_per_frame)
        
        score = compare_features(features, song_chroma)
        results.append((score, name))
    
    print(f"Beat-sync matching results: {[(name, f'{score:.3f}') for score, name in results]}")
    best_match = max(results, key=lambda x: x[0])
    score_match, name_match = best_match
    return (score_match, name_match)

def match_all_songs_features_beat_sync(database: dict[str, pydub.AudioSegment],
                                       covers: dict[str, pydub.AudioSegment],
                                       beats_per_frame: int = 4,
                                       debug_mode=False
                                       ) -> tuple[list[str], list[str]]:
    """
    Match all cover songs to database using beat-synchronous chroma features.
    
    Parameters:
    database: dict of original songs {name: AudioSegment}
    covers: dict of cover songs {name: AudioSegment}
    beats_per_frame: number of beats per chroma frame (4 = quarter beats, 12 = 12th beats)
    debug_mode: whether to enable detailed debugging and visualization
    
    Returns:
    (truth_list, matched_list): tuple of ground truth names and matched names
    """
    from utils import visualize_similarity_analysis  # Import here to avoid circular imports
    
    truth_list = []
    matched_list = []
    features_list = []
    
    print(f'Computing beat-synchronous features dataset (beats_per_frame={beats_per_frame})...')
    for name, data in tqdm(database.items()):
        data = data.set_channels(1) if data.channels > 1 else data
        data_samples = np.array(data.get_array_of_samples()).astype(np.float32)
        data_samples /= np.iinfo(data.array_type).max
        sr = data.frame_rate
        
        # Extract beat-synchronous chroma features
        beat_chroma = extract_beat_chroma(data_samples, sr, beats_per_frame=beats_per_frame)
        features_list.append((beat_chroma, name))
    
    i = 0
    print('Matching covers to original songs using beat-synchronous features...')
    for name, song in tqdm(covers.items()):
        if i >= 15:
            break
        
        # Debug mode: visualize the matching process
        if debug_mode and i < 3:  # Debug first 3 matches
            print(f"\n=== DEBUGGING BEAT-SYNC MATCH {i+1}: {name} ===")
            
            # Find the best match with debugging
            best_score = -1
            best_name = None
            correct_score = None
            correct_features = None
            cover_chroma = None

            # Extract cover features once for all comparisons
            song = song.set_channels(1) if song.channels > 1 else song
            song_samples = np.array(song.get_array_of_samples()).astype(np.float32)
            song_samples /= np.iinfo(song.array_type).max
            sr_song = song.frame_rate
            cover_chroma = extract_beat_chroma(song_samples, sr_song, beats_per_frame=beats_per_frame)

            for db_features, db_name in features_list:
                # Compute similarity using beat-synchronous features
                score = compare_features(db_features, cover_chroma)
                print(f"Beat-sync: {db_name} vs {name}: score = {score:.3f} (shapes: {db_features.shape} vs {cover_chroma.shape})")
                
                # Store correct match info
                if db_name == name:
                    correct_score = score
                    correct_features = db_features
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_name = db_name
            
            # Visualize only correct match and final chosen match
            print(f"\n*** CORRECT BEAT-SYNC MATCH: {name} (score: {correct_score:.3f}) ***")
            if correct_features is not None and cover_chroma is not None:
                visualize_similarity_analysis(f"{name}_BEAT_SYNC", name, correct_features, cover_chroma, save_plots=True)
            
            if best_name != name:
                print(f"*** BEAT-SYNC ALGORITHM CHOSE: {best_name} (score: {best_score:.3f}) ***")
                # Find the chosen match features for visualization
                for db_features, db_name in features_list:
                    if db_name == best_name:
                        visualize_similarity_analysis(f"{best_name}_BEAT_SYNC", name, db_features, cover_chroma, save_plots=True)
                        break
            else:
                print(f"*** BEAT-SYNC ALGORITHM CORRECTLY CHOSE: {best_name} ***")
            
            matched_name = best_name
            print(f"Final beat-sync decision: {name} -> {matched_name} (score: {best_score:.3f})")
            print("="*60)
        else:
            # Normal matching without debugging
            _, matched_name = match_one_song_features_beat_sync(features_list, song, beats_per_frame)
        
        truth_list.append(name)
        matched_list.append(matched_name)
        i += 1
        print(f"Beat-sync: Cover song: {name}. Matched song: {matched_name}")

    return truth_list, matched_list