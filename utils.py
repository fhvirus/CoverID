from collections.abc import Callable
import os

import pydub
from tqdm import tqdm
import numpy as np
import librosa
import matplotlib.pyplot as plt
from algorithms import chroma_features, compare_features

def visualize_similarity_analysis(original_name, cover_name, original_chroma, cover_chroma, save_plots=True):
    """
    Visualize similarity analysis between original and cover song using frame-synchronous chroma.
    
    Parameters:
    original_name, cover_name: song names for titles
    original_chroma, cover_chroma: frame-synchronous chroma feature matrices
    save_plots: whether to save plots to files
    """
    from algorithms import similarity_matrix, D_matrix, smith_waterman
    
    # Compute similarity matrices
    S = similarity_matrix(original_chroma, cover_chroma, norm= True)  # Raw dot product
    D = smith_waterman(S,0.5,0.5)  # Dynamic programming matrix
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Similarity Analysis: {original_name} vs {cover_name}', fontsize=16)
    
    # 1. Original chroma features
    im1 = axes[0,0].imshow(original_chroma, aspect='auto', cmap='Blues', origin='lower')
    axes[0,0].set_title(f'Original: {original_name}')
    axes[0,0].set_ylabel('Chroma Bins')
    axes[0,0].set_xlabel('Time Frames')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    
    # 2. Cover chroma features  
    im2 = axes[0,1].imshow(cover_chroma, aspect='auto', cmap='Reds', origin='lower')
    axes[0,1].set_title(f'Cover: {cover_name}')
    axes[0,1].set_ylabel('Chroma Bins')
    axes[0,1].set_xlabel('Time Frames')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    
    # 3. Raw similarity matrix (S)
    im3 = axes[0,2].imshow(S, aspect='auto', cmap='viridis', origin='lower')
    axes[0,2].set_title(f'Raw Similarity Matrix (S)\nMax: {np.max(S):.3f}')
    axes[0,2].set_ylabel('Original Frames')
    axes[0,2].set_xlabel('Cover Frames')
    plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
    
    # 4. Dynamic programming matrix (D)
    im4 = axes[1,0].imshow(D, aspect='auto', cmap='plasma', origin='lower')
    axes[1,0].set_title(f'DP Similarity Matrix (D)\nMax: {np.max(D):.3f}')
    axes[1,0].set_ylabel('Original Frames')
    axes[1,0].set_xlabel('Cover Frames')
    plt.colorbar(im4, ax=axes[1,0], shrink=0.8)
    
    # 5. Difference matrix (D - S) to show DP effect
    diff_matrix = D - np.pad(S, ((1,0), (1,0)), mode='constant', constant_values=0)[:D.shape[0], :D.shape[1]]
    im5 = axes[1,1].imshow(diff_matrix, aspect='auto', cmap='RdBu', origin='lower')
    axes[1,1].set_title('DP Enhancement (D - S)')
    axes[1,1].set_ylabel('Original Frames')
    axes[1,1].set_xlabel('Cover Frames')
    plt.colorbar(im5, ax=axes[1,1], shrink=0.8)
    
    # 6. Optimal path visualization (trace back the path)
    max_val = np.max(D)
    max_pos = np.unravel_index(np.argmax(D), D.shape)
    
    # Trace back the optimal path from the maximum point
    def trace_optimal_path(D, S, end_i, end_j):
        """Trace back the optimal path in the DP matrix"""
        path = [(end_i, end_j)]
        i, j = end_i, end_j
        
        while i > 0 and j > 0:
            # Find which direction gave us the current value
            current_val = D[i, j]
            
            # Check three possible predecessors
            diagonal = D[i-1, j-1] + S[i-1, j-1] if i > 0 and j > 0 else -np.inf
            from_above = D[i-1, j] + S[i-1, j-1] if i > 0 else -np.inf
            from_left = D[i, j-1] + S[i-1, j-1] if j > 0 else -np.inf
            
            # Choose the direction that led to current value
            if abs(current_val - diagonal) < 1e-10 and diagonal >= max(from_above, from_left):
                i, j = i-1, j-1  # Diagonal move (best alignment)
            elif abs(current_val - from_above) < 1e-10:
                i = i-1  # Vertical move (skip in original)
            else:
                j = j-1  # Horizontal move (skip in cover)
            
            path.append((i, j))
        
        return path[::-1]  # Reverse to get forward path
    
    # Get the optimal path
    optimal_path = trace_optimal_path(D, S, max_pos[0], max_pos[1])
    
    # Debug: Print path information
    print(f"Path length: {len(optimal_path)}")
    print(f"Path start: {optimal_path[0] if optimal_path else 'None'}")
    print(f"Path end: {optimal_path[-1] if optimal_path else 'None'}")
    print(f"Matrix shape: {D.shape}")
    print(f"Max position: {max_pos}")
    
    # Create path visualization
    D_with_path = D.copy()
    axes[1,2].imshow(D_with_path, aspect='auto', cmap='plasma', origin='lower', interpolation='nearest')
    
    # Plot the optimal path with very visible styling
    if len(optimal_path) > 1:
        optimal_path = optimal_path[::-1]

        # Separate coordinates (rows=y, cols=x)
        path_i = [p[0] for p in optimal_path]   # rows (original)
        path_j = [p[1] for p in optimal_path]   # cols (cover)

        print(f"Plotting DTW path with {len(path_i)} points.")
        print(f"First few points: {optimal_path[:5]}")

        # --- Draw multilayer path for visibility ---
        axes[1,2].plot(path_j, path_i, color='yellow', linewidth=6, alpha=1.0, zorder=10, label='Optimal Path')
        axes[1,2].plot(path_j, path_i, color='white',  linewidth=3, alpha=0.9, zorder=11)
        axes[1,2].plot(path_j, path_i, color='red',    linewidth=1, alpha=0.8, zorder=12)

        # --- Highlight start and end points ---
        axes[1,2].plot(path_j[0],  path_i[0],  'go', markersize=14,
                    markeredgecolor='white', markeredgewidth=2, alpha=1.0, zorder=13, label='Start')
        axes[1,2].plot(path_j[-1], path_i[-1], 'bo', markersize=14,
                    markeredgecolor='white', markeredgewidth=2, alpha=1.0, zorder=13, label='End')

        axes[1,2].legend(loc='best')
        axes[1,2].set_title("DTW Cumulative Cost with Optimal Path")
    else:
        print("Warning: Path is too short to plot!")
    
    axes[1,2].plot(max_pos[1], max_pos[0], 'r*', markersize=15, label=f'Max Score: {max_val:.3f}')
    axes[1,2].set_title('Optimal Alignment Path')
    axes[1,2].set_ylabel('Original Frames')
    axes[1,2].set_xlabel('Cover Frames')
    axes[1,2].legend()
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"similarity_analysis_{original_name}_{cover_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
    
    #plt.show()
    
    # Print analysis summary
    print(f"\n=== Frame-Synchronous Similarity Analysis Summary ===")
    print(f"Original shape: {original_chroma.shape}")
    print(f"Cover shape: {cover_chroma.shape}")
    print(f"Similarity matrix shape: {S.shape}")
    print(f"Raw similarity max: {np.max(S):.6f}")
    print(f"DP similarity max: {np.max(D):.6f}")
    print(f"Improvement ratio: {np.max(D)/np.max(S):.2f}x")
    print(f"Best alignment at: Original frame {max_pos[0]}, Cover frame {max_pos[1]}")
    print(f"Frame-synchronous analysis: Using fixed-time windows")
    
    return S, D, max_val

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

    def load_songs_from_list(song_list: list[str], shift: int = 0):
        songs = {}
        print('Loading song list...')
        for song in tqdm(song_list):
            name = song.split('/')[0]
            data = pydub.AudioSegment.from_file(os.path.join(datadir, song) + '.mp3')
            if shift != 0:
                if shift > 0: # added for testing (shift songs in time). Can be removed for final code
                    data = data[shift:] + pydub.AudioSegment.silent(duration=shift)
                else:
                    shift_abs = abs(shift)
                    data = pydub.AudioSegment.silent(duration=shift_abs) + data[:-shift_abs]
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
        # Compute chroma features
        song_chroma = chroma_features(song_samples, sr_song, hop_time=100, n_fft=2048, variation="norm")
        score = compare_features(features, song_chroma)
        results.append((score, name))
    print(results)
    best_match = max(results, key=lambda x: x[0])
    print(best_match)
    score_match, name_match = best_match
    return (score_match, name_match)

def match_all_songs_features(database: dict[str, pydub.AudioSegment],
                    covers: dict[str, pydub.AudioSegment],
                    debug_mode=False
                    ) -> tuple[list[str], list[str]]:
    truth_list = []
    matched_list = []
    features_list = []
    print('Computing features dataset...')
    for name, data in tqdm(database.items()):
        data = data.set_channels(1) if data.channels > 1 else data
        data_samples = np.array(data.get_array_of_samples()).astype(np.float32)
        data_samples /= np.iinfo(data.array_type).max
        sr = data.frame_rate
        features_list.append((chroma_features(data_samples, sr, hop_time=100, n_fft=2048, variation="norm"),name))
    i = 0
    print('Matching covers to original songs...')
    for name, song in tqdm(covers.items()):
        if i >= 75:
            break
        
        # Debug mode: visualize the matching process
        if debug_mode and i < 3:  # Debug first 3 matches
            print(f"\n=== DEBUGGING MATCH {i+1}: {name} ===")
            
            # For debug visualization, we need to re-extract database features using chroma_features
            # to match the visualization features
            debug_features_list = []
            print("Re-extracting database features for visualization...")
            for db_name, db_audio in database.items():
                db_audio = db_audio.set_channels(1) if db_audio.channels > 1 else db_audio
                db_samples = np.array(db_audio.get_array_of_samples()).astype(np.float32)
                db_samples /= np.iinfo(db_audio.array_type).max
                db_sr = db_audio.frame_rate
                db_chroma = chroma_features(db_samples, db_sr, hop_time=100, n_fft=2048, variation="norm")
                debug_features_list.append((db_chroma, db_name))
            
            # Find the best match with debugging
            best_score = -1
            best_name = None
            correct_score = None
            correct_features = None
            cover_chroma = None

            for db_features, db_name in debug_features_list:
                # Extract cover features - use original chroma features for visualization
                song = song.set_channels(1) if song.channels > 1 else song
                song_samples = np.array(song.get_array_of_samples()).astype(np.float32)
                song_samples /= np.iinfo(song.array_type).max
                sr_song = song.frame_rate
                
                # Use original chroma features for better visualization detail
                song_chroma = chroma_features(song_samples, sr_song, hop_time=100, n_fft=2048, variation="norm")
                
                # Compute similarity
                score = compare_features(db_features, song_chroma)
                print(f"Checking {db_name} vs {name}: score = {score:.3f}")
                
                # Store correct match info
                if db_name == name:
                    correct_score = score
                    correct_features = db_features
                    cover_chroma = song_chroma
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_name = db_name
            
            # Visualize only correct match and final chosen match
            print(f"\n*** CORRECT MATCH: {name} (score: {correct_score:.3f}) ***")
            if correct_features is not None and cover_chroma is not None:
                visualize_similarity_analysis(name, name, correct_features, cover_chroma, save_plots=True)
            
            if best_name != name:
                print(f"*** ALGORITHM CHOSE: {best_name} (score: {best_score:.3f}) ***")
                # Find the chosen match features for visualization
                for db_features, db_name in debug_features_list:
                    if db_name == best_name:
                        visualize_similarity_analysis(best_name, name, db_features, cover_chroma, save_plots=True)
                        break
            else:
                print(f"*** ALGORITHM CORRECTLY CHOSE: {best_name} ***")
            
            matched_name = best_name
            print(f"Final decision: {name} -> {matched_name} (score: {best_score:.3f})")
            print("="*60)
        else:
            # Normal matching without debugging
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
