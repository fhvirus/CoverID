from utils import load_covers80, match_all_songs, match_all_songs_features
from beat_chroma import match_all_songs_features_beat_sync
from algorithms import dummy_compare, compare, compare_beat_sync

DATADIR = './coversongs/covers32k/'

if __name__ == '__main__':
    name_list, ver_A, ver_B = load_covers80(DATADIR)
    
    # Choose which approach to use
    use_beat_sync = True  # Set to False for frame-synchronous comparison
    
    if use_beat_sync:
        print("=== USING BEAT-SYNCHRONOUS CHROMA FEATURES ===")
        # Test different beat resolutions
        beat_resolutions = [4, 8]  # quarter beats and eighth beats
        
        for beats_per_frame in beat_resolutions:
            print(f"\n--- Testing with {beats_per_frame} beats per frame ({1/beats_per_frame:.3f} beat resolution) ---")
            
            # Run evaluation with beat-synchronous features
            truth_list, matched_list = match_all_songs_features_beat_sync(
                ver_A, ver_B, 
                beats_per_frame=beats_per_frame, 
                debug_mode=True
            )
            
            precision = sum([a == b for a, b in zip(truth_list, matched_list)]) / len(truth_list)
            print(f'Beat-sync Precision (beats_per_frame={beats_per_frame}): {precision:.3f}')
            
            # Print detailed results
            print(f"Correct matches: {sum([a == b for a, b in zip(truth_list, matched_list)])}/{len(truth_list)}")
            for truth, matched in zip(truth_list, matched_list):
                status = "✓" if truth == matched else "✗"
                print(f"  {status} {truth} -> {matched}")
    else:
        print("=== USING FRAME-SYNCHRONOUS CHROMA FEATURES ===")
        # Run evaluation with frame-synchronous features (original approach)
        truth_list, matched_list = match_all_songs_features(ver_A, ver_B, debug_mode=True)

        precision = sum([a == b for a, b in zip(truth_list, matched_list)]) / len(truth_list)
        print(f'Frame-sync Precision: {precision:.3f}')

    # confusion_matrix = sklearn.metrics.classification_report(truth_list, matched_list)
    # print(confusion_matrix)
