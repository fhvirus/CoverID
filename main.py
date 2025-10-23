from utils import load_covers80, match_all_songs_features_fast, accuracy_score, visualize_normalized_matched_scores
from beat_chroma import match_all_songs_features_beat_sync

DATADIR = './coversongs/covers32k/'

if __name__ == '__main__':
    name_list, ver_A, ver_B = load_covers80(DATADIR)
    
    # Choose which approach to use
    use_beat_sync = False  # Set to False for frame-synchronous comparison
    
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
        # Run evaluation with frame-synchronous features and get top 5 matches
        truth_list, matched_dict, score_dict = match_all_songs_features_fast(ver_A, ver_B)
        # truth_list, matched_dict = match_all_songs_features_beat(ver_A, ver_B)

        # Calculate different accuracy metrics
        top1_accuracy = accuracy_score(truth_list, matched_dict, k=1)
        top3_accuracy = accuracy_score(truth_list, matched_dict, k=3)
        top5_accuracy = accuracy_score(truth_list, matched_dict, k=5)

        print(f'Top-1 Accuracy: {top1_accuracy:.3f}')
        print(f'Top-3 Accuracy: {top3_accuracy:.3f}')
        print(f'Top-5 Accuracy: {top5_accuracy:.3f}')

        visualize_normalized_matched_scores(truth_list, score_dict)
        
    # confusion_matrix = sklearn.metrics.classification_report(truth_list, matched_list)
    # print(confusion_matrix)
