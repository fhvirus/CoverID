#!/usr/bin/env python3
"""
Key Detection Accuracy Test - Small Dataset Version
Tests both key detection algorithms on the first 9 songs (up to Blue_Collar_Man)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import sys

# Import our modules
from utils import load_covers80
from algorithms import detect_key_algorithm, detect_key, chroma_features

# Import ground truth (you'll need to fill this in manually)
try:
    from ground_truth_test_small import get_ground_truth
    GROUND_TRUTH = get_ground_truth()
except ImportError:
    print("Error: ground_truth_test_small.py not found or incomplete!")
    print("Please create and fill in the ground truth annotations first.")
    sys.exit(1)

def key_number_to_name(key_num, algorithm_type="paper"):
    """Convert key number to readable key name."""
    if algorithm_type == "paper":
        # Paper algorithm uses 24 keys (0-23)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
    else:
        # Original algorithm uses 12 keys (0-11, major only)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return keys[key_num] if 0 <= key_num < len(keys) else f"Unknown({key_num})"

def key_name_to_number(key_name):
    """Convert key name to number for paper algorithm (24-key system)."""
    key_map = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        'Cm': 12, 'C#m': 13, 'Dm': 14, 'D#m': 15, 'Em': 16, 'Fm': 17,
        'F#m': 18, 'Gm': 19, 'G#m': 20, 'Am': 21, 'A#m': 22, 'Bm': 23
    }
    return key_map.get(key_name, -1)

def detect_keys_for_all_songs(ver_A, ver_B, ground_truth):
    """Detect keys for all songs using both algorithms."""
    results = {}
    
    print("🎵 Starting key detection for all songs...")
    print(f"📊 Testing {len(ground_truth)} songs")
    
    for song_name in ground_truth.keys():
        print(f"\n🎶 Processing: {song_name}")
        
        if song_name not in ver_A or song_name not in ver_B:
            print(f"⚠️  Warning: {song_name} not found in database!")
            continue
        
        results[song_name] = {}
        
        # Process both versions
        for version, data_dict in [("A", ver_A), ("B", ver_B)]:
            try:
                # Load audio data
                song_audio_segment = data_dict[song_name]
                
                # Convert pydub AudioSegment to numpy array
                song_data = np.array(song_audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # If stereo, convert to mono by taking the mean
                if song_audio_segment.channels == 2:
                    song_data = song_data.reshape((-1, 2)).mean(axis=1)
                
                # Normalize the audio data
                if len(song_data) > 0:
                    song_data = song_data / np.max(np.abs(song_data))
                
                sr = song_audio_segment.frame_rate  # Use actual sample rate
                
                # Extract chroma features
                chroma = chroma_features(song_data, sr, hop_time=100, n_fft=2048, variation="norm")
                
                # Detect keys using both algorithms
                paper_key, paper_correlation = detect_key_algorithm(chroma)  # Paper algorithm (24 keys)
                original_key = detect_key(chroma, krum_schm=True, Algo=False)  # Original K-S algorithm (12 keys)
                
                results[song_name][version] = {
                    "paper_algorithm": paper_key,
                    "original_algorithm": original_key,
                    "ground_truth": ground_truth[song_name][version]
                }
                
                print(f"  📄 {version}: Paper={key_number_to_name(paper_key, 'paper')}, "
                      f"Original={key_number_to_name(original_key, 'original')}, "
                      f"Truth={ground_truth[song_name][version]}")
                
            except Exception as e:
                print(f"❌ Error processing {song_name} version {version}: {e}")
                results[song_name][version] = {
                    "paper_algorithm": -1,
                    "original_algorithm": -1,
                    "ground_truth": ground_truth[song_name][version],
                    "error": str(e)
                }
    
    return results

def calculate_accuracy(results):
    """Calculate accuracy for both algorithms."""
    stats = {
        "paper_algorithm": {"correct": 0, "total": 0, "accuracy": 0.0},
        "original_algorithm": {"correct": 0, "total": 0, "accuracy": 0.0}
    }
    
    detailed_results = []
    
    for song_name, song_data in results.items():
        for version in ["A", "B"]:
            if version not in song_data:
                continue
                
            version_data = song_data[version]
            ground_truth = version_data["ground_truth"]
            
            # Skip if ground truth not filled in
            if isinstance(ground_truth, str) and ground_truth == "FILL_IN_KEY":
                print(f"⚠️  Skipping {song_name} {version}: Ground truth not filled in")
                continue
            
            # Convert ground truth to number if it's a string
            if isinstance(ground_truth, str):
                ground_truth_num = key_name_to_number(ground_truth)
                if ground_truth_num == -1:
                    print(f"⚠️  Warning: Invalid ground truth key '{ground_truth}' for {song_name} {version}")
                    continue
            else:
                ground_truth_num = ground_truth  # Already a number
            
            # Check paper algorithm
            paper_key = version_data["paper_algorithm"]
            if paper_key != -1:
                stats["paper_algorithm"]["total"] += 1
                if paper_key == ground_truth_num:
                    stats["paper_algorithm"]["correct"] += 1
            
            # Check original algorithm (convert to major key equivalent for comparison)
            original_key = version_data["original_algorithm"]
            if original_key != -1:
                stats["original_algorithm"]["total"] += 1
                # For original algorithm, we only compare major keys
                ground_truth_major = ground_truth_num % 12 if ground_truth_num < 12 else -1
                if ground_truth_major != -1 and original_key == ground_truth_major:
                    stats["original_algorithm"]["correct"] += 1
            
            # Store detailed result
            ground_truth_display = key_number_to_name(ground_truth_num, "paper") if isinstance(ground_truth, int) else ground_truth
            detailed_results.append({
                "song": song_name,
                "version": version,
                "ground_truth": ground_truth_display,
                "paper_detected": key_number_to_name(paper_key, "paper") if paper_key != -1 else "Error",
                "original_detected": key_number_to_name(original_key, "original") if original_key != -1 else "Error",
                "paper_correct": paper_key == ground_truth_num,
                "original_correct": (original_key == (ground_truth_num % 12)) if ground_truth_num < 12 else False
            })
    
    # Calculate accuracy percentages
    for algorithm in stats:
        if stats[algorithm]["total"] > 0:
            stats[algorithm]["accuracy"] = (stats[algorithm]["correct"] / stats[algorithm]["total"]) * 100
    
    return stats, detailed_results

def generate_accuracy_report(stats, detailed_results):
    """Generate comprehensive accuracy report."""
    print("\n" + "="*60)
    print("🎯 KEY DETECTION ACCURACY REPORT")
    print("="*60)
    
    print(f"\n📊 Overall Results:")
    print(f"   Paper Algorithm:    {stats['paper_algorithm']['correct']}/{stats['paper_algorithm']['total']} "
          f"({stats['paper_algorithm']['accuracy']:.1f}%)")
    print(f"   Original Algorithm: {stats['original_algorithm']['correct']}/{stats['original_algorithm']['total']} "
          f"({stats['original_algorithm']['accuracy']:.1f}%)")
    
    # Detailed breakdown
    print(f"\n📋 Detailed Results:")
    print(f"{'Song':<20} {'Ver':<3} {'Truth':<6} {'Paper':<6} {'Original':<8} {'P✓':<3} {'O✓':<3}")
    print("-" * 60)
    
    for result in detailed_results:
        p_mark = "✓" if result["paper_correct"] else "✗"
        o_mark = "✓" if result["original_correct"] else "✗"
        
        print(f"{result['song']:<20} {result['version']:<3} {result['ground_truth']:<6} "
              f"{result['paper_detected']:<6} {result['original_detected']:<8} "
              f"{p_mark:<3} {o_mark:<3}")
    
    # Summary
    print(f"\n📈 Summary:")
    if stats['paper_algorithm']['accuracy'] > stats['original_algorithm']['accuracy']:
        print(f"🏆 Paper algorithm performs better by {stats['paper_algorithm']['accuracy'] - stats['original_algorithm']['accuracy']:.1f} percentage points")
    elif stats['original_algorithm']['accuracy'] > stats['paper_algorithm']['accuracy']:
        print(f"🏆 Original algorithm performs better by {stats['original_algorithm']['accuracy'] - stats['paper_algorithm']['accuracy']:.1f} percentage points")
    else:
        print(f"🤝 Both algorithms perform equally well")

def save_results_to_files(results, stats, detailed_results):
    """Save results to CSV and JSON files."""
    
    # Save detailed results to CSV
    df = pd.DataFrame(detailed_results)
    csv_file = "key_detection_results_small.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved to: {csv_file}")
    
    # Save complete data to JSON
    json_data = {
        "metadata": {
            "test_name": "Key Detection Accuracy Test - Small Dataset",
            "songs_tested": len(GROUND_TRUTH),
            "total_samples": len(detailed_results)
        },
        "statistics": stats,
        "detailed_results": detailed_results,
        "raw_detection_data": results
    }
    
    json_file = "key_detection_results_small.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"💾 Complete data saved to: {json_file}")

def main():
    """Main function to run the accuracy test."""
    
    print("🎼 Key Detection Accuracy Test - Small Dataset")
    print("=" * 50)
    
    # Check if ground truth is filled in
    unfilled_count = 0
    for song_name, versions in GROUND_TRUTH.items():
        for version, key in versions.items():
            if isinstance(key, str) and key == "FILL_IN_KEY":
                unfilled_count += 1
    
    if unfilled_count > 0:
        print(f"⚠️  Warning: {unfilled_count} ground truth entries still need to be filled in!")
        print("   Please edit ground_truth_test_small.py and replace 'FILL_IN_KEY' with actual keys")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load the covers80 database
    print("\n📂 Loading covers80 database...")
    try:
        name_list, ver_A, ver_B = load_covers80("coversongs/covers32k")
        print(f"✅ Database loaded successfully")
        print(f"   Song names: {len(name_list)} songs")
        print(f"   Version A (originals): {len(ver_A)} songs")
        print(f"   Version B (covers): {len(ver_B)} songs")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Run key detection
    results = detect_keys_for_all_songs(ver_A, ver_B, GROUND_TRUTH)
    
    # Calculate accuracy
    stats, detailed_results = calculate_accuracy(results)
    
    # Generate report
    generate_accuracy_report(stats, detailed_results)
    
    # Save results
    save_results_to_files(results, stats, detailed_results)
    
    print(f"\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
