#!/usr/bin/env python3
"""
Focused Ground Truth Template for Key Detection Testing
Includes songs from A_Whiter_Shade_Of_Pale through Blue_Collar_Man (9 songs total)
"""

# Ground truth dictionary for the first 9 songs
GROUND_TRUTH = {
    "A_Whiter_Shade_Of_Pale": {
        "A": 17,  # Original version - TODO: Fill in correct key (0-23)
        "B": 0   # Cover version - TODO: Fill in correct key (0-23)
    },
    "Abracadabra": {
        "A": 21,  # Original version - TODO: Fill in correct key (0-23)
        "B": 19   # Cover version - TODO: Fill in correct key (0-23)
    },
    "Addicted_To_Love": {
        "A": 9,  # Original version - TODO: Fill in correct key (0-23)
        "B": 23   # Cover version - TODO: Fill in correct key (0-23)
    },
    "All_Along_The_Watchtower": {
        "A": 21,  # Original version - TODO: Fill in correct key (0-23)
        "B": 21   # Cover version - TODO: Fill in correct key (0-23)
    },
    "All_Tomorrow_s_Parties": {
        "A": 2,  # Original version - TODO: Fill in correct key (0-23)
        "B": 2   # Cover version - TODO: Fill in correct key (0-23)
    },
    "America": {
        "A": 2,  # Original version - TODO: Fill in correct key (0-23)
        "B": 2   # Cover version - TODO: Fill in correct key (0-23)
    },
    "Before_You_Accuse_Me": {
        "A": 7,  # Original version - TODO: Fill in correct key (0-23)
        "B": 16  # Cover version - TODO: Fill in correct key (0-23)
    },
    "Between_The_Bars": {
        "A": 7,  # Original version - TODO: Fill in correct key (0-23)
        "B": 7   # Cover version - TODO: Fill in correct key (0-23)
    },
    "Blue_Collar_Man": {
        "A": 21,  # Original version - TODO: Fill in correct key (0-23)
        "B": 14   # Cover version - TODO: Fill in correct key (0-23)
    }
}

# Key notation reference
KEY_REFERENCE = {
    "Major keys": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    "Minor keys": ["Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"]
}

def get_ground_truth():
    """Return the ground truth dictionary."""
    return GROUND_TRUTH

def get_key_reference():
    """Return the key reference guide."""
    return KEY_REFERENCE

if __name__ == "__main__":
    print("Focused Ground Truth Template")
    print("=" * 50)
    print(f"Songs included: {len(GROUND_TRUTH)}")
    print(f"Total annotations needed: {len(GROUND_TRUTH) * 2}")
    
    print("\n📋 Songs to annotate:")
    for i, song_name in enumerate(GROUND_TRUTH.keys(), 1):
        print(f"  {i}. {song_name}")
    
    print("\n🎵 Key Reference:")
    print("Major keys:", ", ".join(KEY_REFERENCE["Major keys"]))
    print("Minor keys:", ", ".join(KEY_REFERENCE["Minor keys"]))
    
    print("\n🎯 Instructions:")
    print("1. Listen to each song version (A = original, B = cover)")
    print("2. Determine the actual key")
    print("3. Replace 'FILL_IN_KEY' with correct key notation")
    print("4. Save this file as 'ground_truth_test.py'")
    print("5. Use with test_key_detection_accuracy.py")
