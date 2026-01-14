"""
Script to check and analyze labels in the cleaned dataset.
Shows how to distinguish between real and fake samples.
"""

import json
from pathlib import Path
from collections import Counter

def analyze_dataset_labels(metadata_path: str = "cleaned_dataset/metadata.json"):
    """Analyze the dataset to show label distribution and how to identify real vs fake."""
    
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return
    
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)
    
    print("=" * 70)
    print("DATASET LABEL ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        if split_name not in all_metadata:
            continue
        
        samples = all_metadata[split_name]
        labels = [sample['label'] for sample in samples]
        label_counts = Counter(labels)
        
        print(f"{split_name.upper()} Split:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Label distribution:")
        for label, count in sorted(label_counts.items()):
            label_name = "FAKE" if label == 0 else "REAL" if label == 1 else f"UNKNOWN ({label})"
            print(f"    Label {label} ({label_name}): {count} samples")
        print()
    
    # Show examples
    print("=" * 70)
    print("HOW TO IDENTIFY REAL VS FAKE:")
    print("=" * 70)
    print()
    print("1. Check the 'label' field in metadata.json:")
    print("   - label: 0 = FAKE")
    print("   - label: 1 = REAL")
    print()
    print("2. Check the 'original_video' path:")
    print("   - Contains 'fake\\' = FAKE")
    print("   - Contains 'real\\' = REAL")
    print()
    
    # Show sample examples
    print("=" * 70)
    print("SAMPLE ENTRIES:")
    print("=" * 70)
    print()
    
    # Show a fake example
    for split_name in ['train', 'val', 'test']:
        if split_name not in all_metadata:
            continue
        samples = all_metadata[split_name]
        if samples:
            sample = samples[0]
            print(f"Example from {split_name} split:")
            print(f"  sample_id: {sample['sample_id']}")
            print(f"  label: {sample['label']} ({'FAKE' if sample['label'] == 0 else 'REAL'})")
            print(f"  speaker: {sample['speaker']}")
            print(f"  original_video: {sample['original_video']}")
            print()
            break
    
    # Check if there are any real samples
    all_samples = []
    for split_name in ['train', 'val', 'test']:
        if split_name in all_metadata:
            all_samples.extend(all_metadata[split_name])
    
    real_samples = [s for s in all_samples if s['label'] == 1]
    fake_samples = [s for s in all_samples if s['label'] == 0]
    
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Total samples: {len(all_samples)}")
    print(f"FAKE samples (label=0): {len(fake_samples)}")
    print(f"REAL samples (label=1): {len(real_samples)}")
    print()
    
    if len(real_samples) == 0:
        print("WARNING: No REAL samples found in the dataset!")
        print("   The cleaned dataset currently only contains FAKE samples.")
        print("   To add real samples, you would need to:")
        print("   1. Process videos from the 'real/' directory")
        print("   2. Assign them label: 1")
        print("   3. Add them to the metadata.json")
    else:
        print("Dataset contains both REAL and FAKE samples.")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    analyze_dataset_labels()

