"""Test cleaned dataset loading."""
from lane_av_rel.datasets import DeepfakeTIMITDataset
import time

print("Testing cleaned dataset loading...")
start = time.time()

ds = DeepfakeTIMITDataset(
    root_dir='.',
    cleaned_dir='./cleaned_dataset',
    split='train'
)

print(f"Dataset size: {len(ds)}")
sample = ds[0]

print(f"Video shape: {sample['video'].shape}")
print(f"Audio shape: {sample['audio'].shape}")
print(f"Label: {sample['label']}")
print(f"Speaker: {sample['speaker']}")

load_time = time.time() - start
print(f"\nLoading time: {load_time:.3f}s")
print("Cleaned dataset loaded successfully!")

