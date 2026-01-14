# Dataset Preprocessing Guide

## Why Preprocess the Dataset?

Preprocessing the dataset **once** before training provides several benefits:

### ✅ **Performance Benefits**
- **10-100x faster loading**: Preprocessed frames/audio are loaded instantly vs. decoding videos/audio on-the-fly
- **Consistent data**: All samples have the same frame count and audio length
- **Reduced I/O**: No need to read video files during training

### ✅ **Reliability Benefits**
- **Validation**: Invalid/corrupted files are detected and removed upfront
- **Reproducibility**: Fixed train/val/test splits saved in metadata
- **Error handling**: Problems are caught before training starts

### ✅ **Ease of Use**
- **Simple loading**: Just point to cleaned dataset directory
- **No processing overhead**: Training loop focuses on model, not data loading
- **Easy sharing**: Preprocessed dataset can be shared/backed up easily

## How It Works

The preprocessing script (`preprocess_dataset.py`) does the following:

1. **Discovers** all video/audio pairs in the dataset
2. **Validates** each pair (checks if files are readable)
3. **Extracts** frames from videos (uniformly samples to fixed length)
4. **Preprocesses** audio (resamples to 16kHz, converts to mono)
5. **Splits** data into train/val/test (70%/15%/15%)
6. **Saves** everything in a clean format:
   - Frames: `cleaned_dataset/frames/{sample_id}.npy`
   - Audio: `cleaned_dataset/audio/{sample_id}.npy`
   - Metadata: `cleaned_dataset/metadata.json`
   - Summary: `cleaned_dataset/summary.json`

## Usage

### Step 1: Preprocess the Dataset

```bash
python preprocess_dataset.py --data_dir . --output_dir ./cleaned_dataset
```

**Options:**
- `--data_dir`: Root directory containing `fake/` and `real/` folders
- `--output_dir`: Where to save cleaned dataset (default: `./cleaned_dataset`)
- `--quality`: `higher_quality` or `lower_quality` (default: `higher_quality`)
- `--video_length`: Number of frames per video (default: 32)
- `--audio_sample_rate`: Target audio sample rate (default: 16000)
- `--skip_existing`: Skip already processed samples (default: True)

**Example:**
```bash
python preprocess_dataset.py \
    --data_dir . \
    --output_dir ./cleaned_dataset \
    --quality higher_quality \
    --video_length 32 \
    --audio_sample_rate 16000
```

### Step 2: Use Cleaned Dataset in Training

The dataset loader automatically detects and uses cleaned data if available:

```python
from lane_av_rel.datasets import DeepfakeTIMITDataset

# Option 1: Use cleaned dataset (recommended)
train_dataset = DeepfakeTIMITDataset(
    root_dir='.',  # Still needed for fallback
    quality='higher_quality',
    split='train',
    cleaned_dir='./cleaned_dataset'  # Point to cleaned dataset
)

# Option 2: Use raw data (slower, but works)
train_dataset = DeepfakeTIMITDataset(
    root_dir='.',
    quality='higher_quality',
    split='train'
    # No cleaned_dir = uses raw data
)
```

## Output Structure

After preprocessing, you'll have:

```
cleaned_dataset/
├── frames/
│   ├── train_speaker1_00000.npy
│   ├── train_speaker1_00001.npy
│   ├── val_speaker2_00000.npy
│   └── ...
├── audio/
│   ├── train_speaker1_00000.npy
│   ├── train_speaker1_00001.npy
│   ├── val_speaker2_00000.npy
│   └── ...
├── metadata.json      # Full metadata for all samples
└── summary.json       # Dataset statistics
```

### File Formats

- **Frames** (`.npy`): NumPy array of shape `(T, H, W, C)` where:
  - `T` = number of frames (e.g., 32)
  - `H, W` = frame dimensions (e.g., 224, 224)
  - `C` = channels (3 for RGB)
  - Dtype: `uint8` (0-255)

- **Audio** (`.npy`): NumPy array of shape `(T_a,)` where:
  - `T_a` = number of audio samples
  - Dtype: `float32` (-1.0 to 1.0)

- **Metadata** (`.json`): Contains sample information:
  ```json
  {
    "train": [
      {
        "sample_id": "train_speaker1_00000",
        "speaker": "speaker1",
        "quality": "higher_quality",
        "label": 0,
        "original_video": "path/to/video.avi",
        "original_audio": "path/to/audio.wav",
        "frame_shape": [32, 224, 224, 3],
        "audio_shape": [32000]
      },
      ...
    ],
    "val": [...],
    "test": [...]
  }
  ```

## Is This the Right Step?

**Yes!** Preprocessing is a **best practice** for deep learning projects because:

1. ✅ **Faster training**: Data loading is often the bottleneck, not model computation
2. ✅ **Better debugging**: Problems are caught early, not during training
3. ✅ **Reproducibility**: Same splits every time, no randomness in data loading
4. ✅ **Scalability**: Easy to add more data or change preprocessing without retraining

**When to skip preprocessing:**
- Very small datasets (< 100 samples)
- Need to experiment with different frame counts/audio lengths frequently
- Dataset changes frequently and reprocessing is expensive

**For this project:** Preprocessing is **highly recommended** because:
- DeepfakeTIMIT has hundreds of samples
- Fixed frame count (32) and audio length are standard
- Training will be much faster with preprocessed data

## Troubleshooting

### "No samples found"
- Check that `fake/DeepfakeTIMIT/{quality}/` directory exists
- Verify video files (`.avi`) and audio files (`.wav`) are present
- Check file naming matches expected pattern

### "Invalid video/audio"
- Some files may be corrupted
- The script will skip invalid files and continue
- Check the output for which files failed

### "Out of memory"
- Process in smaller batches (modify script to process split-by-split)
- Reduce `video_length` or frame resolution
- Use `--skip_existing` to resume interrupted preprocessing

## Next Steps

After preprocessing:

1. ✅ Verify cleaned dataset loads correctly:
   ```python
   from lane_av_rel.datasets import DeepfakeTIMITDataset
   ds = DeepfakeTIMITDataset(root_dir='.', cleaned_dir='./cleaned_dataset', split='train')
   print(f"Dataset size: {len(ds)}")
   sample = ds[0]
   print(f"Video shape: {sample['video'].shape}")
   print(f"Audio shape: {sample['audio'].shape}")
   ```

2. ✅ Start training with cleaned dataset:
   ```bash
   python train.py --data_dir . --cleaned_dir ./cleaned_dataset
   ```

3. ✅ Monitor training speed - should be much faster now!

