"""
Dataset Preprocessing Script
============================

This script cleans and preprocesses the DeepfakeTIMIT dataset:
1. Validates all video/audio pairs
2. Extracts and saves frames in a clean format
3. Preprocesses audio (resample, mono conversion)
4. Creates proper train/val/test splits
5. Saves metadata for fast loading

Usage:
    python preprocess_dataset.py --data_dir . --output_dir ./cleaned_dataset
"""

import argparse
import cv2
import numpy as np
import soundfile as sf
import librosa
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pickle


def validate_video(video_path: Path) -> Tuple[bool, Optional[Dict]]:
    """Validate video file and extract metadata."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame to verify
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame_count == 0:
            return False, None
        
        return True, {
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height
        }
    except Exception as e:
        print(f"Error validating video {video_path}: {e}")
        return False, None


def validate_audio(audio_path: Path, target_sr: int = 16000) -> Tuple[bool, Optional[Dict]]:
    """Validate audio file and extract metadata."""
    try:
        audio, sr = sf.read(str(audio_path))
        
        # Check if audio is valid
        if len(audio) == 0:
            return False, None
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        duration = len(audio) / target_sr
        
        return True, {
            'sample_rate': target_sr,
            'duration': duration,
            'samples': len(audio)
        }
    except Exception as e:
        print(f"Error validating audio {audio_path}: {e}")
        return False, None


def extract_frames(video_path: Path, video_length: int = 32) -> Optional[np.ndarray]:
    """Extract and sample frames from video."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Sample frames uniformly
        if len(frames) > video_length:
            indices = np.linspace(0, len(frames) - 1, video_length, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < video_length:
            # Pad with last frame
            frames.extend([frames[-1]] * (video_length - len(frames)))
        
        return np.array(frames, dtype=np.uint8)
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return None


def preprocess_audio(audio_path: Path, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Preprocess audio: resample, convert to mono."""
    try:
        audio, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing audio {audio_path}: {e}")
        return None


def discover_samples(root_dir: Path, quality: str = 'higher_quality') -> List[Dict]:
    """Discover all valid video/audio pairs."""
    samples = []
    fake_dir = root_dir / 'fake' / 'DeepfakeTIMIT' / quality
    
    if not fake_dir.exists():
        print(f"Warning: {fake_dir} does not exist")
        return samples
    
    print(f"Discovering samples in {fake_dir}...")
    
    for speaker_dir in tqdm(list(fake_dir.iterdir()), desc="Scanning speakers"):
        if not speaker_dir.is_dir():
            continue
        
        for video_file in speaker_dir.glob('*.avi'):
            # Extract base name before '-video-' (e.g., 'sa1-video-fram1.avi' -> 'sa1.wav')
            video_stem = video_file.stem
            if '-video-' in video_stem:
                base_name = video_stem.split('-video-')[0]
                audio_file = speaker_dir / f"{base_name}.wav"
            else:
                # Fallback: try same name with .wav extension
                audio_file = speaker_dir / f"{video_stem}.wav"
            
            if audio_file.exists():
                samples.append({
                    'video_path': video_file,
                    'audio_path': audio_file,
                    'speaker': speaker_dir.name,
                    'quality': quality
                })
    
    return samples


def create_splits(samples: List[Dict], seed: int = 42) -> Dict[str, List[Dict]]:
    """Create train/val/test splits."""
    np.random.seed(seed)
    np.random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    splits = {
        'train': samples[:n_train],
        'val': samples[n_train:n_train + n_val],
        'test': samples[n_train + n_val:]
    }
    
    return splits


def preprocess_dataset(
    root_dir: Path,
    output_dir: Path,
    quality: str = 'higher_quality',
    video_length: int = 32,
    audio_sample_rate: int = 16000,
    skip_existing: bool = True
):
    """Main preprocessing function."""
    print("=" * 60)
    print("Dataset Preprocessing Script")
    print("=" * 60)
    print(f"Input directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Quality: {quality}")
    print(f"Video length: {video_length} frames")
    print(f"Audio sample rate: {audio_sample_rate} Hz")
    print("=" * 60)
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'frames').mkdir(exist_ok=True)
    (output_dir / 'audio').mkdir(exist_ok=True)
    
    # Discover samples
    samples = discover_samples(root_dir, quality)
    print(f"\nFound {len(samples)} video/audio pairs")
    
    if len(samples) == 0:
        print("No samples found! Exiting.")
        return
    
    # Create splits
    splits = create_splits(samples)
    print(f"\nSplits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Process each split
    all_metadata = {}
    
    for split_name, split_samples in splits.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split ({len(split_samples)} samples)")
        print(f"{'=' * 60}")
        
        split_metadata = []
        valid_count = 0
        invalid_count = 0
        
        for idx, sample in enumerate(tqdm(split_samples, desc=f"Processing {split_name}")):
            # Create unique ID for this sample
            sample_id = f"{split_name}_{sample['speaker']}_{idx:05d}"
            
            # Check if already processed
            frame_file = output_dir / 'frames' / f"{sample_id}.npy"
            audio_file = output_dir / 'audio' / f"{sample_id}.npy"
            
            if skip_existing and frame_file.exists() and audio_file.exists():
                # Load existing metadata
                try:
                    with open(frame_file, 'rb') as f:
                        frames = np.load(f)
                    with open(audio_file, 'rb') as f:
                        audio = np.load(f)
                    
                    split_metadata.append({
                        'sample_id': sample_id,
                        'speaker': sample['speaker'],
                        'quality': sample['quality'],
                        'label': 0,  # Fake
                        'original_video': str(sample['video_path']),
                        'original_audio': str(sample['audio_path']),
                        'frame_shape': frames.shape,
                        'audio_shape': audio.shape
                    })
                    valid_count += 1
                    continue
                except Exception as e:
                    print(f"Error loading existing {sample_id}: {e}")
            
            # Validate video
            video_valid, video_info = validate_video(sample['video_path'])
            if not video_valid:
                print(f"Invalid video: {sample['video_path']}")
                invalid_count += 1
                continue
            
            # Validate audio
            audio_valid, audio_info = validate_audio(sample['audio_path'], audio_sample_rate)
            if not audio_valid:
                print(f"Invalid audio: {sample['audio_path']}")
                invalid_count += 1
                continue
            
            # Extract frames
            frames = extract_frames(sample['video_path'], video_length)
            if frames is None:
                print(f"Failed to extract frames: {sample['video_path']}")
                invalid_count += 1
                continue
            
            # Preprocess audio
            audio = preprocess_audio(sample['audio_path'], audio_sample_rate)
            if audio is None:
                print(f"Failed to preprocess audio: {sample['audio_path']}")
                invalid_count += 1
                continue
            
            # Save frames and audio
            try:
                np.save(frame_file, frames)
                np.save(audio_file, audio)
                
                split_metadata.append({
                    'sample_id': sample_id,
                    'speaker': sample['speaker'],
                    'quality': sample['quality'],
                    'label': 0,  # Fake
                    'original_video': str(sample['video_path']),
                    'original_audio': str(sample['audio_path']),
                    'frame_shape': frames.shape,
                    'audio_shape': audio.shape,
                    'video_info': video_info,
                    'audio_info': audio_info
                })
                valid_count += 1
            except Exception as e:
                print(f"Error saving {sample_id}: {e}")
                invalid_count += 1
        
        all_metadata[split_name] = split_metadata
        print(f"\n{split_name}: {valid_count} valid, {invalid_count} invalid")
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Save summary
    summary = {
        'total_samples': len(samples),
        'train_samples': len(all_metadata['train']),
        'val_samples': len(all_metadata['val']),
        'test_samples': len(all_metadata['test']),
        'video_length': video_length,
        'audio_sample_rate': audio_sample_rate,
        'quality': quality
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("Preprocessing Complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_file}")
    print(f"Summary: {summary_file}")
    print(f"\nSummary:")
    print(f"  Train: {summary['train_samples']} samples")
    print(f"  Val: {summary['val_samples']} samples")
    print(f"  Test: {summary['test_samples']} samples")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess DeepfakeTIMIT dataset')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Root directory containing fake/ and real/ folders')
    parser.add_argument('--output_dir', type=str, default='./cleaned_dataset',
                       help='Output directory for cleaned dataset')
    parser.add_argument('--quality', type=str, default='higher_quality',
                       choices=['higher_quality', 'lower_quality'],
                       help='Quality level to process')
    parser.add_argument('--video_length', type=int, default=32,
                       help='Number of frames per video')
    parser.add_argument('--audio_sample_rate', type=int, default=16000,
                       help='Target audio sample rate')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip already processed samples')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        root_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        quality=args.quality,
        video_length=args.video_length,
        audio_sample_rate=args.audio_sample_rate,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()

