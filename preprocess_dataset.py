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
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pickle


def validate_video(video_path, frames_per_video: int = 30) -> Tuple[bool, Optional[Dict]]:
    """
    Validate video file, image file, image sequence directory, or list of frame paths.
    
    Args:
        video_path: Can be Path to file/directory, or list of Path objects
        frames_per_video: For frame sequences, expected number of frames
    """
    try:
        # Check if it's a list of frame paths (for real videos stored as individual frame files)
        if isinstance(video_path, list):
            if len(video_path) == 0:
                return False, None
            
            # Try reading first frame to get dimensions
            first_frame_path = video_path[0]
            if isinstance(first_frame_path, str):
                first_frame_path = Path(first_frame_path)
            
            img = cv2.imread(str(first_frame_path))
            if img is None:
                return False, None
            
            height, width = img.shape[:2]
            return True, {
                'frame_count': len(video_path),
                'fps': 30.0,  # Default
                'width': width,
                'height': height
            }
        
        # Convert to Path if it's a string
        if isinstance(video_path, str):
            video_path = Path(video_path)
        
        # Check if it's a directory (image sequence)
        if video_path.is_dir():
            # Validate image sequence directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(video_path.glob(f'*{ext}')))
                image_files.extend(list(video_path.glob(f'*{ext.upper()}')))
            
            if len(image_files) == 0:
                return False, None
            
            # Read first image to get dimensions
            first_img = cv2.imread(str(image_files[0]))
            if first_img is None:
                return False, None
            
            height, width = first_img.shape[:2]
            return True, {
                'frame_count': len(image_files),
                'fps': 30.0,  # Default for image sequences
                'width': width,
                'height': height
            }
        
        # It's a file - check if it's an image file
        # Try reading as image first (for files without extensions like 001, 002, etc.)
        img = cv2.imread(str(video_path))
        if img is not None:
            height, width = img.shape[:2]
            return True, {
                'frame_count': 1,
                'fps': 30.0,  # Default
                'width': width,
                'height': height
            }
        
        # It's a video file
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


def extract_frames_from_image_file(img_file: Path, video_length: int = 32) -> Optional[np.ndarray]:
    """Extract frames from a single image file (for real videos stored as image files)."""
    try:
        # Try to read the file as an image
        img = cv2.imread(str(img_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # If it's a single image, repeat it to match video_length
            frames = [img] * video_length
            return np.array(frames, dtype=np.uint8)
        return None
    except Exception as e:
        print(f"Error extracting frames from image file {img_file}: {e}")
        return None


def extract_frames_from_frame_sequence(frame_paths: List[Path], video_length: int = 32) -> Optional[np.ndarray]:
    """Extract frames from a list of frame file paths (for real videos stored as individual frame files)."""
    try:
        frames = []
        for frame_path in frame_paths:
            # Try to read as image (these files have no extension but are images)
            img = cv2.imread(str(frame_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        
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
        print(f"Error extracting frames from frame sequence: {e}")
        return None


def extract_frames_from_image_sequence(seq_dir: Path, video_length: int = 32) -> Optional[np.ndarray]:
    """Extract frames from an image sequence directory (for real videos)."""
    try:
        # Check if it's actually a file, not a directory
        if seq_dir.is_file():
            return extract_frames_from_image_file(seq_dir, video_length)
        
        # Get all image files in the directory, sorted
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(seq_dir.glob(f'*{ext}')))
            image_files.extend(list(seq_dir.glob(f'*{ext.upper()}')))
        
        if len(image_files) == 0:
            return None
        
        # Sort by filename
        image_files.sort(key=lambda x: x.name)
        
        frames = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        
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
        print(f"Error extracting frames from image sequence {seq_dir}: {e}")
        return None


def extract_frames(video_path, video_length: int = 32) -> Optional[np.ndarray]:
    """
    Extract and sample frames from video file, image file, image sequence directory, or list of frame paths.
    
    Args:
        video_path: Can be:
            - Path to a video file
            - Path to an image file
            - Path to a directory containing images
            - List of Path objects (for frame sequences)
        video_length: Number of frames to extract
    """
    try:
        # Check if it's a list of frame paths (for real videos stored as individual frame files)
        if isinstance(video_path, list):
            return extract_frames_from_frame_sequence(video_path, video_length)
        
        # Convert to Path if it's a string
        if isinstance(video_path, str):
            video_path = Path(video_path)
        
        # Check if it's a directory (image sequence)
        if video_path.is_dir():
            return extract_frames_from_image_sequence(video_path, video_length)
        
        # Check if it's an image file (for real videos stored as image files)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if any(video_path.name.lower().endswith(ext) for ext in image_extensions):
            return extract_frames_from_image_file(video_path, video_length)
        
        # Try reading as image first (for files without extensions like 001, 002, etc.)
        img = cv2.imread(str(video_path))
        if img is not None:
            return extract_frames_from_image_file(video_path, video_length)
        
        # It's a video file
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


def extract_real_zip(zip_path: Path, extract_dir: Path) -> Optional[Path]:
    """Extract a real video zip file and return the extracted directory."""
    try:
        speaker_name = zip_path.stem
        target_dir = extract_dir / speaker_name
        
        # Skip if already extracted (check for video/head directory)
        if target_dir.exists():
            video_head = target_dir / 'video' / 'head'
            if video_head.exists() and any(video_head.iterdir()):
                return target_dir
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        return target_dir
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return None


def group_frames_into_videos(frame_files: List[Path], frames_per_video: int = 30) -> List[List[Path]]:
    """
    Group individual frame files into video sequences.
    
    Args:
        frame_files: List of frame file paths, sorted by frame number
        frames_per_video: Approximate number of frames per video
    
    Returns:
        List of video sequences, where each sequence is a list of frame file paths
    """
    if len(frame_files) == 0:
        return []
    
    # Sort by frame number (extract number from filename)
    def get_frame_number(frame_path: Path) -> int:
        try:
            return int(frame_path.stem)
        except ValueError:
            return 0
    
    sorted_frames = sorted(frame_files, key=get_frame_number)
    
    # Group frames into videos
    video_sequences = []
    current_sequence = []
    
    for frame_file in sorted_frames:
        current_sequence.append(frame_file)
        
        # If we've collected enough frames for a video, start a new sequence
        if len(current_sequence) >= frames_per_video:
            video_sequences.append(current_sequence)
            current_sequence = []
    
    # Add remaining frames as a final video if any
    if len(current_sequence) > 0:
        video_sequences.append(current_sequence)
    
    return video_sequences


def discover_samples(root_dir: Path, quality: str = 'higher_quality', include_real: bool = True) -> List[Dict]:
    """Discover all valid video/audio pairs from both fake and real videos."""
    samples = []
    
    # Discover FAKE videos
    fake_dir = root_dir / 'fake' / 'DeepfakeTIMIT' / quality
    
    if fake_dir.exists():
        print(f"Discovering FAKE samples in {fake_dir}...")
        
        for speaker_dir in tqdm(list(fake_dir.iterdir()), desc="Scanning fake speakers"):
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
                        'quality': quality,
                        'label': 0  # Fake
                    })
    else:
        print(f"Warning: {fake_dir} does not exist")
    
    # Discover REAL videos
    if include_real:
        real_dir = root_dir / 'real'
        if real_dir.exists():
            print(f"\nDiscovering REAL samples in {real_dir}...")
            
            # Create temporary extraction directory
            extract_dir = root_dir / 'real_extracted'
            extract_dir.mkdir(exist_ok=True)
            
            zip_files = list(real_dir.glob('*.zip'))
            if zip_files:
                print(f"Found {len(zip_files)} real video zip files")
                
                for zip_file in tqdm(zip_files, desc="Extracting and scanning real videos"):
                    speaker_name = zip_file.stem
                    extracted_dir = extract_real_zip(zip_file, extract_dir)
                    
                    if extracted_dir is None:
                        continue
                    
                    # Real videos are stored as files in video/head/ directory
                    # Structure: speaker_name/video/head/001, 002, 003, etc. (these are FILES, not directories)
                    # Also check if extracted_dir already contains speaker_name subdirectory
                    speaker_subdir = extracted_dir / speaker_name
                    if not speaker_subdir.exists():
                        # Try without speaker_name prefix (might be extracted directly)
                        speaker_subdir = extracted_dir
                    
                    video_head_dir = speaker_subdir / 'video' / 'head'
                    audio_dir = speaker_subdir / 'audio'
                    
                    if video_head_dir.exists() and audio_dir.exists():
                        # Get all numbered FILES (001, 002, etc.) - these are FRAME files, not complete videos
                        frame_files = [f for f in video_head_dir.iterdir() if f.is_file() and f.name.isdigit()]
                        
                        # Get all audio files
                        audio_files = sorted(list(audio_dir.glob('*.wav')))
                        
                        print(f"  Found {len(frame_files)} frame files and {len(audio_files)} audio files for {speaker_name}")
                        
                        # Group frames into video sequences
                        # Estimate frames per video: if we have N audio files and M frames, 
                        # roughly M/N frames per video (but use a reasonable range)
                        if len(audio_files) > 0 and len(frame_files) > 0:
                            # Estimate frames per video (average, but allow some flexibility)
                            estimated_frames_per_video = max(20, len(frame_files) // len(audio_files))
                            # Use a range: 20-50 frames per video seems reasonable
                            frames_per_video = min(50, max(20, estimated_frames_per_video))
                            
                            video_sequences = group_frames_into_videos(frame_files, frames_per_video)
                            
                            print(f"  Grouped into {len(video_sequences)} video sequences (~{frames_per_video} frames each)")
                            
                            # Match video sequences with audio files
                            for idx, frame_sequence in enumerate(video_sequences):
                                # Cycle through audio files if we have fewer audio than videos
                                audio_idx = idx % len(audio_files)
                                audio_file = audio_files[audio_idx]
                                
                                # Store the sequence as a list of frame paths
                                # We'll process this as a directory-like structure
                                samples.append({
                                    'video_path': frame_sequence,  # This is a list of frame file paths
                                    'audio_path': audio_file,
                                    'speaker': speaker_name,
                                    'quality': 'real',  # Real videos don't have quality levels
                                    'label': 1  # Real
                                })
                    
                    # Also check for traditional video files (avi, mov) as fallback
                    video_files = list(extracted_dir.rglob('*.avi')) + list(extracted_dir.rglob('*.mov'))
                    
                    for video_file in video_files:
                        # Skip if already added as image sequence
                        if any(s['video_path'] == video_file for s in samples):
                            continue
                        
                        # Try to find corresponding audio file
                        audio_file = None
                        
                        # Try same name with .wav
                        audio_file = video_file.parent / f"{video_file.stem}.wav"
                        if not audio_file.exists():
                            # Try looking in audio directory
                            if audio_dir.exists():
                                # Try to find matching audio by name
                                audio_candidates = list(audio_dir.glob(f"{video_file.stem}.wav"))
                                if audio_candidates:
                                    audio_file = audio_candidates[0]
                                else:
                                    # Use first available audio file
                                    wav_files = list(audio_dir.glob('*.wav'))
                                    if wav_files:
                                        audio_file = wav_files[0]
                        
                        if audio_file and audio_file.exists():
                            samples.append({
                                'video_path': video_file,
                                'audio_path': audio_file,
                                'speaker': speaker_name,
                                'quality': 'real',
                                'label': 1  # Real
                            })
            else:
                print("No zip files found in real/ directory")
        else:
            print(f"Warning: {real_dir} does not exist")
    
    print(f"\nTotal samples discovered: {len(samples)}")
    print(f"  FAKE samples: {sum(1 for s in samples if s['label'] == 0)}")
    print(f"  REAL samples: {sum(1 for s in samples if s['label'] == 1)}")
    
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
    skip_existing: bool = True,
    include_real: bool = True
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
    
    # Discover samples (both fake and real)
    samples = discover_samples(root_dir, quality, include_real=include_real)
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
                    
                    # Handle video_path which might be a list of frame paths
                    video_path_str = str(sample['video_path'])
                    if isinstance(sample['video_path'], list):
                        # Convert list of paths to a string representation
                        video_path_str = f"[{', '.join(str(p) for p in sample['video_path'][:3])}...]" if len(sample['video_path']) > 3 else str(sample['video_path'])
                    
                    split_metadata.append({
                        'sample_id': sample_id,
                        'speaker': sample['speaker'],
                        'quality': sample['quality'],
                        'label': sample.get('label', 0),  # Use label from sample (0=fake, 1=real)
                        'original_video': video_path_str,
                        'original_audio': str(sample['audio_path']),
                        'frame_shape': frames.shape,
                        'audio_shape': audio.shape
                    })
                    valid_count += 1
                    continue
                except Exception as e:
                    print(f"Error loading existing {sample_id}: {e}")
            
            # Validate video (handle both Path and list of Paths)
            video_path = sample['video_path']
            video_valid, video_info = validate_video(video_path)
            if not video_valid:
                print(f"Invalid video: {video_path}")
                invalid_count += 1
                continue
            
            # Validate audio
            audio_valid, audio_info = validate_audio(sample['audio_path'], audio_sample_rate)
            if not audio_valid:
                print(f"Invalid audio: {sample['audio_path']}")
                invalid_count += 1
                continue
            
            # Extract frames (handle both Path and list of Paths)
            video_path = sample['video_path']
            frames = extract_frames(video_path, video_length)
            if frames is None:
                print(f"Failed to extract frames: {video_path}")
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
                
                # Handle video_path which might be a list of frame paths
                video_path_str = str(sample['video_path'])
                if isinstance(sample['video_path'], list):
                    # Convert list of paths to a string representation
                    video_path_str = f"[{', '.join(str(p) for p in sample['video_path'][:3])}...]" if len(sample['video_path']) > 3 else str(sample['video_path'])
                
                split_metadata.append({
                    'sample_id': sample_id,
                    'speaker': sample['speaker'],
                    'quality': sample['quality'],
                    'label': sample.get('label', 0),  # Use label from sample (0=fake, 1=real)
                    'original_video': video_path_str,
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
    parser.add_argument('--include_real', action='store_true', default=True,
                       help='Include real videos from real/ directory')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        root_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        quality=args.quality,
        video_length=args.video_length,
        audio_sample_rate=args.audio_sample_rate,
        skip_existing=args.skip_existing,
        include_real=args.include_real
    )


if __name__ == '__main__':
    main()

