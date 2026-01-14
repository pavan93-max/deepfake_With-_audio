"""
Dataset loaders for DeepfakeTIMIT and other datasets.
"""

import os
import cv2
import torch
import torch.utils.data as data
import numpy as np
import soundfile as sf
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import zipfile


class DeepfakeTIMITDataset(data.Dataset):
    """
    Dataset loader for DeepfakeTIMIT.
    """
    
    def __init__(self,
                 root_dir: str,
                 quality: str = 'higher_quality',  # 'higher_quality' or 'lower_quality'
                 split: str = 'train',  # 'train', 'val', 'test'
                 video_length: int = 32,
                 audio_sample_rate: int = 16000,
                 audio_length: Optional[int] = None,  # Fixed audio length in samples
                 transform=None,
                 audio_transform=None,
                 cleaned_dir: Optional[str] = None):
        """
        Args:
            root_dir: Root directory containing fake/ and real/ folders (or cleaned dataset)
            quality: Quality level for fake videos
            split: Dataset split
            video_length: Number of frames to extract
            audio_sample_rate: Target audio sample rate
            audio_length: Fixed audio length in samples (if None, uses video_length * sample_rate / 30)
            transform: Video transforms
            audio_transform: Audio transforms
            cleaned_dir: Path to cleaned/preprocessed dataset (if None, uses raw data)
        """
        self.root_dir = Path(root_dir)
        self.quality = quality
        self.split = split
        self.video_length = video_length
        self.audio_sample_rate = audio_sample_rate
        # Default audio length: match video duration (32 frames @ 30fps ≈ 1.07s)
        self.audio_length = audio_length if audio_length is not None else int(video_length * audio_sample_rate / 30)
        self.transform = transform
        self.audio_transform = audio_transform
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else None
        
        # Check if cleaned dataset is available
        self.use_cleaned = False
        if self.cleaned_dir and self.cleaned_dir.exists():
            metadata_file = self.cleaned_dir / 'metadata.json'
            if metadata_file.exists():
                self.use_cleaned = True
                print(f"Using cleaned dataset from {self.cleaned_dir}")
        
        # Load samples
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load all video samples."""
        # Use cleaned dataset if available
        if self.use_cleaned:
            return self._load_cleaned_samples()
        
        # Otherwise, load from raw data
        samples = []
        
        # Fake videos
        fake_dir = self.root_dir / 'fake' / 'DeepfakeTIMIT' / self.quality
        if fake_dir.exists():
            for speaker_dir in fake_dir.iterdir():
                if speaker_dir.is_dir():
                    # Find video files
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
                                'video_path': str(video_file),
                                'audio_path': str(audio_file),
                                'label': 0,  # Fake
                                'speaker': speaker_dir.name,
                                'quality': self.quality
                            })
        
        # Real videos (from zip files)
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            # For now, we'll need to extract zip files or handle them differently
            # This is a simplified version
            for zip_file in real_dir.glob('*.zip'):
                # In practice, extract and index
                # For now, skip or implement extraction logic
                pass
        
        # Split samples
        if len(samples) == 0:
            print(f"Warning: No samples found in {fake_dir} for quality '{self.quality}'")
            return samples
        
        np.random.seed(42)
        np.random.shuffle(samples)
        
        if self.split == 'train':
            samples = samples[:int(0.7 * len(samples))]
        elif self.split == 'val':
            samples = samples[int(0.7 * len(samples)):int(0.85 * len(samples))]
        else:  # test
            samples = samples[int(0.85 * len(samples)):]
        
        print(f"Loaded {len(samples)} samples for {self.split} split (quality: {self.quality})")
        return samples
    
    def _load_cleaned_samples(self) -> List[Dict]:
        """Load samples from cleaned/preprocessed dataset."""
        metadata_file = self.cleaned_dir / 'metadata.json'
        
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found at {metadata_file}, falling back to raw data")
            self.use_cleaned = False
            return self._load_samples()
        
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
        
        if self.split not in all_metadata:
            print(f"Warning: Split '{self.split}' not found in cleaned dataset, falling back to raw data")
            self.use_cleaned = False
            return self._load_samples()
        
        samples = all_metadata[self.split]
        print(f"Loaded {len(samples)} cleaned samples for {self.split} split")
        return samples
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            # Return black frames if video is empty
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.video_length
        
        # Sample frames
        if len(frames) > self.video_length:
            indices = np.linspace(0, len(frames) - 1, self.video_length, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.video_length:
            # Pad with last frame
            frames.extend([frames[-1]] * (self.video_length - len(frames)))
        
        return np.array(frames)
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio waveform."""
        try:
            audio, sr = sf.read(audio_path)
            
            # Resample if needed
            if sr != self.audio_sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sample_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence
            return np.zeros(self.audio_sample_rate * 2, dtype=np.float32)  # 2 seconds
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        if self.use_cleaned:
            # Load from cleaned dataset
            sample_id = sample['sample_id']
            frame_file = self.cleaned_dir / 'frames' / f"{sample_id}.npy"
            audio_file = self.cleaned_dir / 'audio' / f"{sample_id}.npy"
            
            # Load preprocessed frames and audio
            video_frames = np.load(frame_file)  # (T, H, W, C)
            audio = np.load(audio_file)  # (T_a,)
        else:
            # Load from raw data
            video_frames = self._load_video(sample['video_path'])
            audio = self._load_audio(sample['audio_path'])
        
        # Pad or truncate audio to fixed length
        if len(audio) > self.audio_length:
            # Truncate
            audio = audio[:self.audio_length]
        elif len(audio) < self.audio_length:
            # Pad with zeros
            padding = np.zeros(self.audio_length - len(audio), dtype=audio.dtype)
            audio = np.concatenate([audio, padding])
        
        # Apply transforms
        if self.transform:
            video_frames = self.transform(video_frames)
        
        if self.audio_transform:
            audio = self.audio_transform(audio)
        
        # Convert to tensors
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        audio_tensor = torch.from_numpy(audio).float()
        
        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'speaker': sample.get('speaker', 'unknown'),
            'video_path': sample.get('original_video', sample.get('video_path', ''))
        }


class AVDesyncAugmentation:
    """
    Audio-visual desynchronization augmentation.
    """
    
    def __init__(self, max_desync_ms: int = 120):
        self.max_desync_ms = max_desync_ms
    
    def __call__(self, video: torch.Tensor, audio: torch.Tensor, sample_rate: int = 16000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random desynchronization.
        
        Args:
            video: (T, C, H, W) video frames
            audio: (T_a,) audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            Desynchronized video and audio
        """
        # Random desync in milliseconds
        desync_ms = np.random.uniform(-self.max_desync_ms, self.max_desync_ms)
        desync_samples = int(desync_ms * sample_rate / 1000)
        
        if desync_samples > 0:
            # Audio ahead: pad video at start
            video = torch.cat([
                video[0:1].repeat(desync_samples // (1000 // 30), 1, 1, 1),  # Approximate frame rate
                video
            ], dim=0)
            audio = audio[desync_samples:]
        elif desync_samples < 0:
            # Video ahead: pad audio at start
            audio = torch.cat([
                torch.zeros(-desync_samples, device=audio.device),
                audio
            ], dim=0)
            video = video[-desync_samples // (1000 // 30):]
        
        return video, audio

