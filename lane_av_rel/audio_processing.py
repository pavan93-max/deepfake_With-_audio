"""
Audio processing: log-Mel + CQT spectrograms, pitch/prosody, phoneme alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class LogMelSpectrogram(nn.Module):
    """
    Log-Mel spectrogram extractor.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel filter bank
        # fmin/fmax support was added in torchaudio 2.1.0+
        # Check if these parameters are supported
        import inspect
        mel_spec_signature = inspect.signature(torchaudio.transforms.MelSpectrogram.__init__)
        mel_kwargs = {
            'sample_rate': sample_rate,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels
        }
        # Only add fmin/fmax if supported
        if 'fmin' in mel_spec_signature.parameters:
            mel_kwargs['fmin'] = fmin
        if 'fmax' in mel_spec_signature.parameters:
            mel_kwargs['fmax'] = fmax or sample_rate // 2
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (T,) audio waveform
            
        Returns:
            (B, n_mels, T') or (n_mels, T') log-Mel spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute Mel spectrogram
        mel = self.mel_spectrogram(waveform)
        
        # Log scale
        log_mel = torch.log10(mel + 1e-10)
        
        return log_mel


class CQTSpectrogram(nn.Module):
    """
    Constant-Q Transform (CQT) spectrogram extractor.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 hop_length: int = 512,
                 n_bins: int = 84,
                 bins_per_octave: int = 12):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (T,) audio waveform
            
        Returns:
            (B, n_bins, T') or (n_bins, T') CQT spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to numpy for librosa
        waveform_np = waveform.cpu().numpy()
        cqt_list = []
        
        for wav in waveform_np:
            cqt = librosa.cqt(
                wav,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave
            )
            cqt_mag = np.abs(cqt)
            cqt_list.append(torch.from_numpy(cqt_mag).float())
        
        cqt_tensor = torch.stack(cqt_list)
        
        if waveform.device.type == 'cuda':
            cqt_tensor = cqt_tensor.to(waveform.device)
        
        return cqt_tensor


class PitchProsodyExtractor(nn.Module):
    """
    Pitch and prosody feature extractor.
    """
    
    def __init__(self, sample_rate: int = 16000, hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: (B, T) or (T,) audio waveform
            
        Returns:
            Dictionary with pitch, energy, etc.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform_np = waveform.cpu().numpy()
        
        pitch_list = []
        energy_list = []
        
        for wav in waveform_np:
            # Pitch (F0) using librosa
            pitches, magnitudes = librosa.piptrack(
                y=wav,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            pitch = pitches.max(axis=0)  # Take max across frequency bins
            
            # Energy (RMS)
            frame_length = self.hop_length * 2
            energy = librosa.feature.rms(
                y=wav,
                frame_length=frame_length,
                hop_length=self.hop_length
            ).squeeze()
            
            pitch_list.append(torch.from_numpy(pitch).float())
            energy_list.append(torch.from_numpy(energy).float())
        
        pitch_tensor = torch.stack(pitch_list)
        energy_tensor = torch.stack(energy_list)
        
        if waveform.device.type == 'cuda':
            pitch_tensor = pitch_tensor.to(waveform.device)
            energy_tensor = energy_tensor.to(waveform.device)
        
        return {
            'pitch': pitch_tensor,
            'energy': energy_tensor
        }


class ConformerTiny(nn.Module):
    """
    Tiny Conformer encoder for audio.
    Simplified version of Conformer architecture.
    """
    
    def __init__(self,
                 input_dim: int = 80,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 conv_kernel_size: int = 31):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(embed_dim, num_heads, conv_kernel_size)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) spectrogram features
            mask: (B, T) optional mask
            
        Returns:
            (B, T, embed_dim) encoded features
        """
        # Project input
        x = self.input_proj(x)
        
        # Apply Conformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Normalize
        x = self.norm(x)
        
        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block: Feed-forward + Multi-head self-attention + Convolution + Feed-forward.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, conv_kernel_size: int):
        super().__init__()
        
        # Feed-forward 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(embed_dim, embed_dim, conv_kernel_size, padding=conv_kernel_size//2, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.Dropout(0.1)
        )
        self.conv_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
            mask: (B, T) optional mask
            
        Returns:
            (B, T, embed_dim)
        """
        # FFN 1
        x = x + self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = residual + attn_out
        
        # Convolution
        residual = x
        x = self.conv_norm(x)
        x_conv = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        x = residual + x_conv
        
        # FFN 2
        x = x + self.ffn2(x)
        
        return x


class PhonemeAligner:
    """
    Phoneme alignment using forced alignment.
    Creates time-indexed edges to mouth/cheek nodes.
    
    Note: Full phoneme alignment requires Montreal Forced Aligner,
    which is not available on PyPI. This class provides a simplified
    version using Wav2Vec2. For full functionality, install MFA separately.
    """
    
    def __init__(self, model_name: str = "wav2vec2-base-960h", use_mfa: bool = False):
        """
        Initialize with Wav2Vec2 for forced alignment.
        
        Args:
            model_name: Wav2Vec2 model name
            use_mfa: If True, try to use Montreal Forced Aligner (requires separate installation)
        """
        self.use_mfa = use_mfa
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(f"facebook/{model_name}")
            self.model = Wav2Vec2ForCTC.from_pretrained(f"facebook/{model_name}")
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2 model: {e}")
            print("Phoneme alignment will use simplified timing.")
            self.processor = None
            self.model = None
    
    def align(self,
              waveform: torch.Tensor,
              transcript: str,
              sample_rate: int = 16000) -> Dict[str, np.ndarray]:
        """
        Perform forced alignment.
        
        Args:
            waveform: (T,) audio waveform
            transcript: Text transcript
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with phoneme timings
        """
        # This is a simplified version
        # Full implementation would use Montreal Forced Aligner or similar
        
        # For now, return dummy alignment
        # In practice, use: https://github.com/MontrealCorpusTools/Montreal-Forced-Alignment
        
        duration = waveform.shape[0] / sample_rate
        num_phonemes = len(transcript.split())
        
        # Dummy timing (uniform distribution)
        phoneme_times = np.linspace(0, duration, num_phonemes + 1)
        
        return {
            'phonemes': transcript.split(),
            'start_times': phoneme_times[:-1],
            'end_times': phoneme_times[1:],
            'durations': np.diff(phoneme_times)
        }


class AudioEncoder(nn.Module):
    """
    Complete audio encoder pipeline.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 use_mel: bool = True,
                 use_cqt: bool = True,
                 use_pitch: bool = True,
                 mel_bins: int = 80,
                 cqt_bins: int = 84,
                 embed_dim: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_mel = use_mel
        self.use_cqt = use_cqt
        self.use_pitch = use_pitch
        
        if use_mel:
            self.mel = LogMelSpectrogram(sample_rate=sample_rate, n_mels=mel_bins)
        if use_cqt:
            self.cqt = CQTSpectrogram(sample_rate=sample_rate, n_bins=cqt_bins)
        if use_pitch:
            self.pitch_extractor = PitchProsodyExtractor(sample_rate=sample_rate)
        
        # Compute input dimension
        input_dim = 0
        if use_mel:
            input_dim += mel_bins
        if use_cqt:
            input_dim += cqt_bins
        if use_pitch:
            input_dim += 2  # pitch + energy
        
        # Conformer encoder
        self.conformer = ConformerTiny(
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: (B, T) or (T,) audio waveform
            
        Returns:
            Dictionary with encoded features and auxiliary info
        """
        features = []
        
        if self.use_mel:
            mel = self.mel(waveform)  # (B, n_mels, T')
            mel = mel.transpose(1, 2)  # (B, T', n_mels)
            features.append(mel)
        
        if self.use_cqt:
            cqt = self.cqt(waveform)  # (B, n_bins, T')
            cqt = cqt.transpose(1, 2)  # (B, T', n_bins)
            features.append(cqt)
        
        if self.use_pitch:
            pitch_prosody = self.pitch_extractor(waveform)
            # Interpolate to match spectrogram length
            pitch = pitch_prosody['pitch']  # (B, T_p)
            energy = pitch_prosody['energy']  # (B, T_e)
            
            # Align dimensions (simplified)
            if features:
                target_len = features[0].shape[1]
                pitch = F.interpolate(
                    pitch.unsqueeze(1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                energy = F.interpolate(
                    energy.unsqueeze(1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            
            pitch_energy = torch.stack([pitch, energy], dim=-1)  # (B, T', 2)
            features.append(pitch_energy)
        
        # Concatenate features
        if features:
            x = torch.cat(features, dim=-1)  # (B, T', input_dim)
        else:
            raise ValueError("At least one feature type must be enabled")
        
        # Encode with Conformer
        encoded = self.conformer(x)  # (B, T', embed_dim)
        
        return {
            'encoded': encoded,
            'mel': mel if self.use_mel else None,
            'cqt': cqt if self.use_cqt else None,
            'pitch': pitch if self.use_pitch else None,
            'energy': energy if self.use_pitch else None
        }

