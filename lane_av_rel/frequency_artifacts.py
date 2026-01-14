"""
Frequency artifact channels per face region.
Implements DWT (LL/LH/HL/HH), Gabor bank, and FFT magnitude/phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Tuple, Dict
import cv2


class DWTChannelExtractor(nn.Module):
    """
    Discrete Wavelet Transform (DWT) channel extractor.
    Extracts LL, LH, HL, HH subbands.
    """
    
    def __init__(self, wavelet: str = 'haar', levels: int = 2):
        """
        Args:
            wavelet: Wavelet type ('haar', 'db4', etc.)
            levels: Number of decomposition levels
        """
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            (B, C*4, H//2, W//2) DWT coefficients
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Convert to numpy for PyWavelets
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        
        dwt_coeffs = []
        for b in range(B):
            for c in range(C):
                img = x_np[b, :, :, c]
                
                # DWT decomposition
                coeffs = pywt.dwt2(img, self.wavelet)
                LL, (LH, HL, HH) = coeffs
                
                # Stack subbands
                dwt_coeffs.append(torch.stack([
                    torch.from_numpy(LL).to(device),
                    torch.from_numpy(LH).to(device),
                    torch.from_numpy(HL).to(device),
                    torch.from_numpy(HH).to(device)
                ]))
        
        # Reshape: (B, C, 4, H//2, W//2) -> (B, C*4, H//2, W//2)
        dwt_tensor = torch.stack(dwt_coeffs).view(B, C, 4, H//2, W//2)
        dwt_tensor = dwt_tensor.view(B, C*4, H//2, W//2)
        
        return dwt_tensor


class GaborBank(nn.Module):
    """
    Gabor filter bank for texture analysis.
    Multiple orientations and scales.
    """
    
    def __init__(self,
                 num_orientations: int = 6,
                 wavelengths: Tuple[float, ...] = (8, 10, 12),
                 sigma: float = 6.0,
                 gamma: float = 0.5):
        """
        Args:
            num_orientations: Number of orientation angles
            wavelengths: List of wavelengths (λ)
            sigma: Standard deviation of Gaussian envelope
            gamma: Spatial aspect ratio
        """
        super().__init__()
        self.num_orientations = num_orientations
        self.wavelengths = wavelengths
        self.sigma = sigma
        self.gamma = gamma
        
        # Create Gabor kernels
        self.kernels = self._create_kernels()
        
    def _create_kernels(self) -> torch.Tensor:
        """Create Gabor filter kernels."""
        kernel_size = int(6 * self.sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernels = []
        for wavelength in self.wavelengths:
            for theta_idx in range(self.num_orientations):
                theta = theta_idx * np.pi / self.num_orientations
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    self.sigma,
                    theta,
                    wavelength,
                    self.gamma,
                    0,
                    ktype=cv2.CV_32F
                )
                kernels.append(torch.from_numpy(kernel).float())
        
        # Stack: (num_filters, 1, H, W)
        return torch.stack(kernels).unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            (B, C*num_filters, H, W) Gabor responses
        """
        B, C, H, W = x.shape
        device = x.device
        kernels = self.kernels.to(device)
        num_filters = kernels.shape[0]
        
        # Apply Gabor filters
        responses = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # (B, 1, H, W)
            
            # Convolve with each Gabor kernel
            channel_responses = []
            for k in range(num_filters):
                kernel = kernels[k:k+1]  # (1, 1, H_k, W_k)
                response = F.conv2d(channel, kernel, padding=kernel.shape[-1]//2)
                channel_responses.append(response)
            
            # Stack: (B, num_filters, H, W)
            channel_responses = torch.cat(channel_responses, dim=1)
            responses.append(channel_responses)
        
        # Concatenate across channels: (B, C*num_filters, H, W)
        return torch.cat(responses, dim=1)


class FFTChannelExtractor(nn.Module):
    """
    FFT magnitude and phase channel extractor.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            (B, C*2, H, W) FFT magnitude and phase
        """
        B, C, H, W = x.shape
        
        # Convert to complex (rfft2 returns (B, C, H, W//2+1))
        x_complex = torch.fft.rfft2(x, norm='ortho')
        
        # Extract magnitude and phase
        magnitude = torch.abs(x_complex)  # (B, C, H, W//2+1)
        phase = torch.angle(x_complex)     # (B, C, H, W//2+1)
        
        # Pad both to match original spatial dimensions
        # Pad format: (pad_left, pad_right, pad_top, pad_bottom)
        # For last dimension (width): pad from W//2+1 to W
        pad_width = W - magnitude.shape[-1]
        magnitude_padded = F.pad(magnitude, (0, pad_width))  # (B, C, H, W)
        phase_padded = F.pad(phase, (0, pad_width))          # (B, C, H, W)
        
        # Concatenate: (B, C*2, H, W)
        return torch.cat([magnitude_padded, phase_padded], dim=1)


class FrequencyArtifactExtractor(nn.Module):
    """
    Complete frequency artifact extractor combining DWT, Gabor, and FFT.
    """
    
    def __init__(self,
                 use_dwt: bool = True,
                 use_gabor: bool = True,
                 use_fft: bool = True,
                 dwt_wavelet: str = 'haar',
                 gabor_orientations: int = 6,
                 gabor_wavelengths: Tuple[float, ...] = (8, 10, 12)):
        super().__init__()
        
        self.use_dwt = use_dwt
        self.use_gabor = use_gabor
        self.use_fft = use_fft
        
        if use_dwt:
            self.dwt = DWTChannelExtractor(wavelet=dwt_wavelet)
        if use_gabor:
            self.gabor = GaborBank(
                num_orientations=gabor_orientations,
                wavelengths=gabor_wavelengths
            )
        if use_fft:
            self.fft = FFTChannelExtractor()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input ROI tensor
            
        Returns:
            Dictionary with artifact channels
        """
        artifacts = {}
        
        if self.use_dwt:
            artifacts['dwt'] = self.dwt(x)
        
        if self.use_gabor:
            artifacts['gabor'] = self.gabor(x)
        
        if self.use_fft:
            artifacts['fft'] = self.fft(x)
        
        return artifacts
    
    def get_output_channels(self, input_channels: int) -> int:
        """Compute total output channels."""
        total = 0
        if self.use_dwt:
            total += input_channels * 4  # LL, LH, HL, HH
        if self.use_gabor:
            total += input_channels * len(self.gabor.wavelengths) * self.gabor.num_orientations
        if self.use_fft:
            total += input_channels * 2  # magnitude, phase
        return total

