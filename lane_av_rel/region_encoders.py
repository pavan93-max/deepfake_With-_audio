"""
Per-region encoders: CNN branch + tiny ViT with artifact channel fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import timm
from .frequency_artifacts import FrequencyArtifactExtractor


class SpatialKernelSelection(nn.Module):
    """
    Adaptive kernel selection layer (Halo/Spatial-Kernel-Selection).
    Small kernels for eyelids, larger for cheeks.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple = (3, 5, 7)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # Separate convs for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Attention weights for kernel selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(kernel_sizes), 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            
        Returns:
            (B, out_channels, H, W)
        """
        # Compute attention weights
        attn = self.attention(x)  # (B, num_kernels, 1, 1)
        
        # Apply each convolution
        conv_outputs = [conv(x) for conv in self.convs]
        conv_stack = torch.stack(conv_outputs, dim=1)  # (B, num_kernels, out_channels, H, W)
        
        # Weighted combination
        attn = attn.unsqueeze(2)  # (B, num_kernels, 1, 1, 1)
        output = (conv_stack * attn).sum(dim=1)  # (B, out_channels, H, W)
        
        return output


class CNNBranch(nn.Module):
    """
    Lightweight CNN branch for local texture features.
    """
    
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 32,
                 use_adaptive_kernel: bool = True,
                 region_type: str = 'eye'):
        super().__init__()
        self.region_type = region_type
        
        # Adaptive kernel selection based on region
        if use_adaptive_kernel:
            if region_type in ['left_eye', 'right_eye']:
                kernel_sizes = (3, 5)  # Smaller for eyes
            elif region_type == 'mouth':
                kernel_sizes = (3, 5, 7)
            else:
                kernel_sizes = (5, 7)  # Larger for cheeks/jawline
        else:
            kernel_sizes = (3, 5)
        
        # Feature extraction layers
        layers = []
        
        # First block with adaptive kernel
        if use_adaptive_kernel:
            layers.append(SpatialKernelSelection(in_channels, base_channels, kernel_sizes))
        else:
            layers.append(nn.Conv2d(in_channels, base_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        
        # Second block
        layers.append(nn.Conv2d(base_channels, base_channels * 2, 3, padding=1))
        layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        
        # Third block
        layers.append(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1))
        layers.append(nn.BatchNorm2d(base_channels * 4))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.cnn = nn.Sequential(*layers)
        self.out_channels = base_channels * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            
        Returns:
            (B, out_channels)
        """
        return self.cnn(x).squeeze(-1).squeeze(-1)


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer for global within-ROI features.
    """
    
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 8,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 depth: int = 4,
                 num_heads: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification token (optional, or use mean pooling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            
        Returns:
            (B, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Transformer
        x = self.transformer(x)
        
        # Use cls token
        x = x[:, 0]  # (B, embed_dim)
        
        # Normalize and project
        x = self.norm(x)
        x = self.head(x)
        
        return x


class PerRegionEncoder(nn.Module):
    """
    Per-region encoder combining CNN, ViT, and artifact channels.
    """
    
    def __init__(self,
                 region_name: str,
                 roi_size: int = 64,
                 in_channels: int = 3,
                 cnn_channels: int = 32,
                 vit_embed_dim: int = 128,
                 use_artifacts: bool = True,
                 artifact_config: Optional[Dict] = None):
        super().__init__()
        self.region_name = region_name
        self.roi_size = roi_size
        self.use_artifacts = use_artifacts
        
        # Artifact extractor
        if use_artifacts:
            artifact_config = artifact_config or {}
            self.artifact_extractor = FrequencyArtifactExtractor(**artifact_config)
            artifact_channels = self.artifact_extractor.get_output_channels(in_channels)
        else:
            self.artifact_extractor = None
            artifact_channels = 0
        
        # CNN branch
        self.cnn = CNNBranch(
            in_channels=in_channels + artifact_channels,
            base_channels=cnn_channels,
            region_type=region_name
        )
        
        # ViT branch
        self.vit = TinyViT(
            img_size=roi_size,
            patch_size=8,
            in_channels=in_channels,
            embed_dim=vit_embed_dim
        )
        
        # Fusion layer
        total_dim = self.cnn.out_channels + vit_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LayerNorm(total_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.out_dim = total_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) ROI tensor
            
        Returns:
            (B, out_dim) region embedding
        """
        # Extract artifacts
        if self.use_artifacts and self.artifact_extractor is not None:
            artifacts = self.artifact_extractor(x)
            # Concatenate all artifact channels
            artifact_list = list(artifacts.values())
            if artifact_list:
                # Resize artifacts to match input size if needed
                artifact_tensor = torch.cat([
                    F.interpolate(a, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                    if a.shape[2:] != x.shape[2:] else a
                    for a in artifact_list
                ], dim=1)
                x_with_artifacts = torch.cat([x, artifact_tensor], dim=1)
            else:
                x_with_artifacts = x
        else:
            x_with_artifacts = x
        
        # CNN branch
        cnn_feat = self.cnn(x_with_artifacts if self.use_artifacts else x)
        
        # ViT branch (on original image)
        vit_feat = self.vit(x)
        
        # Concatenate and fuse
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        output = self.fusion(combined)
        
        return output


class MultiRegionEncoder(nn.Module):
    """
    Encoder for all face regions.
    """
    
    REGION_NAMES = [
        'left_eye', 'right_eye', 'mouth', 'nose',
        'left_cheek', 'right_cheek', 'jawline'
    ]
    
    def __init__(self,
                 roi_size: int = 64,
                 in_channels: int = 3,
                 cnn_channels: int = 32,
                 vit_embed_dim: int = 128,
                 use_artifacts: bool = True,
                 artifact_config: Optional[Dict] = None):
        super().__init__()
        
        # Create encoder for each region
        self.encoders = nn.ModuleDict({
            name: PerRegionEncoder(
                region_name=name,
                roi_size=roi_size,
                in_channels=in_channels,
                cnn_channels=cnn_channels,
                vit_embed_dim=vit_embed_dim,
                use_artifacts=use_artifacts,
                artifact_config=artifact_config
            )
            for name in self.REGION_NAMES
        })
        
        # Store output dimensions
        self.region_dims = {name: enc.out_dim for name, enc in self.encoders.items()}
    
    def forward(self, rois: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            rois: Dictionary of (B, C, H, W) tensors for each region
            
        Returns:
            Dictionary of (B, embed_dim) region embeddings
        """
        embeddings = {}
        for name, encoder in self.encoders.items():
            if name in rois:
                embeddings[name] = encoder(rois[name])
            else:
                # Zero padding if region missing
                embeddings[name] = torch.zeros(
                    rois[list(rois.keys())[0]].shape[0],
                    encoder.out_dim,
                    device=rois[list(rois.keys())[0]].device
                )
        
        return embeddings

