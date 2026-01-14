"""
Temporal modeling: attention/LSTM per region with temporal persistence loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention module for region sequences.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) sequence of region embeddings
            mask: (B, T) optional mask
            
        Returns:
            (B, T, embed_dim) attended sequence
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = residual + attn_out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class CompactLSTM(nn.Module):
    """
    Compact LSTM for temporal modeling.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            
        Returns:
            (B, T, hidden_dim)
        """
        out, _ = self.lstm(x)
        return out


class RegionTemporalEncoder(nn.Module):
    """
    Temporal encoder for a single region.
    """
    
    def __init__(self,
                 embed_dim: int,
                 use_attention: bool = True,
                 use_lstm: bool = False,
                 num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        
        if use_attention:
            self.attention = TemporalAttention(embed_dim, num_heads)
        
        if use_lstm:
            self.lstm = CompactLSTM(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) sequence
            mask: (B, T) optional mask
            
        Returns:
            (B, T, embed_dim) or (B, embed_dim) if pooling
        """
        if self.use_attention:
            x = self.attention(x, mask)
        
        if self.use_lstm:
            x = self.lstm(x)
        
        x = self.output_proj(x)
        
        return x


class MultiRegionTemporalEncoder(nn.Module):
    """
    Temporal encoder for all regions.
    """
    
    def __init__(self,
                 region_dims: Dict[str, int],
                 use_attention: bool = True,
                 use_lstm: bool = False,
                 num_heads: int = 4):
        super().__init__()
        
        self.region_encoders = nn.ModuleDict({
            name: RegionTemporalEncoder(
                embed_dim=dim,
                use_attention=use_attention,
                use_lstm=use_lstm,
                num_heads=num_heads
            )
            for name, dim in region_dims.items()
        })
    
    def forward(self,
                region_sequences: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            region_sequences: Dict of (B, T, embed_dim) sequences
            masks: Optional dict of (B, T) masks
            
        Returns:
            Dict of (B, T, embed_dim) encoded sequences
        """
        masks = masks or {}
        
        encoded = {}
        for name, encoder in self.region_encoders.items():
            if name in region_sequences:
                mask = masks.get(name)
                encoded[name] = encoder(region_sequences[name], mask)
            else:
                # Zero padding
                seq = region_sequences[list(region_sequences.keys())[0]]
                encoded[name] = torch.zeros_like(seq)
        
        return encoded


class TemporalPersistenceLoss(nn.Module):
    """
    Temporal persistence loss to prevent artifact attention from "jumping".
    Encourages smooth temporal transitions.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_maps: (B, T, H, W) or (B, T, N) attention maps
            
        Returns:
            Scalar loss
        """
        # Compute temporal differences
        if attention_maps.dim() == 4:
            # (B, T, H, W) -> (B, T-1, H, W)
            diff = attention_maps[:, 1:] - attention_maps[:, :-1]
        else:
            # (B, T, N) -> (B, T-1, N)
            diff = attention_maps[:, 1:] - attention_maps[:, :-1]
        
        # L2 penalty on differences (encourage smoothness)
        loss = (diff ** 2).mean()
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss using KL divergence on attention maps.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, attn_t: torch.Tensor, attn_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_t: (B, N) attention at time t
            attn_t1: (B, N) attention at time t+1
            
        Returns:
            KL divergence loss
        """
        # Normalize to probabilities
        attn_t = F.softmax(attn_t / 0.1, dim=-1)
        attn_t1 = F.softmax(attn_t1 / 0.1, dim=-1)
        
        # KL divergence
        kl = F.kl_div(
            F.log_softmax(attn_t / 0.1, dim=-1),
            attn_t1,
            reduction='batchmean'
        )
        
        return kl

