"""
Loss functions: CE + AV consistency, temporal consistency, calibration, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class AVConsistencyLoss(nn.Module):
    """
    Audio-visual consistency loss.
    Contrastive + additive angular margin.
    """
    
    def __init__(self, margin: float = 0.3, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self,
                visual_features: torch.Tensor,
                audio_features: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, D_v) visual embeddings
            audio_features: (B, D_a) audio embeddings
            labels: (B,) binary labels (0=fake, 1=real)
            
        Returns:
            Dictionary with contrastive and angular margin losses
        """
        # Normalize features
        v_norm = F.normalize(visual_features, dim=-1)
        a_norm = F.normalize(audio_features, dim=-1)
        
        # Project to same dimension
        if v_norm.shape[-1] != a_norm.shape[-1]:
            min_dim = min(v_norm.shape[-1], a_norm.shape[-1])
            v_norm = v_norm[..., :min_dim]
            a_norm = a_norm[..., :min_dim]
        
        # Contrastive loss (InfoNCE)
        # Positive pairs: same sample, negative pairs: different samples
        similarity = torch.matmul(v_norm, a_norm.t()) / self.temperature  # (B, B)
        
        # Positive pairs are diagonal
        labels_pos = torch.arange(v_norm.shape[0], device=v_norm.device)
        contrastive_loss = F.cross_entropy(similarity, labels_pos)
        
        # Angular margin loss (for real samples, encourage alignment)
        # For fake samples, encourage misalignment
        cos_sim = (v_norm * a_norm).sum(dim=-1)  # (B,)
        
        # Real samples (label=1): should have high similarity
        # Fake samples (label=0): should have low similarity
        real_mask = (labels == 1).float()
        fake_mask = (labels == 0).float()
        
        # Angular margin for real samples
        real_sim = cos_sim * real_mask
        angular_margin_real = F.relu(self.margin - real_sim).mean()
        
        # Penalize high similarity for fake samples
        fake_sim = cos_sim * fake_mask
        angular_margin_fake = F.relu(fake_sim - (1 - self.margin)).mean()
        
        angular_loss = angular_margin_real + angular_margin_fake
        
        return {
            'contrastive_loss': contrastive_loss,
            'angular_loss': angular_loss,
            'total_loss': contrastive_loss + angular_loss
        }


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_maps: (B, T, N) attention maps over time
            
        Returns:
            Temporal consistency loss
        """
        if attention_maps.dim() != 3:
            return torch.tensor(0.0, device=attention_maps.device)
        
        # Compute temporal differences
        diff = attention_maps[:, 1:] - attention_maps[:, :-1]  # (B, T-1, N)
        
        # L2 penalty
        loss = (diff ** 2).mean()
        
        return loss


class EdgeSparsityRegularizer(nn.Module):
    """
    Edge sparsity regularizer for graph.
    Encourages sparse graph structure.
    """
    
    def __init__(self, lambda_sparse: float = 0.01):
        super().__init__()
        self.lambda_sparse = lambda_sparse
    
    def forward(self, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_weights: (E,) edge weights or attention scores
            
        Returns:
            Sparsity regularization loss
        """
        # L1 penalty on edge weights
        loss = self.lambda_sparse * edge_weights.abs().mean()
        
        return loss


class CalibrationLoss(nn.Module):
    """
    Calibration loss (MMCE - Maximum Mean Calibration Error).
    Encourages well-calibrated predictions.
    """
    
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
    
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, num_classes) predicted probabilities
            targets: (B,) true labels
            
        Returns:
            MMCE loss
        """
        # Get confidence (max probability) and accuracy
        confidence = probs.max(dim=-1)[0]  # (B,)
        predictions = probs.argmax(dim=-1)  # (B,)
        accuracy = (predictions == targets).float()  # (B,)
        
        # Bin confidence scores
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mmce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                
                # Calibration error for this bin
                bin_error = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                mmce += prop_in_bin * bin_error
        
        return mmce


class CombinedLoss(nn.Module):
    """
    Combined loss function for training.
    """
    
    def __init__(self,
                 ce_weight: float = 1.0,
                 av_weight: float = 0.5,
                 temporal_weight: float = 0.1,
                 edge_sparse_weight: float = 0.01,
                 calibration_weight: float = 0.1,
                 use_av: bool = True,
                 use_temporal: bool = True,
                 use_calibration: bool = True):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.av_weight = av_weight
        self.temporal_weight = temporal_weight
        self.edge_sparse_weight = edge_sparse_weight
        self.calibration_weight = calibration_weight
        
        self.use_av = use_av
        self.use_temporal = use_temporal
        self.use_calibration = use_calibration
        
        self.ce_loss = CrossEntropyLoss()
        self.av_loss = AVConsistencyLoss() if use_av else None
        self.temporal_loss = TemporalConsistencyLoss() if use_temporal else None
        self.edge_sparse = EdgeSparsityRegularizer() if edge_sparse_weight > 0 else None
        self.calibration_loss = CalibrationLoss() if use_calibration else None
    
    def forward(self,
                logits: torch.Tensor,
                probs: torch.Tensor,
                targets: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None,
                attention_maps: Optional[torch.Tensor] = None,
                edge_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: (B, num_classes) classification logits
            probs: (B, num_classes) predicted probabilities
            targets: (B,) true labels
            visual_features: Optional visual embeddings
            audio_features: Optional audio embeddings
            attention_maps: Optional attention maps for temporal loss
            edge_weights: Optional edge weights for sparsity
            
        Returns:
            Dictionary with all losses
        """
        losses = {}
        
        # Cross-entropy
        ce = self.ce_loss(logits, targets)
        losses['ce'] = ce
        total_loss = self.ce_weight * ce
        
        # AV consistency
        if self.use_av and visual_features is not None and audio_features is not None:
            av_losses = self.av_loss(visual_features, audio_features, targets)
            losses.update({f'av_{k}': v for k, v in av_losses.items()})
            total_loss += self.av_weight * av_losses['total_loss']
        
        # Temporal consistency
        if self.use_temporal and attention_maps is not None:
            temporal = self.temporal_loss(attention_maps)
            losses['temporal'] = temporal
            total_loss += self.temporal_weight * temporal
        
        # Edge sparsity
        if self.edge_sparse is not None and edge_weights is not None:
            edge_sparse = self.edge_sparse(edge_weights)
            losses['edge_sparse'] = edge_sparse
            total_loss += self.edge_sparse_weight * edge_sparse
        
        # Calibration
        if self.use_calibration:
            calibration = self.calibration_loss(probs, targets)
            losses['calibration'] = calibration
            total_loss += self.calibration_weight * calibration
        
        losses['total'] = total_loss
        
        return losses

