"""
Multi-modal relation graph with masked-relation learning.
Nodes: region embeddings + audio segments.
Edges: spatial adjacency, frequency similarity, AV timing, blink/prosody links.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional imports for graph neural networks
try:
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Create dummy classes for when torch_geometric is not available
    class GATConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("torch_geometric is not installed. Install with: pip install torch-geometric")
        def forward(self, *args, **kwargs):
            pass
    class Data:
        pass


class MultiModalRelationGraph(nn.Module):
    """
    Multi-modal relation graph for deepfake detection.
    """
    
    def __init__(self,
                 region_dims: Dict[str, int],
                 audio_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.region_dims = region_dims
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Project all nodes to same dimension
        self.region_projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in region_dims.items()
        })
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        
        # Graph attention layers
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for relation graph. "
                "Install with: pip install torch-geometric\n"
                "Or set use_graph=False in model config to disable graph features."
            )
        
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=(i < num_layers - 1)  # Last layer averages heads
            )
            for i in range(num_layers)
        ])
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(4, hidden_dim)  # 4 edge types
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def build_graph(self,
                    region_embeddings: Dict[str, torch.Tensor],
                    audio_embeddings: torch.Tensor,
                    phoneme_times: Optional[Dict] = None) -> Data:
        """
        Build graph from region and audio embeddings.
        
        Args:
            region_embeddings: Dict of (B, T, embed_dim) region sequences
            audio_embeddings: (B, T_a, audio_dim) audio sequence
            phoneme_times: Optional phoneme timing information
            
        Returns:
            PyG Data object
        """
        B, T_a, _ = audio_embeddings.shape
        
        # Project all nodes
        node_features = []
        node_types = []
        
        # Region nodes (per time step)
        for name, emb in region_embeddings.items():
            B_r, T_r, D_r = emb.shape
            # Project and flatten
            proj_emb = self.region_projections[name](emb)  # (B, T_r, hidden_dim)
            node_features.append(proj_emb.view(B * T_r, self.hidden_dim))
            node_types.extend([f'region_{name}'] * (B * T_r))
        
        # Audio nodes
        proj_audio = self.audio_projection(audio_embeddings)  # (B, T_a, hidden_dim)
        node_features.append(proj_audio.view(B * T_a, self.hidden_dim))
        node_types.extend(['audio'] * (B * T_a))
        
        # Stack all nodes
        x = torch.cat(node_features, dim=0)  # (N_total, hidden_dim)
        
        # Build edges
        edge_index = []
        edge_attr = []
        
        # 1. Spatial adjacency (between regions at same time step)
        num_regions = len(region_embeddings)
        region_times = [emb.shape[1] for emb in region_embeddings.values()]
        max_time = max(region_times) if region_times else 0
        
        region_node_start = 0
        for i, (name1, emb1) in enumerate(region_embeddings.items()):
            T1 = emb1.shape[1]
            for j, (name2, emb2) in enumerate(region_embeddings.items()):
                if i != j:  # Different regions
                    T2 = emb2.shape[1]
                    T_min = min(T1, T2)
                    for t in range(T_min):
                        # Connect nodes at same time step
                        node1 = region_node_start + i * max_time + t
                        node2 = region_node_start + j * max_time + t
                        edge_index.append([node1, node2])
                        edge_attr.append(0)  # Edge type: spatial adjacency
        
        # 2. Frequency similarity (between same region across time)
        # (Simplified: connect temporally adjacent nodes)
        for name, emb in region_embeddings.items():
            T = emb.shape[1]
            for t in range(T - 1):
                node1 = region_node_start + list(region_embeddings.keys()).index(name) * max_time + t
                node2 = region_node_start + list(region_embeddings.keys()).index(name) * max_time + t + 1
                edge_index.append([node1, node2])
                edge_attr.append(1)  # Edge type: frequency similarity
        
        # 3. AV timing (phoneme↔viseme)
        audio_node_start = sum(region_times) * B
        if phoneme_times:
            # Connect audio nodes to mouth region nodes based on timing
            mouth_idx = list(region_embeddings.keys()).index('mouth') if 'mouth' in region_embeddings else None
            if mouth_idx is not None:
                T_mouth = region_embeddings['mouth'].shape[1]
                for t_a in range(T_a):
                    # Find corresponding mouth time (simplified)
                    t_mouth = int(t_a * T_mouth / T_a)
                    if t_mouth < T_mouth:
                        audio_node = audio_node_start + t_a
                        mouth_node = region_node_start + mouth_idx * max_time + t_mouth
                        edge_index.append([audio_node, mouth_node])
                        edge_index.append([mouth_node, audio_node])  # Bidirectional
                        edge_attr.append(2)  # Edge type: AV timing
                        edge_attr.append(2)
        
        # 4. Blink/prosody links (eye nodes ↔ audio prosody)
        # Connect eye regions to audio based on prosody
        for eye_name in ['left_eye', 'right_eye']:
            if eye_name in region_embeddings:
                eye_idx = list(region_embeddings.keys()).index(eye_name)
                T_eye = region_embeddings[eye_name].shape[1]
                for t_eye in range(T_eye):
                    t_audio = int(t_eye * T_a / T_eye)
                    if t_audio < T_a:
                        eye_node = region_node_start + eye_idx * max_time + t_eye
                        audio_node = audio_node_start + t_audio
                        edge_index.append([eye_node, audio_node])
                        edge_attr.append(3)  # Edge type: blink/prosody
        
        # Convert to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def forward(self,
                region_embeddings: Dict[str, torch.Tensor],
                audio_embeddings: torch.Tensor,
                phoneme_times: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass through graph.
        
        Args:
            region_embeddings: Dict of (B, T, embed_dim) region sequences
            audio_embeddings: (B, T_a, audio_dim) audio sequence
            phoneme_times: Optional phoneme timing
            
        Returns:
            (B, hidden_dim) graph representation
        """
        # Build graph
        data = self.build_graph(region_embeddings, audio_embeddings, phoneme_times)
        
        x = data.x
        edge_index = data.edge_index
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = self.dropout(x)
        
        x = self.norm(x)
        
        # Global pooling (mean)
        # Reshape back to batch format
        # This is simplified - in practice, need to track node indices per batch
        x_pooled = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        
        return x_pooled


class MaskedRelationLearner(nn.Module):
    """
    Masked-relation learner (MRL++) for self-supervised pretraining.
    Reconstructs masked edges/vertices + cross-clip contrastive tasks.
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 mask_rate: float = 0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_rate = mask_rate
        
        # Reconstruction heads
        self.vertex_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.edge_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Contrastive projection
        self.contrastive_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def mask_nodes(self, x: torch.Tensor, mask_prob: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask nodes.
        
        Args:
            x: (N, hidden_dim) node features
            mask_prob: Masking probability
            
        Returns:
            Masked features and mask tensor
        """
        mask_prob = mask_prob or self.mask_rate
        num_nodes = x.shape[0]
        num_mask = int(num_nodes * mask_prob)
        
        # Random mask indices
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask[mask_indices] = True
        
        # Mask features (replace with learnable mask token)
        masked_x = x.clone()
        mask_token = nn.Parameter(torch.randn(1, self.hidden_dim)).to(x.device)
        masked_x[mask] = mask_token.expand(num_mask, -1)
        
        return masked_x, mask
    
    def mask_edges(self, edge_index: torch.Tensor, num_edges: int, mask_prob: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask edges.
        
        Args:
            edge_index: (2, E) edge indices
            num_edges: Total number of edges
            mask_prob: Masking probability
            
        Returns:
            Masked edge indices and mask tensor
        """
        mask_prob = mask_prob or self.mask_rate
        num_mask = int(num_edges * mask_prob)
        
        mask_indices = torch.randperm(num_edges)[:num_mask]
        mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
        mask[mask_indices] = True
        
        # Remove masked edges
        keep_mask = ~mask
        masked_edge_index = edge_index[:, keep_mask]
        
        return masked_edge_index, mask
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                mode: str = 'vertex') -> Dict[str, torch.Tensor]:
        """
        Forward pass for masked reconstruction.
        
        Args:
            x: (N, hidden_dim) node features
            edge_index: (2, E) edge indices
            mode: 'vertex' or 'edge'
            
        Returns:
            Dictionary with reconstruction outputs
        """
        if mode == 'vertex':
            # Mask nodes
            masked_x, mask = self.mask_nodes(x)
            
            # Reconstruct
            reconstructed = self.vertex_reconstructor(masked_x)
            
            # Compute loss (MSE on masked nodes)
            target = x[mask]
            pred = reconstructed[mask]
            loss = F.mse_loss(pred, target)
            
            return {
                'reconstructed': reconstructed,
                'loss': loss,
                'mask': mask
            }
        
        elif mode == 'edge':
            # Mask edges
            num_edges = edge_index.shape[1]
            masked_edge_index, mask = self.mask_edges(edge_index, num_edges)
            
            # Get node features for edges
            src_nodes = x[edge_index[0]]
            dst_nodes = x[edge_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
            
            # Reconstruct edge existence
            edge_pred = self.edge_reconstructor(edge_features).squeeze(-1)
            edge_target = torch.ones(num_edges, device=x.device)
            edge_target[mask] = 0  # Masked edges should be 0
            
            loss = F.binary_cross_entropy_with_logits(edge_pred, edge_target)
            
            return {
                'edge_pred': edge_pred,
                'loss': loss,
                'mask': mask
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def contrastive_loss(self,
                        x1: torch.Tensor,
                        x2: torch.Tensor,
                        temperature: float = 0.1) -> torch.Tensor:
        """
        Contrastive loss between two graph representations.
        
        Args:
            x1: (B, hidden_dim) first graph
            x2: (B, hidden_dim) second graph
            temperature: Temperature for contrastive loss
            
        Returns:
            Contrastive loss
        """
        # Project
        z1 = F.normalize(self.contrastive_proj(x1), dim=-1)
        z2 = F.normalize(self.contrastive_proj(x2), dim=-1)
        
        # Compute similarity
        sim = torch.matmul(z1, z2.t()) / temperature
        
        # Positive pairs are diagonal
        labels = torch.arange(z1.shape[0], device=z1.device)
        
        loss = F.cross_entropy(sim, labels)
        
        return loss

