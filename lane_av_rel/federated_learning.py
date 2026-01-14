"""
Federated Landmark Distillation (FLD) framework.
Share tiny per-region embeddings + robust aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import copy


class LandmarkDistillationClient:
    """
    Client for federated landmark distillation.
    Shares per-region embedding prototypes instead of full gradients.
    """
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 region_names: List[str],
                 prototype_dim: int = 128,
                 device: str = 'cpu'):
        self.client_id = client_id
        self.model = model
        self.region_names = region_names
        self.prototype_dim = prototype_dim
        self.device = device
        self.model.to(device)
        
        # Per-region prototypes (embeddings)
        self.region_prototypes = {
            name: torch.zeros(prototype_dim, device=device)
            for name in region_names
        }
        
        # Attention statistics (for aggregation)
        self.attention_stats = {
            name: {'mean': 0.0, 'std': 0.0, 'count': 0}
            for name in region_names
        }
    
    def train_local(self,
                   dataloader,
                   optimizer,
                   criterion,
                   num_epochs: int = 1) -> Dict[str, float]:
        """
        Train locally on client data.
        
        Returns:
            Training statistics
        """
        self.model.train()
        losses = []
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                video = batch['video'].to(self.device)
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                output = self.model(video, audio)
                logits = output['logits']
                
                # Loss
                loss = criterion(logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        return {'loss': np.mean(losses)}
    
    def extract_prototypes(self) -> Dict[str, torch.Tensor]:
        """
        Extract per-region embedding prototypes.
        Returns lightweight prototypes instead of full gradients.
        """
        self.model.eval()
        prototypes = {}
        
        with torch.no_grad():
            # Extract region embeddings from model
            # This is simplified - in practice, use hooks to extract embeddings
            for name in self.region_names:
                # Get region encoder output dimension
                if hasattr(self.model, 'region_encoder'):
                    encoder = self.model.region_encoder.encoders.get(name)
                    if encoder is not None:
                        # Create dummy input
                        dummy_input = torch.zeros(1, 3, 64, 64, device=self.device)
                        embedding = encoder(dummy_input)
                        
                        # Project to prototype dimension
                        if embedding.shape[-1] != self.prototype_dim:
                            proj = nn.Linear(embedding.shape[-1], self.prototype_dim).to(self.device)
                            prototype = proj(embedding)
                        else:
                            prototype = embedding
                        
                        prototypes[name] = prototype.squeeze(0).cpu()
        
        return prototypes
    
    def extract_attention_stats(self) -> Dict[str, Dict]:
        """
        Extract attention statistics (no raw data).
        """
        return self.attention_stats
    
    def update_from_prototypes(self, global_prototypes: Dict[str, torch.Tensor]):
        """
        Update local model from global prototypes.
        """
        # In practice, this would update the model weights based on prototypes
        # This is a simplified version
        for name, prototype in global_prototypes.items():
            if name in self.region_prototypes:
                self.region_prototypes[name] = prototype.to(self.device)


class RobustAggregator:
    """
    Robust aggregation for federated learning.
    Implements coordinate-wise median, Krum, and reputation-based weighting.
    """
    
    def __init__(self, method: str = 'median', f: int = 0):
        """
        Args:
            method: Aggregation method ('median', 'krum', 'reputation')
            f: Number of Byzantine clients to tolerate (for Krum)
        """
        self.method = method
        self.f = f
        self.client_reputations = defaultdict(float)
    
    def aggregate(self,
                 client_prototypes: List[Dict[str, torch.Tensor]],
                 client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client prototypes.
        
        Args:
            client_prototypes: List of prototype dictionaries from clients
            client_ids: List of client IDs
            
        Returns:
            Aggregated global prototypes
        """
        if not client_prototypes:
            return {}
        
        # Get all region names
        region_names = set()
        for prototypes in client_prototypes:
            region_names.update(prototypes.keys())
        
        aggregated = {}
        
        for region_name in region_names:
            # Collect prototypes for this region
            region_prototypes = []
            valid_clients = []
            
            for i, prototypes in enumerate(client_prototypes):
                if region_name in prototypes:
                    region_prototypes.append(prototypes[region_name])
                    valid_clients.append(i)
            
            if not region_prototypes:
                continue
            
            # Stack: (num_clients, prototype_dim)
            stacked = torch.stack(region_prototypes)
            
            # Aggregate
            if self.method == 'median':
                aggregated[region_name] = torch.median(stacked, dim=0)[0]
            
            elif self.method == 'krum':
                aggregated[region_name] = self._krum(stacked, valid_clients)
            
            elif self.method == 'reputation':
                weights = torch.tensor([
                    self.client_reputations.get(client_ids[i], 1.0)
                    for i in valid_clients
                ], device=stacked.device)
                weights = weights / weights.sum()
                aggregated[region_name] = (stacked * weights.unsqueeze(1)).sum(dim=0)
            
            else:
                # Default: mean
                aggregated[region_name] = stacked.mean(dim=0)
        
        return aggregated
    
    def _krum(self, stacked: torch.Tensor, client_indices: List[int]) -> torch.Tensor:
        """
        Krum aggregation: select client with minimum sum of distances to n-f-2 nearest neighbors.
        """
        num_clients = stacked.shape[0]
        n = num_clients - self.f - 2  # Number of nearest neighbors
        
        if n <= 0:
            return stacked.mean(dim=0)
        
        # Compute pairwise distances
        distances = torch.cdist(stacked, stacked)  # (num_clients, num_clients)
        
        # For each client, sum distances to n nearest neighbors
        krum_scores = []
        for i in range(num_clients):
            # Get n nearest neighbors (excluding self)
            dists = distances[i]
            dists[i] = float('inf')  # Exclude self
            nearest = torch.topk(dists, n, largest=False)[0]
            krum_scores.append(nearest.sum().item())
        
        # Select client with minimum score
        best_idx = np.argmin(krum_scores)
        
        return stacked[best_idx]
    
    def update_reputation(self, client_id: int, performance: float):
        """
        Update client reputation based on performance.
        """
        # Exponential moving average
        alpha = 0.1
        self.client_reputations[client_id] = (
            alpha * performance + (1 - alpha) * self.client_reputations.get(client_id, 1.0)
        )


class FederatedLandmarkDistillation:
    """
    Main federated learning coordinator.
    """
    
    def __init__(self,
                 global_model: nn.Module,
                 region_names: List[str],
                 num_clients: int = 10,
                 aggregation_method: str = 'median',
                 prototype_dim: int = 128,
                 device: str = 'cpu'):
        self.global_model = global_model
        self.region_names = region_names
        self.num_clients = num_clients
        self.device = device
        
        # Create clients
        self.clients = []
        for i in range(num_clients):
            client_model = copy.deepcopy(global_model)
            client = LandmarkDistillationClient(
                client_id=i,
                model=client_model,
                region_names=region_names,
                prototype_dim=prototype_dim,
                device=device
            )
            self.clients.append(client)
        
        # Aggregator
        self.aggregator = RobustAggregator(method=aggregation_method)
        
        # Global prototypes
        self.global_prototypes = {
            name: torch.zeros(prototype_dim)
            for name in region_names
        }
    
    def train_round(self,
                   client_dataloaders: List,
                   num_local_epochs: int = 1,
                   lr: float = 0.001) -> Dict[str, float]:
        """
        Perform one federated learning round.
        
        Args:
            client_dataloaders: List of dataloaders for each client
            num_local_epochs: Number of local training epochs
            lr: Learning rate
            
        Returns:
            Round statistics
        """
        # Train clients locally
        client_prototypes = []
        client_ids = []
        client_stats = []
        
        for i, (client, dataloader) in enumerate(zip(self.clients, client_dataloaders)):
            # Local training
            optimizer = torch.optim.Adam(client.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            stats = client.train_local(dataloader, optimizer, criterion, num_local_epochs)
            client_stats.append(stats)
            
            # Extract prototypes
            prototypes = client.extract_prototypes()
            client_prototypes.append(prototypes)
            client_ids.append(client.client_id)
        
        # Aggregate prototypes
        aggregated_prototypes = self.aggregator.aggregate(client_prototypes, client_ids)
        
        # Update global model (simplified - in practice, update based on prototypes)
        self.global_prototypes = aggregated_prototypes
        
        # Distribute to clients
        for client in self.clients:
            client.update_from_prototypes(aggregated_prototypes)
        
        # Compute round statistics
        avg_loss = np.mean([s['loss'] for s in client_stats])
        
        return {
            'avg_loss': avg_loss,
            'num_clients': len(client_prototypes)
        }
    
    def add_dp_noise(self, prototypes: Dict[str, torch.Tensor], sigma: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to prototypes.
        """
        noisy_prototypes = {}
        for name, prototype in prototypes.items():
            noise = torch.randn_like(prototype) * sigma
            noisy_prototypes[name] = prototype + noise
        
        return noisy_prototypes
    
    def compute_bandwidth(self, prototypes: Dict[str, torch.Tensor]) -> float:
        """
        Compute bandwidth required to transmit prototypes (in KB).
        """
        total_params = sum(p.numel() for p in prototypes.values())
        total_bytes = total_params * 4  # float32 = 4 bytes
        total_kb = total_bytes / 1024
        return total_kb

