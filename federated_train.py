"""
Federated training script.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

from lane_av_rel.model import LANeAVRel
from lane_av_rel.federated_learning import FederatedLandmarkDistillation
from lane_av_rel.datasets import DeepfakeTIMITDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lane_av_rel.yaml')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='outputs_federated')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    config['federated']['num_clients'] = args.num_clients
    config['federated']['num_rounds'] = args.num_rounds
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Create global model
    global_model = LANeAVRel(
        roi_size=config.get('roi_size', 64),
        region_embed_dim=config.get('region_embed_dim', 128),
        audio_embed_dim=config.get('audio_embed_dim', 128),
        graph_hidden_dim=config.get('graph_hidden_dim', 256),
        num_classes=2,
        use_artifacts=config.get('use_artifacts', True),
        use_temporal=config.get('use_temporal', True),
        use_audio=config.get('use_audio', True),
        use_graph=config.get('use_graph', True)
    ).to(device)
    
    # Region names
    region_names = list(global_model.region_encoder.region_dims.keys())
    
    # Federated learning coordinator
    fl_coordinator = FederatedLandmarkDistillation(
        global_model=global_model,
        region_names=region_names,
        num_clients=config['federated']['num_clients'],
        aggregation_method=config['federated']['aggregation_method'],
        prototype_dim=config['federated']['prototype_dim'],
        device=device
    )
    
    # Load dataset
    full_dataset = DeepfakeTIMITDataset(
        root_dir=args.data_dir,
        quality='higher_quality',
        split='train',
        video_length=config.get('video_length', 32)
    )
    
    # Split dataset among clients (non-IID simulation)
    num_samples_per_client = len(full_dataset) // config['federated']['num_clients']
    client_datasets = []
    
    for i in range(config['federated']['num_clients']):
        start_idx = i * num_samples_per_client
        end_idx = start_idx + num_samples_per_client if i < config['federated']['num_clients'] - 1 else len(full_dataset)
        indices = list(range(start_idx, end_idx))
        client_dataset = Subset(full_dataset, indices)
        client_datasets.append(client_dataset)
    
    # Create dataloaders
    client_dataloaders = [
        DataLoader(
            dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=2
        )
        for dataset in client_datasets
    ]
    
    # Training rounds
    round_stats = []
    for round_num in tqdm(range(config['federated']['num_rounds']), desc='FL Rounds'):
        # Train round
        stats = fl_coordinator.train_round(
            client_dataloaders=client_dataloaders,
            num_local_epochs=config['federated']['local_epochs'],
            lr=config.get('lr', 0.001)
        )
        
        stats['round'] = round_num
        round_stats.append(stats)
        
        # Compute bandwidth
        bandwidth = fl_coordinator.compute_bandwidth(fl_coordinator.global_prototypes)
        stats['bandwidth_kb'] = bandwidth
        
        print(f"Round {round_num}: Loss={stats['avg_loss']:.4f}, "
              f"Bandwidth={bandwidth:.2f} KB")
        
        # Save checkpoint
        if (round_num + 1) % 10 == 0:
            torch.save({
                'round': round_num,
                'global_model_state_dict': global_model.state_dict(),
                'global_prototypes': fl_coordinator.global_prototypes,
                'stats': round_stats,
                'config': config
            }, output_dir / f'checkpoint_round_{round_num}.pth')
    
    # Save final model
    torch.save({
        'global_model_state_dict': global_model.state_dict(),
        'global_prototypes': fl_coordinator.global_prototypes,
        'stats': round_stats,
        'config': config
    }, output_dir / 'final_model.pth')
    
    print(f"\nFederated training complete. Results saved to {output_dir}")
    print(f"Total rounds: {config['federated']['num_rounds']}")
    print(f"Average bandwidth per round: {np.mean([s['bandwidth_kb'] for s in round_stats]):.2f} KB")


if __name__ == '__main__':
    main()

