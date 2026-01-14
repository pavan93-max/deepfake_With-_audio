"""
Training script for LANe-AV-Rel.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import tqdm

from lane_av_rel.model import LANeAVRel
from lane_av_rel.losses import CombinedLoss
from lane_av_rel.datasets import DeepfakeTIMITDataset, AVDesyncAugmentation


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        video = batch['video'].to(device)
        audio = batch.get('audio', None)
        if audio is not None:
            audio = audio.to(device)
        labels = batch['label'].to(device)
        
        # Forward
        output = model(video, audio)
        logits = output['logits']
        probs = output['probs']
        
        # Extract features for loss
        visual_features = output.get('features', None)
        audio_features = output.get('audio_embeddings', None)
        if audio_features is not None:
            audio_features = audio_features.mean(dim=1)  # Pool over time
        
        # Compute loss
        losses = criterion(
            logits=logits,
            probs=probs,
            targets=labels,
            visual_features=visual_features,
            audio_features=audio_features
        )
        
        loss = losses['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': correct / total if total > 0 else 0
        })
    
    return total_loss / len(dataloader), correct / total if total > 0 else 0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Validation'):
            video = batch['video'].to(device)
            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(device)
            labels = batch['label'].to(device)
            
            output = model(video, audio)
            logits = output['logits']
            probs = output['probs']
            
            visual_features = output.get('features', None)
            audio_features = output.get('audio_embeddings', None)
            if audio_features is not None:
                audio_features = audio_features.mean(dim=1)
            
            losses = criterion(
                logits=logits,
                probs=probs,
                targets=labels,
                visual_features=visual_features,
                audio_features=audio_features
            )
            
            loss = losses['total']
            total_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lane_av_rel.yaml')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--cleaned_dir', type=str, default='./cleaned_dataset',
                       help='Path to cleaned/preprocessed dataset (if None, uses raw data)')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    
    # Model
    model = LANeAVRel(
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
    
    # Loss
    criterion = CombinedLoss(
        ce_weight=config.get('ce_weight', 1.0),
        av_weight=config.get('av_weight', 0.5),
        temporal_weight=config.get('temporal_weight', 0.1),
        calibration_weight=config.get('calibration_weight', 0.1),
        use_av=config.get('use_audio', True),
        use_temporal=config.get('use_temporal', True)
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 100)
    )
    
    # Datasets
    train_dataset = DeepfakeTIMITDataset(
        root_dir=args.data_dir,
        quality='higher_quality',
        split='train',
        video_length=config.get('video_length', 32),
        cleaned_dir=args.cleaned_dir if Path(args.cleaned_dir).exists() else None
    )
    
    val_dataset = DeepfakeTIMITDataset(
        root_dir=args.data_dir,
        quality='higher_quality',
        split='val',
        video_length=config.get('video_length', 32),
        cleaned_dir=args.cleaned_dir if Path(args.cleaned_dir).exists() else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Training loop
    num_epochs = config.get('num_epochs', 100)
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, output_dir / 'best.pth')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': config
        }, output_dir / 'latest.pth')
    
    writer.close()


if __name__ == '__main__':
    main()

