"""
Evaluation script with forensic reporting.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_curve
import tqdm

from lane_av_rel.model import LANeAVRel
from lane_av_rel.datasets import DeepfakeTIMITDataset
from lane_av_rel.explainability import (
    GradCAM, RegionHeatmapGenerator, ForensicReportGenerator
)


def evaluate(model, dataloader, device, output_dir=None):
    """Evaluate model and generate forensic reports."""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_predictions = []
    
    report_generator = ForensicReportGenerator()
    heatmap_generator = RegionHeatmapGenerator()
    
    reports = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
            video = batch['video'].to(device)
            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(device)
            labels = batch['label'].to(device)
            video_paths = batch.get('video_path', [])
            
            output = model(video, audio)
            logits = output['logits']
            probs = output['probs']
            
            predictions = logits.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            # Generate forensic reports for samples
            if output_dir:
                for i in range(video.shape[0]):
                    video_path = video_paths[i] if isinstance(video_paths, list) else str(video_paths)
                    
                    # Generate report
                    prediction_dict = {
                        'class': int(predictions[i].item()),
                        'confidence': float(probs[i].max().item()),
                        'probs': probs[i].cpu().numpy().tolist()
                    }
                    
                    # Generate heatmaps (simplified)
                    heatmaps = {}
                    if 'rois' in output:
                        for region_name, roi_seq in output['rois'].items():
                            roi = roi_seq[i, 0].cpu().numpy().transpose(1, 2, 0)
                            # Dummy attention weights
                            attention = np.ones((roi.shape[0], roi.shape[1])) * 0.5
                            heatmap = heatmap_generator.generate_heatmap(roi, attention)
                            heatmaps[region_name] = heatmap
                    
                    report = report_generator.generate_report(
                        video_path=video_path,
                        prediction=prediction_dict,
                        heatmaps=heatmaps,
                        attention_maps={},
                        model_info={'version': '1.0.0'}
                    )
                    
                    reports.append(report)
    
    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Compute metrics
    fake_probs = all_probs[:, 0]  # Probability of being fake
    real_probs = all_probs[:, 1]  # Probability of being real
    
    # AUC
    try:
        auc = roc_auc_score(all_labels, fake_probs)
    except:
        auc = 0.0
    
    # Accuracy
    acc = accuracy_score(all_labels, all_predictions)
    
    # EER (Equal Error Rate)
    fpr, tpr, thresholds = roc_curve(all_labels, fake_probs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    # Precision, Recall, F1
    precision, recall, _ = precision_recall_curve(all_labels, fake_probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1 = f1[~np.isnan(f1)].max() if len(f1[~np.isnan(f1)]) > 0 else 0.0
    
    metrics = {
        'auc': float(auc),
        'accuracy': float(acc),
        'eer': float(eer),
        'f1': float(f1),
        'num_samples': len(all_labels)
    }
    
    # Save reports
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save reports
        with open(output_dir / 'reports.json', 'w') as f:
            json.dump(reports, f, indent=2)
        
        # Save visual reports
        for i, report in enumerate(reports[:10]):  # Save first 10
            report_generator.export_visual_report(
                report,
                output_dir / f'report_{i}.png'
            )
    
    return metrics, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dataset
    dataset = DeepfakeTIMITDataset(
        root_dir=args.data_dir,
        quality='higher_quality',
        split=args.split,
        video_length=config.get('video_length', 32)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    metrics, reports = evaluate(model, dataloader, device, args.output_dir)
    
    print("\nEvaluation Results:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

