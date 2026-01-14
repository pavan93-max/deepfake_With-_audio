"""
Explainability module: Grad-CAM, heatmaps, forensic reports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GradCAM:
    """
    Grad-CAM for region encoders.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self,
                    input_tensor: torch.Tensor,
                    target_class: int = 1) -> np.ndarray:
        """
        Generate class activation map.
        
        Args:
            input_tensor: (1, C, H, W) input
            target_class: Target class index
            
        Returns:
            (H, W) CAM heatmap
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        
        # Normalize
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-10)
        
        return cam.detach().cpu().numpy()


class RegionHeatmapGenerator:
    """
    Generate per-region heatmaps for explainability.
    """
    
    def __init__(self, roi_size: int = 64):
        self.roi_size = roi_size
    
    def generate_heatmap(self,
                        roi: np.ndarray,
                        attention_weights: np.ndarray,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate heatmap overlay for ROI.
        
        Args:
            roi: (H, W, 3) ROI image
            attention_weights: (H, W) attention weights
            colormap: OpenCV colormap
            
        Returns:
            (H, W, 3) heatmap overlay
        """
        # Resize attention to match ROI
        if attention_weights.shape != roi.shape[:2]:
            attention_weights = cv2.resize(
                attention_weights,
                (roi.shape[1], roi.shape[0])
            )
        
        # Normalize
        attention_weights = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-10
        )
        
        # Apply colormap
        attention_uint8 = (attention_weights * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attention_uint8, colormap)
        
        # Overlay on original image
        overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)
        
        return overlay


class GraphAttentionVisualizer:
    """
    Visualize graph attention weights.
    """
    
    def __init__(self):
        pass
    
    def visualize_edges(self,
                       edge_index: torch.Tensor,
                       edge_weights: torch.Tensor,
                       node_positions: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Visualize graph edges with attention weights.
        
        Args:
            edge_index: (2, E) edge indices
            edge_weights: (E,) edge weights
            node_positions: Dict mapping node names to (x, y) positions
            
        Returns:
            Visualization image
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Normalize edge weights
        if edge_weights.max() > edge_weights.min():
            edge_weights_norm = (edge_weights - edge_weights.min()) / (
                edge_weights.max() - edge_weights.min()
            )
        else:
            edge_weights_norm = edge_weights
        
        # Draw edges
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            weight = edge_weights_norm[i].item()
            
            # Get positions (simplified)
            src_pos = node_positions.get(f'node_{src}', (0, 0))
            dst_pos = node_positions.get(f'node_{dst}', (1, 1))
            
            # Draw edge with color based on weight
            color = plt.cm.viridis(weight)
            ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
                   color=color, alpha=0.6, linewidth=2 * weight)
        
        # Draw nodes
        for name, pos in node_positions.items():
            ax.scatter(pos[0], pos[1], s=100, alpha=0.8)
            ax.text(pos[0], pos[1], name, fontsize=8)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Convert to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
        
        return img


class AVDesyncMarker:
    """
    Mark audio-visual desynchronization points.
    """
    
    def __init__(self):
        pass
    
    def mark_desync(self,
                   video_timestamps: np.ndarray,
                   audio_timestamps: np.ndarray,
                   phoneme_times: Dict,
                   mismatch_scores: np.ndarray) -> Dict:
        """
        Mark AV mismatch points.
        
        Args:
            video_timestamps: (T_v,) video timestamps
            audio_timestamps: (T_a,) audio timestamps
            phoneme_times: Phoneme timing information
            mismatch_scores: (T_v, T_a) mismatch scores
            
        Returns:
            Dictionary with desync markers
        """
        # Find high mismatch regions
        threshold = np.percentile(mismatch_scores, 90)
        high_mismatch = mismatch_scores > threshold
        
        markers = {
            'desync_intervals': [],
            'phoneme_mismatches': []
        }
        
        # Find continuous intervals
        for t_v in range(len(video_timestamps)):
            for t_a in range(len(audio_timestamps)):
                if high_mismatch[t_v, t_a]:
                    markers['desync_intervals'].append({
                        'video_time': video_timestamps[t_v],
                        'audio_time': audio_timestamps[t_a],
                        'mismatch_score': float(mismatch_scores[t_v, t_a])
                    })
        
        # Phoneme-specific mismatches
        if phoneme_times:
            for phoneme, start, end in zip(
                phoneme_times['phonemes'],
                phoneme_times['start_times'],
                phoneme_times['end_times']
            ):
                # Check if there's high mismatch in this interval
                # (simplified)
                markers['phoneme_mismatches'].append({
                    'phoneme': phoneme,
                    'start': float(start),
                    'end': float(end),
                    'mismatch': True  # Simplified
                })
        
        return markers


class ForensicReportGenerator:
    """
    Generate forensic-grade reports with signed provenance.
    """
    
    def __init__(self):
        pass
    
    def generate_report(self,
                       video_path: str,
                       prediction: Dict,
                       heatmaps: Dict[str, np.ndarray],
                       attention_maps: Dict[str, np.ndarray],
                       av_markers: Optional[Dict] = None,
                       calibration_info: Optional[Dict] = None,
                       model_info: Optional[Dict] = None) -> Dict:
        """
        Generate forensic report.
        
        Args:
            video_path: Path to input video
            prediction: Prediction dictionary
            heatmaps: Dictionary of region heatmaps
            attention_maps: Dictionary of attention maps
            av_markers: Optional AV desync markers
            calibration_info: Optional calibration information
            model_info: Optional model information
            
        Returns:
            Forensic report dictionary
        """
        # Compute checksums
        video_hash = self._compute_hash(video_path)
        
        # Build report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'video_hash': video_hash,
                'model_version': model_info.get('version', '1.0.0') if model_info else '1.0.0'
            },
            'prediction': {
                'class': int(prediction['class']),
                'confidence': float(prediction['confidence']),
                'probabilities': {
                    'real': float(prediction['probs'][0]),
                    'fake': float(prediction['probs'][1])
                }
            },
            'attributions': {
                'regions': {},
                'modality': prediction.get('modality_attribution', None)
            },
            'calibration': calibration_info or {},
            'av_analysis': av_markers or {},
            'attention_maps': {
                # Store paths or base64 encoded images
                name: self._encode_image(img) for name, img in attention_maps.items()
            },
            'heatmaps': {
                name: self._encode_image(img) for name, img in heatmaps.items()
            }
        }
        
        # Add region attributions
        if 'region_scores' in prediction:
            for region, score in prediction['region_scores'].items():
                report['attributions']['regions'][region] = {
                    'anomaly_score': float(score),
                    'contribution': float(score / sum(prediction['region_scores'].values()))
                }
        
        # Compute report hash
        report_json = json.dumps(report, sort_keys=True)
        report_hash = hashlib.sha256(report_json.encode()).hexdigest()
        report['report_hash'] = report_hash
        
        # Sign report (simplified - in practice use cryptographic signing)
        report['signature'] = self._sign_report(report_hash)
        
        return report
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _encode_image(self, img: np.ndarray) -> str:
        """Encode image to base64."""
        import base64
        from io import BytesIO
        import PIL.Image
        
        img_pil = PIL.Image.fromarray(img)
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def _sign_report(self, report_hash: str) -> str:
        """Sign report hash (simplified)."""
        # In practice, use RSA/ECDSA signing
        return hashlib.sha256(f"signature_key_{report_hash}".encode()).hexdigest()
    
    def save_report(self, report: Dict, output_path: str):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def export_visual_report(self,
                            report: Dict,
                            output_path: str,
                            video_path: Optional[str] = None):
        """
        Export visual report with overlays.
        
        Args:
            report: Forensic report dictionary
            output_path: Output image path
            video_path: Optional video path for frame extraction
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prediction summary
        ax = axes[0, 0]
        probs = report['prediction']['probabilities']
        ax.bar(['Real', 'Fake'], [probs['real'], probs['fake']])
        ax.set_title(f"Prediction: {report['prediction']['class']} "
                    f"(Confidence: {report['prediction']['confidence']:.2f})")
        ax.set_ylim(0, 1)
        
        # Region attributions
        ax = axes[0, 1]
        if report['attributions']['regions']:
            regions = list(report['attributions']['regions'].keys())
            scores = [report['attributions']['regions'][r]['anomaly_score'] for r in regions]
            ax.barh(regions, scores)
            ax.set_title('Region Anomaly Scores')
        
        # Calibration plot
        ax = axes[1, 0]
        if 'calibration_curve' in report['calibration']:
            # Plot calibration curve
            pass
        ax.set_title('Calibration')
        
        # AV analysis
        ax = axes[1, 1]
        if report['av_analysis']:
            num_desync = len(report['av_analysis'].get('desync_intervals', []))
            ax.text(0.5, 0.5, f'AV Desync Points: {num_desync}',
                   ha='center', va='center', fontsize=12)
        ax.set_title('AV Synchronization')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

