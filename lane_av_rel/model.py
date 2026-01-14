"""
Main LANe-AV-Rel model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .face_parsing import FaceParserTorch
from .region_encoders import MultiRegionEncoder
from .temporal_modeling import MultiRegionTemporalEncoder
from .audio_processing import AudioEncoder
from .relation_graph import MultiModalRelationGraph, MaskedRelationLearner


class LANeAVRel(nn.Module):
    """
    LANe-AV-Rel: Landmark-Aware Audio-Visual Relational Deepfake Detector
    """
    
    def __init__(self,
                 roi_size: int = 64,
                 num_regions: int = 7,
                 region_embed_dim: int = 128,
                 audio_embed_dim: int = 128,
                 graph_hidden_dim: int = 256,
                 num_classes: int = 2,
                 use_artifacts: bool = True,
                 use_temporal: bool = True,
                 use_audio: bool = True,
                 use_graph: bool = True):
        super().__init__()
        
        self.roi_size = roi_size
        self.use_artifacts = use_artifacts
        self.use_temporal = use_temporal
        self.use_audio = use_audio
        self.use_graph = use_graph
        
        # Face parser
        try:
            self.face_parser = FaceParserTorch(roi_size=roi_size)
        except ImportError as e:
            print(f"Warning: Could not initialize face parser: {e}")
            print("Face parsing is required for this model. Please install MediaPipe:")
            print("  pip install mediapipe")
            print("  Or on Windows: pip install --upgrade --force-reinstall mediapipe")
            raise
        
        # Region encoders
        self.region_encoder = MultiRegionEncoder(
            roi_size=roi_size,
            in_channels=3,
            cnn_channels=32,
            vit_embed_dim=region_embed_dim,
            use_artifacts=use_artifacts
        )
        
        # Temporal encoders
        if use_temporal:
            self.temporal_encoder = MultiRegionTemporalEncoder(
                region_dims=self.region_encoder.region_dims,
                use_attention=True,
                use_lstm=False
            )
        
        # Audio encoder
        if use_audio:
            self.audio_encoder = AudioEncoder(
                sample_rate=16000,
                embed_dim=audio_embed_dim
            )
        
        # Relation graph
        if use_graph:
            try:
                self.relation_graph = MultiModalRelationGraph(
                    region_dims=self.region_encoder.region_dims,
                    audio_dim=audio_embed_dim,
                    hidden_dim=graph_hidden_dim
                )
            except ImportError as e:
                print(f"Warning: Could not initialize relation graph: {e}")
                print("Falling back to non-graph mode. Set use_graph=False to suppress this warning.")
                use_graph = False
                self.use_graph = False
        
        if not use_graph:
            self.relation_graph = None
        
        # Classification head
        if use_graph:
            classifier_input_dim = graph_hidden_dim
        elif use_audio:
            classifier_input_dim = sum(self.region_encoder.region_dims.values()) + audio_embed_dim
        else:
            classifier_input_dim = sum(self.region_encoder.region_dims.values())
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.LayerNorm(classifier_input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_input_dim // 2, classifier_input_dim // 4),
            nn.LayerNorm(classifier_input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_input_dim // 4, num_classes)
        )
        
        # Modality attribution head (which modality contributes most)
        if use_audio:
            self.modality_attribution = nn.Sequential(
                nn.Linear(classifier_input_dim, 3),  # visual, audio, both
                nn.Softmax(dim=-1)
            )
    
    def forward(self,
                video: torch.Tensor,
                audio: Optional[torch.Tensor] = None,
                phoneme_times: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video: (B, T, C, H, W) video frames
            audio: (B, T_a) or (B, T_a, 1) audio waveform
            phoneme_times: Optional phoneme timing information
            
        Returns:
            Dictionary with predictions and intermediate features
        """
        B, T, C, H, W = video.shape
        
        # Parse faces and extract ROIs
        # Reshape for batch processing
        video_flat = video.view(B * T, C, H, W)
        rois = self.face_parser(video_flat)  # Dict of (B*T, C, roi_size, roi_size)
        
        # Reshape back to (B, T, ...)
        rois_reshaped = {
            name: roi.view(B, T, *roi.shape[1:])
            for name, roi in rois.items()
        }
        
        # Encode regions (per frame)
        region_embeddings = {}
        for name, roi_seq in rois_reshaped.items():
            # Process each frame
            B_t, T_t = roi_seq.shape[:2]
            roi_flat = roi_seq.view(B_t * T_t, *roi_seq.shape[2:])
            emb_flat = self.region_encoder.encoders[name](roi_flat)
            emb = emb_flat.view(B_t, T_t, -1)
            region_embeddings[name] = emb
        
        # Temporal encoding
        if self.use_temporal:
            region_embeddings = self.temporal_encoder(region_embeddings)
        
        # Audio encoding
        audio_embeddings = None
        if self.use_audio and audio is not None:
            audio_output = self.audio_encoder(audio)
            audio_embeddings = audio_output['encoded']  # (B, T_a, audio_embed_dim)
        
        # Relation graph
        if self.use_graph and self.relation_graph is not None:
            if audio_embeddings is not None:
                graph_repr = self.relation_graph(
                    region_embeddings,
                    audio_embeddings,
                    phoneme_times
                )
            else:
                # Fallback: use dummy audio
                dummy_audio = torch.zeros(
                    B, T, self.audio_encoder.embed_dim,
                    device=video.device
                )
                graph_repr = self.relation_graph(
                    region_embeddings,
                    dummy_audio,
                    phoneme_times
                )
            
            # Global pooling (mean over time if needed)
            if graph_repr.dim() > 2:
                graph_repr = graph_repr.mean(dim=1)  # (B, hidden_dim)
            
            features = graph_repr
        else:
            # Concatenate region embeddings
            region_concat = torch.cat(list(region_embeddings.values()), dim=-1)  # (B, T, sum_dims)
            region_pooled = region_concat.mean(dim=1)  # (B, sum_dims)
            
            if audio_embeddings is not None:
                audio_pooled = audio_embeddings.mean(dim=1)  # (B, audio_embed_dim)
                features = torch.cat([region_pooled, audio_pooled], dim=-1)
            else:
                features = region_pooled
        
        # Classification
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        # Modality attribution
        modality_attribution = None
        if self.use_audio and hasattr(self, 'modality_attribution'):
            modality_attribution = self.modality_attribution(features)
        
        return {
            'logits': logits,
            'probs': probs,
            'features': features,
            'region_embeddings': region_embeddings,
            'audio_embeddings': audio_embeddings,
            'modality_attribution': modality_attribution,
            'rois': rois_reshaped
        }
    
    def get_attention_maps(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for explainability.
        Returns attention weights from graph and temporal modules.
        """
        # This would be implemented with hooks to extract attention
        # For now, return dummy
        return {}


class LANeAVRelWithMRL(LANeAVRel):
    """
    LANe-AV-Rel with Masked Relation Learning for pretraining.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add MRL module
        if self.use_graph:
            self.mrl = MaskedRelationLearner(
                hidden_dim=kwargs.get('graph_hidden_dim', 256)
            )
    
    def pretrain_step(self,
                     video: torch.Tensor,
                     audio: Optional[torch.Tensor] = None,
                     mode: str = 'vertex') -> Dict[str, torch.Tensor]:
        """
        Self-supervised pretraining step.
        
        Args:
            video: (B, T, C, H, W) video frames
            audio: Optional audio
            mode: 'vertex' or 'edge' for masking
            
        Returns:
            Dictionary with pretraining loss
        """
        # Get region and audio embeddings (without classification)
        B, T, C, H, W = video.shape
        video_flat = video.view(B * T, C, H, W)
        rois = self.face_parser(video_flat)
        
        rois_reshaped = {
            name: roi.view(B, T, *roi.shape[1:])
            for name, roi in rois.items()
        }
        
        region_embeddings = {}
        for name, roi_seq in rois_reshaped.items():
            B_t, T_t = roi_seq.shape[:2]
            roi_flat = roi_seq.view(B_t * T_t, *roi_seq.shape[2:])
            emb_flat = self.region_encoder.encoders[name](roi_flat)
            emb = emb_flat.view(B_t, T_t, -1)
            region_embeddings[name] = emb
        
        if self.use_temporal:
            region_embeddings = self.temporal_encoder(region_embeddings)
        
        audio_embeddings = None
        if self.use_audio and audio is not None:
            audio_output = self.audio_encoder(audio)
            audio_embeddings = audio_output['encoded']
        
        # Build graph
        if audio_embeddings is None:
            dummy_audio = torch.zeros(B, T, self.audio_encoder.embed_dim, device=video.device)
            audio_embeddings = dummy_audio
        
        graph_data = self.relation_graph.build_graph(
            region_embeddings,
            audio_embeddings
        )
        
        # Apply MRL
        mrl_output = self.mrl(graph_data.x, graph_data.edge_index, mode=mode)
        
        return {
            'mrl_loss': mrl_output['loss'],
            'mrl_output': mrl_output
        }

