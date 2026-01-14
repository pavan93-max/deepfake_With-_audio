# LANe-AV-Rel: Landmark-Aware Audio-Visual Relational Deepfake Detector

A forensic-grade deepfake detection system that learns frequency-temporal artifacts per face part, trains via lightweight federated distillation, and outputs time/region attributions.

## Key Innovations

1. **Landmark-first, frequency-aware parts**: Systematic DWT/Gabor/FFT channels per face region
2. **AV-anchored relation graph**: Phoneme↔viseme edges with self-supervised masked-relation pretraining
3. **Federated landmark distillation**: Share tiny part-wise prototypes with robust aggregation
4. **Forensic-grade outputs**: Region/time heatmaps + signed JSON evidence

## Architecture

- **Face Parsing**: MediaPipe landmarks → fixed ROIs (eyes, mouth, nose, cheeks, jawline)
- **Frequency Artifacts**: DWT (LL/LH/HL/HH), Gabor bank, FFT magnitude/phase per region
- **Per-Region Encoders**: CNN + tiny ViT with artifact channel fusion
- **Temporal Modeling**: Attention/LSTM per region with persistence loss
- **Audio Processing**: Log-Mel + CQT, phoneme alignment, prosody extraction
- **Multi-Modal Graph**: Spatial + frequency + AV timing edges with masked-relation learning
- **Federated Learning**: Landmark distillation with DP and robust aggregation

## Installation

### Python Version Requirements

**⚠️ IMPORTANT: Python 3.12 is NOT fully supported yet!**

**Recommended: Python 3.10 or 3.11**

The project requires:
- **Python 3.8 - 3.11** (for full compatibility)
- **Python 3.10 or 3.11** (strongly recommended - best compatibility)
- **Python 3.12** (NOT recommended - many packages have compatibility issues)

**Why Python 3.12 has issues:**
- MediaPipe: Build failures with Python 3.12
- pyannote.audio: May have compatibility issues
- Many dependencies haven't been updated for Python 3.12 yet

To check your Python version:
```bash
python --version
```

**If you have Python 3.12:**
1. **Option 1 (Recommended):** Use Python 3.10 or 3.11 instead
   - Install Python 3.10/3.11 alongside 3.12
   - Use virtual environment: `python3.10 -m venv venv`
   
2. **Option 2:** Install core dependencies only (some features disabled)
   ```bash
   pip install -r requirements.txt
   # Skip optional packages that don't support Python 3.12
   ```

**Recommended Python versions:**
- **Python 3.10** (best balance of compatibility and features) ⭐
- **Python 3.11** (latest stable, good performance) ⭐

### Install Dependencies

**Step 1: Install core dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Install optional dependencies (if needed)**
```bash
# For graph neural networks (relation graph feature)
pip install dgl torch-geometric

# For advanced audio processing (Python 3.10-3.11 only)
pip install pyannote.audio

# For phoneme alignment (install via conda or from source)
# conda install -c conda-forge montreal-forced-alignment
# OR see: https://github.com/MontrealCorpusTools/Montreal-Forced-Alignment
```

**Note**: Some packages may require additional system dependencies:
- **MediaPipe**: May need Visual C++ Redistributable on Windows
- **PyTorch**: Install CUDA version if using GPU
- **librosa**: Requires soundfile backend
- **Montreal Forced Alignment**: Not on PyPI - install via conda or from source

**Troubleshooting Python 3.12:**
If you're using Python 3.12 and encounter errors:
1. Try installing packages one by one to identify problematic ones
2. Some features (phoneme alignment, advanced audio) may be disabled
3. Consider using Python 3.10 or 3.11 for full functionality

## Usage

```bash
# Training
python train.py --config configs/lane_av_rel.yaml

# Evaluation
python evaluate.py --checkpoint checkpoints/best.pth --dataset deepfaketimit

# Federated training
python federated_train.py --num_clients 10 --rounds 100
```

## Dataset Structure

Expected structure:
```
data/
  fake/DeepfakeTIMIT/
    higher_quality/
    lower_quality/
  real/
```

## Citation

If you use this code, please cite:
```
@article{lane_av_rel_2024,
  title={LANe-AV-Rel: Landmark-Aware Audio-Visual Relational Deepfake Detection},
  author={...},
  year={2024}
}
```

