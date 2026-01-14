# Quick Start Guide

## 🚀 Training Commands

### 1. **Start Training (Recommended - Uses Cleaned Dataset)**

```bash
c\Scripts\python.exe train.py --data_dir . --cleaned_dir ./cleaned_dataset --output_dir outputs
```

This uses the preprocessed cleaned dataset for **10-100x faster training**.

### 2. **Start Training (Raw Data - Slower)**

```bash
c\Scripts\python.exe train.py --data_dir . --output_dir outputs
```

This processes videos/audio on-the-fly (much slower).

### 3. **Resume Training from Checkpoint**

```bash
c\Scripts\python.exe train.py --data_dir . --cleaned_dir ./cleaned_dataset --resume outputs/latest.pth
```

### 4. **Train on CPU (if no GPU)**

```bash
c\Scripts\python.exe train.py --data_dir . --cleaned_dir ./cleaned_dataset --device cpu
```

### 5. **Use Custom Config**

```bash
c\Scripts\python.exe train.py --config configs/lane_av_rel.yaml --data_dir . --cleaned_dir ./cleaned_dataset
```

## 📊 Monitor Training

### TensorBoard (Real-time Metrics)

```bash
c\Scripts\tensorboard.exe --logdir outputs/logs
```

Then open: http://localhost:6006

### Check Outputs

- **Best model**: `outputs/best.pth`
- **Latest checkpoint**: `outputs/latest.pth`
- **Logs**: `outputs/logs/` (TensorBoard)

## 🧪 Evaluate Model

```bash
c\Scripts\python.exe evaluate.py --checkpoint outputs/best.pth --data_dir . --cleaned_dir ./cleaned_dataset
```

## 📝 Configuration

Edit `configs/lane_av_rel.yaml` to adjust:
- `batch_size`: Increase if you have more GPU memory
- `lr`: Learning rate
- `num_epochs`: Training epochs
- `video_length`: Number of frames per video

## ⚡ Performance Tips

1. **Use cleaned dataset** - 10-100x faster
2. **Increase batch_size** if you have GPU memory
3. **Reduce num_workers** if you have memory issues
4. **Use GPU** - Much faster than CPU

## 🔍 Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `num_workers` in config
- Use smaller `video_length`

### Slow Training
- Make sure you're using `--cleaned_dir ./cleaned_dataset`
- Check if GPU is being used: `nvidia-smi`
- Reduce `num_workers` if CPU is bottleneck

### Dataset Not Found
- Run preprocessing first: `python preprocess_dataset.py`
- Check `cleaned_dataset/` directory exists

