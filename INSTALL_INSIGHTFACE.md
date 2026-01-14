# Installing InsightFace

## Why InsightFace?

InsightFace is **better than MediaPipe** for deepfake detection because:
- ✅ **Higher accuracy** for face detection and landmarks
- ✅ **No TensorFlow/protobuf conflicts** (uses ONNX Runtime)
- ✅ **Better for deepfake detection** (more precise landmarks)
- ✅ **GPU acceleration** support

## Installation Steps

### 1. Install Visual C++ Build Tools (Windows only)

InsightFace requires compilation on Windows. Download and install:

**Download:** https://visualstudio.microsoft.com/visual-cpp-build-tools/

1. Download "Build Tools for Visual Studio"
2. During installation, select:
   - **C++ build tools**
   - **Windows 10/11 SDK**
   - **CMake tools** (optional but recommended)

### 2. Install InsightFace and ONNX Runtime

```bash
pip install insightface onnxruntime
```

### 3. Verify Installation

```python
python -c "import insightface; print('InsightFace installed successfully!')"
```

### 4. Test Face Parser

```python
from lane_av_rel.face_parsing import FaceParser
parser = FaceParser()
print("Using:", "InsightFace" if parser.use_insightface else "MediaPipe (fallback)")
```

## Alternative: Use Pre-built Models

If you can't install Visual C++ Build Tools, you can use InsightFace models directly with ONNX Runtime (more complex setup). Contact the maintainers for guidance.

## Current Status

- ✅ ONNX Runtime: Installed
- ❌ InsightFace: Requires Visual C++ Build Tools
- ⚠️ MediaPipe: Available but has protobuf conflicts (fallback only)

## Next Steps

1. Install Visual C++ Build Tools
2. Run: `pip install insightface`
3. Test: `python test_setup.py`

The code will automatically use InsightFace if available, otherwise fall back to MediaPipe.

