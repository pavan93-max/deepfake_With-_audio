"""
Quick test script to verify installation and setup.
Run this before starting training.
"""

import sys
import torch

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"[OK] OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] OpenCV failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] NumPy: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("[OK] MediaPipe: OK")
    except ImportError as e:
        print(f"[FAIL] MediaPipe failed: {e}")
        return False
    
    try:
        import librosa
        print(f"[OK] Librosa: {librosa.__version__}")
    except ImportError as e:
        print(f"[FAIL] Librosa failed: {e}")
        return False
    
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"[OK] TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"[FAIL] TorchVision failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"[OK] TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"[FAIL] TorchAudio failed: {e}")
        return False
    
    return True


def test_model():
    """Test that the model can be created."""
    print("\nTesting model creation...")
    
    try:
        from lane_av_rel.model import LANeAVRel
        
        # Try with graph first
        try:
            model = LANeAVRel(
                roi_size=64,
                num_regions=7,
                region_embed_dim=128,
                audio_embed_dim=128,
                graph_hidden_dim=256,
                num_classes=2,
                use_graph=True
            )
            num_params = sum(p.numel() for p in model.parameters())
            print(f"[OK] Model created successfully (with graph)!")
            print(f"  Total parameters: {num_params:,}")
            return True
        except ImportError as e:
            if "torch_geometric" in str(e):
                print("⚠ Graph features require torch_geometric")
                print("  Testing without graph features...")
                # Try without graph
                model = LANeAVRel(
                    roi_size=64,
                    num_regions=7,
                    region_embed_dim=128,
                    audio_embed_dim=128,
                    graph_hidden_dim=256,
                    num_classes=2,
                    use_graph=False  # Disable graph
                )
                num_params = sum(p.numel() for p in model.parameters())
                print(f"[OK] Model created successfully (without graph)!")
                print(f"  Total parameters: {num_params:,}")
                print("  Note: Install 'pip install torch-geometric' for graph features")
                return True
            else:
                raise
        
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test that dataset can be loaded."""
    print("\nTesting dataset...")
    
    try:
        from lane_av_rel.datasets import DeepfakeTIMITDataset
        from pathlib import Path
        
        data_dir = Path(".")
        fake_dir = data_dir / "fake" / "DeepfakeTIMIT" / "higher_quality"
        
        if not fake_dir.exists():
            print(f"⚠ Dataset directory not found: {fake_dir}")
            print("  Make sure you have the DeepfakeTIMIT dataset in the correct structure")
            return False
        
        # Try to create dataset (don't load all, just check structure)
        print(f"[OK] Dataset directory found: {fake_dir}")
        
        # Count some files
        speaker_dirs = list(fake_dir.iterdir())
        if speaker_dirs:
            print(f"  Found {len(speaker_dirs)} speaker directories")
            return True
        else:
            print("  No speaker directories found")
            return False
            
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA available!")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        return True
    else:
        print("[WARN] CUDA not available - will use CPU (slower)")
        return True  # Not a failure, just info


def main():
    """Run all tests."""
    print("=" * 60)
    print("LANe-AV-Rel Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test model
    results.append(("Model", test_model()))
    
    # Test dataset
    results.append(("Dataset", test_dataset()))
    
    # Test GPU
    results.append(("GPU", test_gpu()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("  python train.py --config configs/lane_av_rel.yaml --data_dir . --output_dir outputs")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()

