#!/usr/bin/env python3
"""
Enhanced Grad-CAM Demonstration Script

This script demonstrates the merged and improved Grad-CAM implementation
that combines the best features of both previous versions.

Features demonstrated:
- Automatic and manual layer selection
- OpenCV (fast) vs matplotlib (high-quality) overlays
- Multiple visualization modes
- Proper memory cleanup
- Error handling and robustness

Usage:
    python demo_gradcam.py --image path/to/image.jpg --model models/best_model.pth
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from frontend.gradcam import GradCAM, overlay_heatmap, create_gradcam_comparison
from frontend.inference import load_model, preprocess_image
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description="Enhanced Grad-CAM Demo")
    parser.add_argument("--image", type=str, default="sample.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class (None=use prediction)")
    parser.add_argument("--backend", type=str, choices=["opencv", "matplotlib", "auto"], default="auto",
                       help="Overlay backend: opencv (fast) or matplotlib (quality)")
    parser.add_argument("--colormap", type=str, default="jet", help="Colormap name")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity (0-1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device: cuda or cpu")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Enhanced Grad-CAM Demonstration")
    print("=" * 60)

    # Load model
    print(f"\n[1/5] Loading model from {args.model}...")
    try:
        model = load_model(args.model, device=args.device)
        model.eval()
        print(f"✓ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1

    # Load image
    print(f"\n[2/5] Loading image from {args.image}...")
    try:
        image = Image.open(args.image).convert("RGB")
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return 1

    # Preprocess image
    print("\n[3/5] Preprocessing image...")
    try:
        input_tensor = preprocess_image(image)
        print(f"✓ Preprocessed to tensor: {input_tensor.shape}")
    except Exception as e:
        print(f"✗ Failed to preprocess: {e}")
        return 1

    # Run inference
    print("\n[4/5] Running inference...")
    try:
        with torch.no_grad():
            # preprocess_image already includes the batch dimension [1,C,H,W]
            input_batch = input_tensor.to(args.device)
            output = model(input_batch)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
        
        class_names = ["Real", "AI Generated", "AI Edited"]
        print(f"✓ Prediction: {class_names[pred_class]} ({probs[pred_class]:.2%} confidence)")
        print(f"  Class probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            print(f"    {i}: {name:15s} {prob:.2%}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return 1

    # Determine target class
    target_class = args.class_idx if args.class_idx is not None else pred_class
    print(f"\n  Using target class: {target_class} ({class_names[target_class]})")

    # Generate Grad-CAM
    print("\n[5/5] Generating Grad-CAM heatmap...")
    try:
        # Initialize GradCAM with optional verbosity
        cam = GradCAM(model, verbose=args.verbose)
        
        # Generate heatmap for target class
        heatmap = cam(input_tensor, class_idx=target_class)
        print(f"✓ Heatmap generated: {heatmap.shape}")
        print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Cleanup (important for memory management)
        cam.cleanup()
        
    except Exception as e:
        print(f"✗ Grad-CAM generation failed: {e}")
        return 1

    # Create visualizations
    print("\n[6/5] Creating visualizations...")
    
    # Determine backend
    use_opencv = None if args.backend == "auto" else (args.backend == "opencv")
    
    # 1. Simple overlay
    print("  - Creating overlay...")
    overlay = overlay_heatmap(
        image, heatmap, 
        alpha=args.alpha, 
        colormap=args.colormap,
        use_opencv=use_opencv
    )
    overlay_path = output_dir / f"gradcam_overlay_{args.colormap}.png"
    overlay.save(overlay_path)
    print(f"    ✓ Saved: {overlay_path}")
    
    # 2. Comparison view (original | heatmap | overlay)
    print("  - Creating comparison view...")
    comparison = create_gradcam_comparison(image, heatmap, alpha=args.alpha)
    comparison_path = output_dir / "gradcam_comparison.png"
    comparison.save(comparison_path)
    print(f"    ✓ Saved: {comparison_path}")
    
    # 3. Try different colormaps (if using default)
    if args.colormap == "jet":
        print("  - Creating alternative colormaps...")
        for cmap in ["viridis", "hot"]:
            try:
                alt_overlay = overlay_heatmap(
                    image, heatmap, 
                    alpha=args.alpha, 
                    colormap=cmap,
                    use_opencv=False  # matplotlib for more colormap options
                )
                alt_path = output_dir / f"gradcam_overlay_{cmap}.png"
                alt_overlay.save(alt_path)
                print(f"    ✓ Saved: {alt_path}")
            except Exception as e:
                print(f"    ⚠ Failed to create {cmap} overlay: {e}")

    print("\n" + "=" * 60)
    print("✓ Grad-CAM generation complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"  - Overlay: {overlay_path.name}")
    print(f"  - Comparison: {comparison_path.name}")
    
    # Performance comparison info
    print("\n" + "=" * 60)
    print("Backend Performance Notes:")
    print("=" * 60)
    print("• OpenCV backend:      Fast (~2-5ms),   Good quality")
    print("• matplotlib backend:  Slower (~10-20ms), Best quality")
    print("• Use OpenCV for real-time applications")
    print("• Use matplotlib for publication-quality figures")
    
    return 0


if __name__ == "__main__":
    exit(main())
