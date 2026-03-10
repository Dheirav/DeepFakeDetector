from typing import Optional, Literal, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


class GradCAM:
    """Enhanced Grad-CAM for ResNet-like architectures with performance optimizations.

    This implementation combines the best features of multiple Grad-CAM approaches:
    - Automatic layer detection with manual override option
    - Robust normalization with epsilon
    - Proper hook cleanup to prevent memory leaks
    - Support for custom target layers
    - Thread-safe instance-based state management

    Args:
        model: PyTorch model for which to compute Grad-CAM
        target_layer: Optional specific layer to hook. If None, automatically finds last Conv2d
        verbose: If True, prints debug information

    Example:
        >>> model = load_model("model.pth")
        >>> cam = GradCAM(model)
        >>> heatmap = cam(input_tensor, class_idx=1)
        >>> overlay = overlay_heatmap(image, heatmap)
    """

    def __init__(
        self,
        model: object,
        target_layer: Optional[object] = None,
        verbose: bool = False
    ):
        if torch is None:
            raise RuntimeError("PyTorch required for GradCAM")
        
        self.model = model
        self.backbone = model.backbone if hasattr(model, "backbone") else model
        self.device = next(model.parameters()).device
        self.verbose = verbose
        self.activations = None
        self.gradients = None
        self.hooks = []  # Track hooks for cleanup
        
        # Use provided layer or auto-detect
        if target_layer is not None:
            self.target_module = target_layer
        else:
            self.target_module = self._find_target_module(self.backbone)
        
        if self.target_module is None:
            raise RuntimeError("Could not find target convolutional layer. Specify target_layer manually.")
        
        if self.verbose:
            print(f"GradCAM target layer: {self.target_module}")
        
        # Register hooks
        self._register_hooks()

    def _find_target_module(self, model):
        """Automatically find the last Conv2d layer in the model."""
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        return None

    def _register_hooks(self):
        """Register forward and backward hooks."""
        forward_hook = self.target_module.register_forward_hook(self._save_activation)
        backward_hook = self.target_module.register_full_backward_hook(self._save_gradient)
        self.hooks.extend([forward_hook, backward_hook])

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx: int = 0) -> np.ndarray:
        """Generate Grad-CAM heatmap for the specified class.

        Args:
            input_tensor: Input tensor of shape (1, C, H, W) or (C, H, W)
            class_idx: Target class index for which to compute Grad-CAM

        Returns:
            Normalized heatmap as numpy array of shape (H, W) with values in [0, 1]
        """
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Reset gradients
        self.model.zero_grad()
        
        # Move to device and enable gradients
        input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Handle different output shapes
        if logits.dim() == 1:
            score = logits[class_idx]
        else:
            score = logits[0, class_idx]
        
        # Backward pass (no need to retain graph for single backward)
        score.backward()
        
        # Check if hooks captured data
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Failed to capture activations/gradients for Grad-CAM. "
                "Check if target_layer is correct or if model is in eval mode."
            )
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.cpu().squeeze(0).numpy()
        
        # Robust normalization with epsilon to prevent division by zero
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # All values are the same, return zeros
            cam = np.zeros_like(cam)
            if self.verbose:
                print("Warning: Grad-CAM heatmap is uniform (all values identical)")
        
        return cam

    def cleanup(self):
        """Remove hooks to free memory. Call this when done with the GradCAM instance."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations = None
        self.gradients = None

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup()


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    use_opencv: bool = None
) -> Image.Image:
    """Overlay heatmap on image with configurable backend.

    Args:
        pil_image: PIL Image to overlay heatmap on
        heatmap: 2D numpy array with values in [0, 1]
        alpha: Opacity of heatmap overlay (0=transparent, 1=opaque)
        colormap: Colormap name ('jet', 'viridis', 'hot', etc.)
        use_opencv: If True, use OpenCV (faster). If False, use matplotlib (better quality).
                   If None, auto-detect (prefer OpenCV if available)

    Returns:
        PIL Image with heatmap overlay
    """
    # Auto-detect backend
    if use_opencv is None:
        use_opencv = CV2_AVAILABLE
    
    if use_opencv and CV2_AVAILABLE:
        return _overlay_opencv(pil_image, heatmap, alpha, colormap)
    else:
        return _overlay_matplotlib(pil_image, heatmap, alpha, colormap)


def _overlay_opencv(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float,
    colormap: str
) -> Image.Image:
    """Fast OpenCV-based overlay (2-5ms per image)."""
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, pil_image.size)
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    colormap_cv = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_cv)
    
    # Convert PIL image to RGB numpy array
    image_np = np.array(pil_image.convert("RGB"))
    
    # Blend
    overlay_np = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to PIL
    return Image.fromarray(overlay_np)


def _overlay_matplotlib(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float,
    colormap: str
) -> Image.Image:
    """High-quality matplotlib-based overlay (10-20ms per image)."""
    import matplotlib.cm as cm
    
    # Resize heatmap to image size
    heat = Image.fromarray(np.uint8(heatmap * 255)).resize(
        pil_image.size, resample=Image.BILINEAR
    )
    heat_np = np.array(heat) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heat_np)[:, :, :3]  # Drop alpha channel
    colored_img = (colored * 255).astype(np.uint8)
    colored_pil = Image.fromarray(colored_img).convert("RGBA")
    
    # Convert base image to RGBA
    base = pil_image.convert("RGBA")
    
    # Blend
    return Image.blend(base, colored_pil, alpha=alpha)


def create_gradcam_comparison(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """Create side-by-side comparison: original | heatmap | overlay.

    Args:
        pil_image: Original PIL Image
        heatmap: 2D numpy array with Grad-CAM heatmap
        alpha: Opacity for overlay

    Returns:
        PIL Image with 3-panel comparison
    """
    # Resize heatmap to match image
    heat_pil = Image.fromarray(np.uint8(heatmap * 255)).resize(
        pil_image.size, resample=Image.BILINEAR
    )
    
    # Convert heatmap to RGB using jet colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    heat_np = np.array(heat_pil) / 255.0
    colored = cmap(heat_np)[:, :, :3]
    colored_img = (colored * 255).astype(np.uint8)
    heat_colored = Image.fromarray(colored_img)
    
    # Create overlay
    overlay = overlay_heatmap(pil_image, heatmap, alpha=alpha)
    
    # Combine into single image
    width, height = pil_image.size
    combined = Image.new('RGB', (width * 3, height))
    combined.paste(pil_image.convert('RGB'), (0, 0))
    combined.paste(heat_colored, (width, 0))
    combined.paste(overlay.convert('RGB'), (width * 2, 0))
    
    return combined
