# Explainability Scripts

This directory contains scripts for model explainability and visualization:

- `grad_cam.py`: Generates Grad-CAM heatmaps for visualizing model attention on input images.

**Usage:**
- Run the script after training to generate heatmaps for selected images.
- Example:
  ```bash
  python grad_cam.py --model_path models/best_resnet18.pth --image_path data/real/example.jpg
  ```
