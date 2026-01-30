import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "models/baseline_resnet18.pth"
IMAGE_PATH = "sample.jpg"   # put any test image path here
OUTPUT_PATH = "results/gradcam_output.jpg"

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Hook storage
features = None
gradients = None

# Hook functions
def forward_hook(module, input, output):
    global features
    features = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

# Register hooks (last conv layer)
target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Forward pass
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# Backward pass
model.zero_grad()
output[0, pred_class].backward()

# Generate Grad-CAM
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * features).sum(dim=1).squeeze()

cam = F.relu(cam)
cam = cam.cpu().numpy()
cam = cam / cam.max()

# Resize CAM to image size
cam = cv2.resize(cam, image.size)

# Convert image for overlay
image_np = np.array(image)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

# Save output
os.makedirs("results", exist_ok=True)
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("Prediction:", CLASS_NAMES[pred_class])
print("Grad-CAM saved at:", OUTPUT_PATH)
