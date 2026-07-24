"""Grad-CAM auf mehreren Beispielbildern - Original und Heatmap nebeneinander."""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data import transform
from model import build_model

CHECKPOINT_PATH = "best_model.pth"
TEST_DIR = "data/chest_xray/test"
CLASSES = ["NORMAL", "PNEUMONIA"]
N_PER_CLASS = 3

model = build_model(pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

n_rows = len(CLASSES) * N_PER_CLASS
fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))

row = 0
for cls in CLASSES:
    folder = os.path.join(TEST_DIR, cls)
    filenames = sorted(os.listdir(folder))[:N_PER_CLASS]

    for fname in filenames:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
        pred_class = CLASSES[probs.argmax().item()]
        pred_prob = probs.max().item()

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = np.array(img.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_title(f"Original - Wahr: {cls}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(visualization)
        axes[row, 1].set_title(f"Grad-CAM - Vorhersage: {pred_class} ({pred_prob:.2f})", fontsize=10)
        axes[row, 1].axis("off")

        row += 1

plt.tight_layout()
plt.savefig("gradcam_grid.png")
print("Gespeichert als gradcam_grid.png")