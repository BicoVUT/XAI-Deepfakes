import torch
from torchvision.transforms import v2
import sys
sys.path.append("..")
from model.frame import FrameModel
import numpy as np
from decimal import Decimal
import cv2
import glob
import matplotlib.pyplot as plt

# --- Configuration (Adjust these) ---

# Choose ONE explanation method.
explanation_method = "GradCAM++"
#explanation_method = "LIME"

# Base directory for images
image_dir = "/mnt/d/REPOS/XAI-Deepfakes/explanation/methods"

# --- Model Loading ---

rs_size = 224
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt", map_location=torch.device('cpu')).eval()
task = "multiclass"

# --- Image Transforms ---

interpolation = 3
inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(rs_size, interpolation=interpolation, antialias=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
visualize_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(rs_size, interpolation=interpolation, antialias=False),
    v2.ToDtype(torch.float32, scale=True),
])

# --- Find all JPG files ---
image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))  # Sort for consistent order
if not image_paths:
    raise FileNotFoundError(f"No JPG images found in {image_dir}")

# --- Process Each Image and Store Results ---
results = []  # Store original images and heatmaps

for image_path in image_paths:
    # --- Image Loading ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image at {image_path}, skipping.")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Apply Transforms ---
    inference_image = inference_transforms(image)
    visualize_image = visualize_transforms(image)

    # --- Dummy Label ---
    label_real = torch.tensor([0], dtype=torch.float32)

    # --- Inference (NOW INSIDE THE LOOP) ---
    with torch.no_grad():
        frame = inference_image.to(model.device)
        output = model(frame.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)  # Get probabilities
        predicted_class = torch.argmax(probs, dim=1).item()  # Get predicted class index
        predicted_prob = probs[0, predicted_class].item()  # Probability of predicted class
        real_prob = probs[0, 0].item() #probability of being real
        fake_prob = 1 - real_prob # Probability of being fake


    # --- Explanation ---
    explanation_label_index = predicted_class  # Explain the predicted class

    if explanation_method == "GradCAM++":
        from methods.gradcam_xai import explain as GradCAM
        heatmap = GradCAM(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model, visualize=False)

    elif explanation_method == "LIME":
        from methods.lime_xai import explain as LIME
        heatmap = LIME(visualize_image.permute(1, 2, 0).numpy(), inference_transforms, explanation_label_index, model,  visualize=False)

    else:
        print("Incorrect explanation method")
        sys.exit(0)

    # --- Store Results ---
    results.append({
        "original": visualize_image.permute(1, 2, 0).numpy(),
        "heatmap": heatmap,
        "filename": image_path.split('/')[-1],
        "predicted_prob": predicted_prob,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "predicted_class": predicted_class
    })

# --- Create Subplots and Display ---
num_images = len(results)
fig, axes = plt.subplots(num_images, 2, figsize=(5, 12))

for i, result in enumerate(results):
    # Original Image
    if num_images > 1:
        ax_orig = axes[i, 0]
        ax_expl = axes[i, 1]
    else:  # Handle the case of a single image
        ax_orig = axes[0]
        ax_expl = axes[1]

    class_names = ["Real", "NT", "F2F", "DF", "FSw"]
    predicted_class_name = class_names[result["predicted_class"]]

    ax_orig.imshow(result["original"])
    ax_orig.set_title(
        f"{result['filename']}\n"
        f"Class: {predicted_class_name} ({result['predicted_prob']:.2%})\n"  # Predicted class and percentage
        f"Real: {result['real_prob']:.2%} | Fake: {result['fake_prob']:.2%}"  # Real/Fake percentages
    )
    ax_orig.axis('off')

    # Explanation (Heatmap)
    ax_expl.imshow(result["original"])
    ax_expl.imshow(result["heatmap"], cmap='jet', alpha=0.5)
    ax_expl.set_title(f"{explanation_method}")
    ax_expl.axis('off')


plt.tight_layout()
plt.show()

print("Visualization complete. Plot displayed.")