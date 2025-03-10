import torch
from torchvision.transforms import v2
import sys
# Simplified sys.path modification:
sys.path.append("..")  # Add the parent directory (project root) to the path
from model.frame import FrameModel
import numpy as np
from decimal import Decimal
import cv2  # Import OpenCV for image loading

# --- Configuration (Adjust these) ---

# Choose ONE explanation method.  Start with GradCAM++ or LIME.
explanation_method = "GradCAM++"
#explanation_method = "LIME"
#explanation_method = "RISE"
#explanation_method = "SHAP"
#explanation_method = "SOBOL"


# Path to your test image (ABSOLUTE path is safest)
dataset_example_index = "/mnt/d/REPOS/XAI-Deepfakes/explanation/methods/testing_NT.jpg"  # Example with the provided image
# dataset_example_index = "/path/to/your/downloaded/image.jpg" # Or, path to your own image

# --- Model Loading ---

rs_size = 224
# Use map_location='cpu' if you have GPU issues.  Try 'cuda' first, then 'cpu'.
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt", map_location=torch.device('cpu')).eval()
task = "multiclass"

# --- Image Transforms (Keep these, they're important) ---

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

# --- Image Loading (Modified to use OpenCV) ---

# Load the image using OpenCV
image = cv2.imread(dataset_example_index)
if image is None:
    raise FileNotFoundError(f"Could not load image at {dataset_example_index}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB

# Apply the transforms
inference_image = inference_transforms(image)
visualize_image = visualize_transforms(image)

# --- Dummy Label ---
# We don't have a real label, so create a dummy one.  It doesn't matter for visualization.
label_real = torch.tensor([0], dtype=torch.float32)

# --- Inference (Keep this) ---

with torch.no_grad():
    frame = inference_image.to(model.device)  # Move to GPU if available, otherwise CPU
    output = model(frame.unsqueeze(0))  # Add a batch dimension (unsqueeze)

output = output.cpu().reshape(-1, ).numpy()
real_label = int(label_real.reshape(-1, ).numpy()[0])

print("Output scores: ", end='')
for o in output:
    o = Decimal(str(o))
    print(format(o, 'f'), end=' ')
print("\nPredicted label: " + str(np.argmax(output)))
print("Real label: " + str(real_label))

explanation_label_index = np.argmax(output)
print("\nExplaining predicted label")


# --- Explanation (Modified for dynamic import) ---
if explanation_method == "GradCAM++":
    from methods.gradcam_xai import explain as GradCAM
    GradCAM(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model)
elif explanation_method == "RISE":
    from methods.rise_xai import explain as RISE
    RISE(inference_image, visualize_image.unsqueeze(0), explanation_label_index, model)
elif explanation_method == "SHAP":
    from methods.shap_xai import explain as SHAP
    SHAP(inference_image, visualize_image.permute(1, 2, 0).numpy(), explanation_label_index, model)
elif explanation_method == "LIME":
    from methods.lime_xai import explain as LIME
    LIME(visualize_image.permute(1, 2, 0).numpy(), inference_transforms, explanation_label_index, model)
elif explanation_method == "SOBOL":
    from methods.sobol_xai import explain as SOBOL
    SOBOL(inference_image, visualize_image, explanation_label_index, model)
else:
    print("Incorrect explanation method")
    sys.exit(0)

print("Visualization complete. Check the output in the 'explanation' directory.")