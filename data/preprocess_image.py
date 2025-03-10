import logging as log
import os
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm
import click
from utils import (
    FaceDetector,
    apply_bboxes,
)  # Import only necessary utilities

torch.backends.cudnn.benchmark = False


@click.group()
def cli():
    pass


@click.command()
@click.option("--image_path", "-i", type=str, required=True, help="Path to the input image")
@click.option("--output_path", "-o", type=str, required=True, help="Path to save the cropped face")
@click.option("--device", "-d", type=str, default="cuda:0", help="Device (e.g., 'cuda:0' or 'cpu')")
@click.option("--batch_size", "-bs", type=int, default=64, help="Batch size for the face detector")
@click.option("--threshold", "-t", type=float, default=0.95, help="Detection threshold")
@click.option("--increment", "-inc", type=float, default=0.01, help="Increment for threshold adjustment")
@click.option("--scale", "-s", type=float, default=1.3, help="Scaling factor for the bounding box")
@click.option("--ext", "-e", type=str, default=".png", help="Output image extension (.jpg, .png, etc.)")
@click.option("--quality", "-q", type=int, default=95, help="Image quality (for JPEG, 0-100)")
def crop_face(
    image_path,
    output_path,
    device,
    batch_size,
    threshold,
    increment,
    scale,
    ext,
    quality,
):
    """Crops a face from a single image."""

    # --- Input Validation and Setup ---
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if ext.lower() not in {".jpg", ".jpeg", ".png"}:  # More robust extension check
        raise ValueError("Invalid output extension.  Must be .jpg, .jpeg, or .png")

    # --- Face Detection ---
    detector = FaceDetector(device, batch_size, threshold, increment)
    image = cv2.imread(str(image_path))  # Read with cv2, ensure string path
    if image is None:
      raise ValueError(f"Could not open or read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # To RGB
    image_batch = [image]  # Create a "batch" of one image

    boxes, _, _ = detector(image_batch)  # Use the detector
    boxes = boxes[0]  # Get boxes for the *first* (and only) image in the batch

    if boxes is None or len(boxes) == 0:  # Correctly check for empty boxes
        print(f"No faces detected in {image_path}")
        return  # Exit if no faces found

    if len(boxes) > 1:
        print(f"Warning: Multiple faces detected in {image_path}.  Cropping the first detected face.")
        #  Consider adding logic to choose the "best" face (e.g., highest confidence)

    # --- Cropping ---
    cropped_faces = apply_bboxes(image_batch, [boxes], scale=scale)
    cropped_face = cropped_faces[0][0]  # Get the first (and likely only) cropped face

    # --- Saving ---
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR) # Back to BGR for cv2

    if ext.lower() == ".jpg" or ext.lower() == ".jpeg":
        cv2.imwrite(str(output_path), cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif ext.lower() == ".png":
        cv2.imwrite(str(output_path), cropped_face)  # PNG doesn't use quality parameter
    else: # Should not be possible, because of previous check
        raise ValueError("Invalid output extension.")
    print(f"Cropped face saved to {output_path}")



cli.add_command(crop_face)


if __name__ == "__main__":
    cli()