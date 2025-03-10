import os
import pandas as pd
import cv2

# Define the root directory of your dataset
root_dir = "/mnt/d/DATASETS/FF_test_Small"

# Initialize an empty list to store the data
data = []

# Function to process a video and add its information to the data list
def process_video(video_path, label, manipulation_type=None):
    # Get video name without extension
    video_name = os.path.basename(video_path).split('.')[0]

    # Original video path (handle both fake and real)
    original_name = video_name.split('_')[0] + ".mp4"
    if label == "REAL":
        original_name = os.path.basename(video_path)

    # Construct the relative path for c40_path
    relative_path = os.path.relpath(video_path, root_dir)

    # Get video metadata using OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Convert to integer!
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        print(f"Error getting metadata for {video_path}: {e}")
        height, width, frame_rate, nb_frames = 0, 0, 0, 0

    data.append({
        "c0_path": "",
        "split": "test",
        "label": label,
        "original": original_name,
        "manipulation": manipulation_type if label == "FAKE" else "",
        "height": height,
        "width": width,
        "frame_rate": frame_rate,  # Now an integer
        "nb_frames": nb_frames,
        "c23_path": "",
        "c40_path": relative_path,
    })

# Walk through the dataset directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(subdir, file)

            # Determine label and manipulation type
            if "original_sequences" in subdir:
                label = "REAL"
                manipulation_type = None
            elif "manipulated_sequences" in subdir:
                label = "FAKE"
                # Extract manipulation type
                for manipulation in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]:
                    if manipulation in subdir:
                        manipulation_type = manipulation
                        break
            else:
                continue  # Skip files not in expected directories

            process_video(video_path, label, manipulation_type)

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_path = os.path.join("data", "csvs", "dataset_info_v2.csv")
df.to_csv(csv_path, index=False)

print(f"dataset_info.csv created at {csv_path}")