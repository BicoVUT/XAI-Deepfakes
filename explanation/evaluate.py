import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
# sys.path.append('../src')   # REMOVE THESE LINES
# sys.path.append('./methods') # REMOVE THESE LINES
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.frame import FrameModel  # Corrected relative import
from data.datasets import DeepfakeDataset  # Corrected relative import
from evaluation.compute_metrics import computeExplanationMetrics #NO DOT
from evaluation.generate_ff_test_data import getFFPath #NO DOT


#"GradCAM++" - "RISE" - "SHAP" - "LIME" - "SOBOL" - "All"
evaluation_explanation_methods="All"


valid_methods=["GradCAM++", "RISE", "SHAP", "LIME", "SOBOL", "All"]
if(evaluation_explanation_methods not in valid_methods):
    print("Invalid explanation method(s) to evaluate")
    sys.exit(0)

#Load the model
rs_size = 224
task = "multiclass"
model = FrameModel.load_from_checkpoint("../model/checkpoint/ff_attribution.ckpt",map_location='cuda').eval()

#Create the transforms for inference and visualization purposes
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

#Create the deepfake test examples and load the dataset
ds_path = getFFPath("../data/csvs/ff_Smaller_test.csv")

#Dataset with inference transformations
target_transforms = lambda x: torch.tensor(x, dtype=torch.float32)
ds = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=inference_transforms,
    target_transforms=target_transforms,
    task=task
)
#Dataset with visualization transformations
ds_visualize = DeepfakeDataset(
    ds_path,
    "../data/xai_test_data.lmdb",
    transforms=visualize_transforms,
    target_transforms=target_transforms,
    task=task
)

if not os.path.exists('./results'):
    os.makedirs('./results')

#Call the corresponding function to compute them
computeExplanationMetrics(model, ds, ds_visualize, inference_transforms, evaluation_explanation_methods)


#Load the saved results
save_name="results_"+evaluation_explanation_methods
scores = np.load("./results/"+save_name+".npy")
scores = list(scores)

#Compute the mean values
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    scores=np.nanmean(scores,axis=0)

#Create a dataframe and save the results in a csv
if(evaluation_explanation_methods=="All"):
    index_values = ["Original", "GradCAM++", "LIME"]
else:
    index_values = ["Original", evaluation_explanation_methods]

column_values = ["Sufficiency DF Top 1","Sufficiency DF Top 2","Sufficiency DF Top 3", "Stability DF",
                 "Sufficiency F2F Top 1","Sufficiency F2F Top 2","Sufficiency F2F Top 3", "Stability F2F",
                 "Sufficiency FS Top 1","Sufficiency FS Top 2","Sufficiency FS Top 3", "Stability FS",
                 "Sufficiency NT Top 1","Sufficiency NT Top 2","Sufficiency NT Top 3", "Stability NT",
                 "Accuracy",
                 "Accuracy DF (Top 1)","Accuracy DF Top 2", "Accuracy DF Top 3",
                 "Accuracy F2F (Top 1)","Accuracy F2F Top 2", "Accuracy F2F Top 3",
                 "Accuracy FS (Top 1)","Accuracy FS Top 2", "Accuracy FS Top 3",
                 "Accuracy NT (Top 1)","Accuracy NT Top 2", "Accuracy NT Top 3"]

df = pd.DataFrame(data=scores,index=index_values,columns=column_values)
csv_save_name="scores_"+evaluation_explanation_methods
df.round(3).to_csv("./results/"+csv_save_name+".csv",sep=',')
print(df.round(3).to_string())