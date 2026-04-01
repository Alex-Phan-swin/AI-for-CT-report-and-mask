from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np


#Directory 
input_dir = "output"
output_dir = "image_features"

os.makedirs(output_dir, exist_ok=True)

