import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(
    BASE_DIR,
    "..",
    "demo_input",
    "TCGA_CS_4941_19960909_11.tif"
)

img = Image.open(img_path)

img_array = np.array(img)

print(img_array.shape)