import os
import pandas as pd
import numpy as np
import nibabel as nib
import gzip

class CTDatasetTF:
    # Initialize the dataset with the path to the CSV file, data directory, and file type (nii or npy)
    def __init__(self, csv_file, data_dir, file_type="nii"):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.file_type = file_type

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.df)

    # Load scan based on file type (nii or npy)
    def load_scan(self, path):
        if self.file_type == "nii":
            scan = nib.load(path).get_fdata()   # (H, W, D)
        elif self.file_type == "npy":
            with gzip.open(path, "rb") as f:
                scan = np.load(f)
        else:
            raise ValueError("Unsupported file type")
        return scan

    # Get item by index, load the corresponding scan and return it along with the bdmap_id
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bdmap_id = row["bdmap_id"]

        file_path = os.path.join(self.data_dir,bdmap_id,'ct.nii.gz')

        scan = self.load_scan(file_path)

        return bdmap_id, scan