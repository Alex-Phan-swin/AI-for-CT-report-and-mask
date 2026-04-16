import os
import pandas as pd
import nibabel as nib
import numpy as np


class CTDatasetTF:
    """
    Loads CT scans from:
    Dataset/BDMAP_xxx/ct.nii.gz
    """

    def __init__(self, csv_file, data_dir, split="train"):
        self.df = pd.read_csv(csv_file)

        # keep only required split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def load_scan(self, path):
        return nib.load(path).get_fdata()  # (H, W, D)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bdmap_id = row["bdmap_id"]

        # ✅ FIXED PATH (NO bdmap_id in filename)
        file_path = os.path.join(
            self.data_dir,
            bdmap_id,
            "ct.nii.gz"
        )

        scan = self.load_scan(file_path)

        return bdmap_id, scan