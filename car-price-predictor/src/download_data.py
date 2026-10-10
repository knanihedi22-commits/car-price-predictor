"""Optional: download the dataset automatically (pip install kagglehub).
If it fails, download manually from Kaggle and unzip the CSVs into data/raw/."""
import glob
import os
import shutil

import kagglehub

path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")
os.makedirs("data/raw", exist_ok=True)
for f in glob.glob(os.path.join(path, "*.csv")):
    shutil.copy(f, "data/raw/")
print("CSV files copied to data/raw/")
