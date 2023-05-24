import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import os 
from tqdm import tqdm
def read_image(path_to_nifti, return_numpy=True):
    """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))

 
path = "G:/Dataset/ADNI/TrainingData"  #
folders = os.listdir(path)
for folder in tqdm(folders):
    path_folder = os.path.join(path, folder)
    files = os.listdir(path_folder)
    for filename in files:
        img = read_image(os.path.join(path_folder, filename))
        if img.min() == img.max():
            print(folder, filename)
            print(img.min(), img.max())
            print()