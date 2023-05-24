import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch 
import os 

class MedDataset(Dataset):
    
    def __init__(self, data_frame, config, mean, std, mode='train'):
        self.std_age = std[-1]
        self.mean = mean
        self.std = std
        self.data_frame = data_frame.set_index("PTID")
        self.path_root=config.PATH_ROOT_IMAGE
        self.patients_ID = list(data_frame["PTID"].unique())
        self.config=config
        if mode not in ['train', 'test', 'valid']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __len__(self):
        return len(self.patients_ID)
    
    def preprocess_image(self, path):
        img = self.read_image(path)
        # normalize
        img[img < 0.] = 0.
        img = (img- img.min()) / (img.max() - img.min())
        ## padding
        img_pad = np.pad(img, ((1,0), (1,0), (1, 0)), 'constant', constant_values=0)
        return img_pad

    def load_sequence_data(self, patientInfo):
        data_cs = np.full([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_COGNITIVE_SCORE) ], np.nan)
        mask_cs = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_COGNITIVE_SCORE) ],  dtype=bool)
        delta_cs = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_COGNITIVE_SCORE) ], np.float32)

        data_dg = np.full([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_DEMOGRAPHIC)], np.nan)
        mask_dg = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_DEMOGRAPHIC)],  dtype=bool)
        delta_dg = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.FEATURES_DEMOGRAPHIC)], np.float32)        

        data_i = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.MODALITY) + 1]+self.config.IMAGE_SIZE, np.float32) # DTI have 2 images
        mask_i = np.zeros([self.config.NUM_PRED_YEAR + 1, len(self.config.MODALITY)], dtype=bool) 
        delta_i = np.zeros([self.config.NUM_PRED_YEAR + 1, self.config.Z_DIM], np.float32) 

        data_dx = np.full([self.config.NUM_PRED_YEAR + 1, self.config.NUM_CLASSES], np.nan)
        mask_dx = np.zeros([self.config.NUM_PRED_YEAR + 1, self.config.NUM_CLASSES],  dtype=bool)
        delta_dx = np.zeros([self.config.NUM_PRED_YEAR + 1, self.config.NUM_CLASSES], np.float32)
        for i in range(len(patientInfo)):
            year = int(patientInfo["Month"][i] // 12)
            if year == 0: # baseline -> FIll missing value
                ### Fill missing data 
                values = {"APOE4_0": 0, "APOE4_1": 1, "APOE4_2": 0}
                for idx_fea in range(len(self.config.FEATURES_DYNAMIC)):
                    values[self.config.FEATURES_DYNAMIC[idx_fea]] = self.mean[idx_fea]
                patientInfo = patientInfo.copy()
                patientInfo[patientInfo["Month"]==year] = patientInfo[patientInfo["Month"]==year].fillna(value=values)

            if year > self.config.NUM_PRED_YEAR:
                break
            data_cs[year] = np.array(patientInfo[self.config.FEATURES_COGNITIVE_SCORE][patientInfo["Month"] == year*12])
            data_dx[year] = np.array(patientInfo[self.config.LABELS][patientInfo["Month"] == year*12])
            if patientInfo["MRI"][i] == 1:
                mri = self.preprocess_image(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_MRI.nii".format(year*12)))
                data_i[year, 0] = mri
                mask_i[year, 0] = 1

            if patientInfo["PET"][i] == 1:
                if os.path.exists(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_PET_AV45.nii".format(year*12))):
                    pet = self.preprocess_image(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_PET_AV45.nii".format(year*12)))
                else:
                    pet = self.preprocess_image(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_PET_FDG.nii".format(year*12)))
                data_i[year, 1] = pet
                mask_i[year, 1] = 1

            if patientInfo["DTI"][i] == 1:
                dti_fa = self.preprocess_image(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_DTI_FA.nii".format(year*12)))
                dti_md = self.preprocess_image(os.path.join(self.path_root, patientInfo.index[0], "visit_{:0>3}_DTI_MD.nii".format(year*12)))
                data_i[year, 2] = dti_fa
                data_i[year, 3] = dti_md
                mask_i[year, 2] = 1

        demographic = np.array(patientInfo[self.config.FEATURES_DEMOGRAPHIC][patientInfo["Month"] == 0]).reshape(1, -1)
        data_dg = np.repeat(demographic, self.config.NUM_PRED_YEAR + 1, axis=0)
        mask_dg[data_dg == data_dg] = 1. # Check if not nan

        #### mask
        mask_cs[data_cs == data_cs] = 1. # Check if not nan
        mask_dx[data_dx == data_dx] = 1.
        
        ### Delta
        last_observed_cs = np.zeros(len(self.config.FEATURES_COGNITIVE_SCORE), np.float32)
        last_observed_dg = np.zeros(len(self.config.FEATURES_DEMOGRAPHIC), np.float32 )
        last_observed_dx = np.zeros(self.config.NUM_CLASSES, np.float32)
        last_observed_i = np.zeros(self.config.Z_DIM, np.float32)
        for tps in range(self.config.NUM_PRED_YEAR + 1):
            # demographic
            if tps == 0:
                age_bl = np.array(patientInfo["AGE"][patientInfo["Month"] == 0])
            data_dg[tps, 0] =  age_bl + tps / self.std_age # first column is age features
            delta_dg[tps] = tps - last_observed_dg
            last_observed_dg[mask_dg[tps]] = tps

            # Cognitive scores
            delta_cs[tps] = tps - last_observed_cs
            last_observed_cs[mask_cs[tps]] = tps
            # Labels
            delta_dx[tps] = tps - last_observed_dx
            last_observed_dx[mask_dx[tps]] = tps
            # Images
            delta_i[tps] = tps - last_observed_i
            last_observed_i[np.any(mask_i[tps]==True)] = tps

        return data_cs, mask_cs, delta_dg, data_dg, mask_dg, delta_cs, data_i, mask_i, delta_i, data_dx, mask_dx, delta_dx
    
    def __getitem__(self, index):
        sample = dict()

        patientid = self.patients_ID[index]
        sample['id'] = patientid
        patientInfo = self.data_frame.loc[patientid]
        data_cs, mask_cs, delta_dg, data_dg, mask_dg, delta_cs, data_i, mask_i, delta_i, data_dx, mask_dx, delta_dx = self.load_sequence_data(patientInfo)

        sample["data_cs"]  = data_cs.astype(np.float32)
        sample["mask_cs"]  = mask_cs
        sample["delta_cs"] = delta_cs

        sample["data_dg"]  = data_dg.astype(np.float32)
        sample["mask_dg"]  = mask_dg
        sample["delta_dg"] = delta_dg

        sample["data_i"]  = data_i.astype(np.float32)
        sample["mask_i"]  = mask_i
        sample["delta_i"] = delta_i

        sample["data_dx"]  = data_dx.astype(np.float32)
        sample["mask_dx"]  = mask_dx
        sample["delta_dx"] = delta_dx
   
        return sample

    @staticmethod
    def read_image(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))
    
import pathlib as plb
def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)
        
        #dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs