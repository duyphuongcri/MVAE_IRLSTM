import numpy as np
import torch

class Config(object):
    MODEL_NAME = "MVAE_IRLSTM"
    DATA_TYPE = "IMAGE" # "IMAGE" or "IMAGE_CS" or "IMAGE_DG", "or" "IMAGE_CS_DG"
    PATH_ROOT_IMAGE = "G:/Dataset/ADNI/TrainingData"
    PRETRAIN_DIR = None
    IMAGE_SIZE = [80, 96, 80]
    NUM_PRED_YEAR = 5
    BATCH_SIZE = 2
    NUM_GPUs = 1
    DEVICE=torch.device("cuda:0")
    NUM_EPOCHs = 100
    LABELS = ["CN", "MCI", "AD"]
    NUM_CLASSES = len(LABELS)  # Override in sub-classes
    NUM_HISTORY_VISIT = 2 

    # BACKBONE
    BACKBONE_RNN = "BiLSTM"
    BACKBONE_VAE = "ResNet" ### "ResNet" or "Swin"
    
    ### Feature dimension ---- VAE---
    FEATURES_COGNITIVE_SCORE = ["CDRSB", "ADAS13", "MMSE", "FAQ", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting"]
    FEATURES_DEMOGRAPHIC = ["AGE", "PTEDUCAT", "Male", "Female", "APOE4_0", "APOE4_1", "APOE4_2"]

    FEATURES_DYNAMIC = ["CDRSB", "ADAS13", "MMSE", "FAQ", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting","PTEDUCAT", "AGE"]
    MODALITY = ["MRI", "PET", "DTI"]
    IMAGE_BASE_DIMEN = 16
    IMAGE_LATENT_DIM = IMAGE_BASE_DIMEN * 8
    LATENT_IMG_SIZE = [5, 6, 5]
    LATENT_IMG_DIM = LATENT_IMG_SIZE[0] * LATENT_IMG_SIZE[1] * LATENT_IMG_SIZE[2] * IMAGE_LATENT_DIM
    Z_DIM = 256
    PRIOR_EXPERT= True
    ############## LSTM ################
    INPUT_SIZE = NUM_CLASSES
    if "IMAGE" in DATA_TYPE:
        INPUT_SIZE += Z_DIM
    if "CS" in DATA_TYPE:
        INPUT_SIZE += len(FEATURES_COGNITIVE_SCORE)
    if "DG" in DATA_TYPE:
        INPUT_SIZE += len(FEATURES_DEMOGRAPHIC)

    HIDDEN_SIZE = 256
    OUTPUT_SIZE = INPUT_SIZE - NUM_CLASSES





    
    OPTIMIZER = "Adam"
    # Learning rate and momentum
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    
model_config = Config()
