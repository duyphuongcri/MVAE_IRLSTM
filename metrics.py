import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix # classification
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error # imputation
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio  # reconstruct

class evaluate_DX():
    def __init__(self,):
        self.preds = []
        self.targets_num = []
        self.probs = []
        self.targets_cat = []
        self.auc = 0
        self.N = 0

    def average(self):
        acc = accuracy_score(self.targets_num, self.preds)
        pre = precision_score(self.targets_num, self.preds, average="weighted")
        rec = recall_score(self.targets_num, self.preds, average="weighted")
        auc = roc_auc_score(self.targets_cat, self.probs, average="weighted", multi_class='ovr')

        return acc, pre, rec, auc 
    
    def measure(self, logits, targets, mask):
        """
        logits : B x L x 3
        targets: B x L x 3
        mask   : B x L x 3
        """
        probs = logits.softmax(-1)[mask].detach().cpu().numpy().reshape(-1, 3)
        targets_cat = targets[mask].detach().cpu().numpy().reshape(-1, 3)
        if np.isnan(probs).any():
            print(probs)
            print(targets_cat)
        preds   = torch.argmax(logits[mask].reshape(-1, 3), dim=-1).detach().cpu().numpy()
        targets_num = torch.argmax(targets[mask].reshape(-1, 3), dim=-1).detach().cpu().numpy()

        if self.preds == []:
            self.preds = preds
            self.targets_num = targets_num
            self.probs = probs
            self.targets_cat = targets_cat
        else:
            self.preds       = np.concatenate((self.preds, preds), axis=0)
            self.targets_num = np.concatenate((self.targets_num, targets_num), axis=0)
            self.probs       = np.concatenate((self.probs, probs), axis=0)
            self.targets_cat = np.concatenate((self.targets_cat, targets_cat), axis=0)

class evaluate_reconstruct():
    def __init__(self,):
        self.ssim = 0
        self.psnr = 0
        self.mse = 0
        self.N = 0

    def average(self):
        ssim = self.ssim /self.N
        psnr = self.psnr /self.N
        mse = self.mse /self.N
        return ssim, psnr, mse
       
    def measure(self, logits, targets, mask):
        """
        logits : B x L x C x D x H x W
        targets: B x L x C x D x H x W
        mask   : B x L x 3
        """
        mask = torch.cat((mask, mask[:, :, 2:]), dim=2) ## duplicate mask of DTI due to its multiple channels (2 channels: FA and MD)

        logits = logits[mask].detach().cpu().numpy().astype(np.float64)
        targets = targets[mask].detach().cpu().numpy().astype(np.float64)

        # logits[logits < 0.] = 0.
        # logits[logits > 1.] = 1.

        # targets[targets < 0.] = 0.
        # targets[targets > 1.] = 1.

        self.ssim += structural_similarity(targets, logits, data_range=1., channel_axis=0)
        self.psnr += peak_signal_noise_ratio(targets, logits)
        self.mse  += mean_squared_error(targets, logits)
        self.N += 1
        # print(structural_similarity(targets, logits, data_range=1.), peak_signal_noise_ratio(targets, logits), mean_squared_error(targets, logits))