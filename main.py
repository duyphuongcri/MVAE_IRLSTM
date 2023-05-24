import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
import misc 
from config import model_config
import dataloader
from tqdm import tqdm
from models import mvae_irlstm
from loss import *
from metrics import *

def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))

def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    #print('Initialize network with %s' % init_type)
    net.apply(init_func)

def postprocess_dataloader(sample, device):
    mask = torch.cat([sample["mask_cs"], sample["mask_i"], sample["mask_dx"]], dim=-1) # B x L x Dim
    last_tps_has_data = 0
    for tps in range(mask.shape[1]):
        if torch.any(mask[:, tps]==True):
            last_tps_has_data = tps
    sample["data_cs"]  = sample["data_cs"][:, : last_tps_has_data + 1].to(device)
    sample["mask_cs"]  = sample["mask_cs"][:, : last_tps_has_data + 1].to(device)
    sample["delta_cs"] = sample["delta_cs"][:, : last_tps_has_data + 1].to(device)

    sample["data_dg"]  = sample["data_dg"][:, : last_tps_has_data + 1].to(device)
    sample["mask_dg"]  = sample["mask_dg"][:, : last_tps_has_data + 1].to(device)
    sample["delta_dg"] = sample["delta_dg"][:, : last_tps_has_data + 1].to(device)

    sample["data_i"]  = sample["data_i"][:, : last_tps_has_data + 1].to(device)
    sample["mask_i"]  = sample["mask_i"][:, : last_tps_has_data + 1].to(device)
    sample["delta_i"] = sample["delta_i"][:, : last_tps_has_data + 1].to(device)

    sample["data_dx"]  = sample["data_dx"][:, : last_tps_has_data + 1].to(device)
    sample["mask_dx"]  = sample["mask_dx"][:, : last_tps_has_data + 1].to(device)
    sample["delta_dx"] = sample["delta_dx"][:, : last_tps_has_data + 1].to(device)

    return  (sample["data_cs"], sample["data_dg"], sample["data_i"], sample["data_dx"]), \
            (sample["mask_cs"], sample["mask_dg"], sample["mask_i"], sample["mask_dx"]), \
            (sample["delta_cs"], sample["delta_dg"], sample["delta_i"], sample["delta_dx"])

if __name__=="__main__":
    data_plit = pd.read_csv("./dataset/ADNI/data_ptid_split_5_folds.csv")
    frame = pd.read_csv("./dataset/ADNI/data_cleaned.csv")
    for fold in range(1):
        ##################### LOAD DATA ###################
        list_ptid_train = list(data_plit[data_plit["Fold_{}".format(fold)] == 0.]["PTID"])
        list_ptid_valid = list(data_plit[data_plit["Fold_{}".format(fold)] == 1.]["PTID"])

        frame_train = frame[frame["PTID"].isin(list_ptid_train)]
        frame_valid = frame[frame["PTID"].isin(list_ptid_valid)]
        # Normalize and standardize dataset.
        mean = frame_train[model_config.FEATURES_DYNAMIC].mean()
        std = frame_train[model_config.FEATURES_DYNAMIC].std()

        frame_train, frame_valid = frame_train.copy(), frame_valid.copy()
        frame_train["RAVLT_perc_forgetting"] = frame_train["RAVLT_perc_forgetting"].abs()
        frame_valid["RAVLT_perc_forgetting"] = frame_valid["RAVLT_perc_forgetting"].abs()

        frame_train, frame_valid = frame_train.copy(), frame_valid.copy()
        frame_train[model_config.FEATURES_DYNAMIC] = (frame_train[model_config.FEATURES_DYNAMIC] - mean) / std
        frame_valid[model_config.FEATURES_DYNAMIC] = (frame_valid[model_config.FEATURES_DYNAMIC] - mean) / std

        train_set = dataloader.MedDataset(frame_train, model_config, mean, std, mode="train")
        valid_set = dataloader.MedDataset(frame_valid, model_config, mean, std, mode="valid")
        # Dataloaders:
        train_loader = DataLoader(train_set, batch_size=model_config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        dataloaders = {
            'train': train_loader,
            'valid': valid_loader
        }
        #######################################################
        verbose = True
        log = print if verbose else lambda *x, **i: None
        np.random.seed(10)
        torch.manual_seed(10)

        ######################## LOAD MODEL  ###############################
        if model_config.MODEL_NAME == "MVAE_IRLSTM":
            model = mvae_irlstm.MVAE_IRLSTM(model_config)


        ######################## MODEL SETTTING  ###################
        if model_config.NUM_GPUs == 1:
            model.to(model_config.DEVICE)
        else:
            model = nn.DataParallel(model)
            model.to(model_config.DEVICE)
        
        ################### Load pretrained model ###########################################
        if model_config.PRETRAIN_DIR is not None:
            print("Loading pretrained model")
            assert os.path.exists(model_config.PRETRAIN_DIR), "Pretrained dir does not exist"
            pretrain = torch.load(model_config.PRETRAIN_DIR)
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k, v in pretrain.state_dict().items() if k in model.state_dict()}
            model_dict.update(pretrained_dict)
                
            model.load_state_dict(model_dict)
        else: 
            print("Training from Scratch")
            model.apply(init_weights)

        #################### TRAINING SETTING #############################
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE, betas=(0.9, 0.99))
        ###################### TRAINING PHASE ###############################

        dx_metrics_train, recon_metrics_train    = evaluate_DX(), evaluate_reconstruct() 
        dx_metrics_valid, recon_metrics_valid    = evaluate_DX(), evaluate_reconstruct() 

        filename = "model.pt"
        data_training = []
        min_loss_dx=np.inf
        for epoch in range(model_config.NUM_EPOCHs):
            loss_total_train, loss_kl_train, loss_rec_train, loss_diag_train = 0, 0, 0, 0
            loss_total_valid, loss_kl_valid, loss_rec_valid, loss_diag_valid = 0, 0, 0, 0

            model.train()
            for sample in tqdm(dataloaders['train']):
                # if "032_S_2247" != sample["id"][0]:
                #     continue
                data, mask, delta = postprocess_dataloader(sample, model_config.DEVICE) ### [CS, DG, I, Y]
                # fed data into model
                data_i_hat, multi_mu_logvar, data_dx_hat, data_cs_hat = model(data, mask, delta)

                loss_kl    = KLDivergence(multi_mu_logvar, mask[2])
                loss_rec   = recontructLoss(data_i_hat, data[2], mask[2])
                loss_diag  = dxLoss(data_dx_hat, data[3][:,1:], mask[3][:,1:]) ### get target data from 2nd timepoints
                loss_total = loss_kl + loss_rec + loss_diag
                ### UPdate paras
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                ######################## 
                dx_metrics_train.measure(data_dx_hat, data[3][:,1:], mask[3][:,1:])
                recon_metrics_train.measure(data_i_hat, data[2], mask[2])
                
                ##########
                loss_kl_train    += loss_kl.item()
                loss_rec_train   += loss_rec.item()
                loss_diag_train  += loss_diag.item()
                loss_total_train += loss_total.item() 

            ###
            loss_kl_train    = loss_kl_train / len(dataloaders['train'])
            loss_rec_train   = loss_rec_train / len(dataloaders['train'])
            loss_diag_train  = loss_diag_train / len(dataloaders['train'])
            loss_total_train = loss_total_train / len(dataloaders['train'])

            acc_train, pre_train, rec_train, auc_train = dx_metrics_train.average()
            ssim_train, psnr_train, mse_train = recon_metrics_train.average()
            # print(acc_train, pre_train, rec_train, auc_train)
            #######

            ######################################################################################## 
            model.eval()  
            with torch.no_grad():
                for sample in tqdm(dataloaders['valid']):
                    data, mask, delta = postprocess_dataloader(sample, model_config.DEVICE) ### [CS, DG, I, Y]
                    # fed data into model
                    data_i_hat, multi_mu_logvar, data_dx_hat, data_cs_hat = model(data, mask, delta)
                    loss_kl    = KLDivergence(multi_mu_logvar, mask[2])
                    loss_rec   = recontructLoss(data_i_hat, data[2], mask[2])
                    loss_diag  = dxLoss(data_dx_hat, data[3][:,1:], mask[3][:,1:]) ### get target data from 2nd timepoints
                    loss_total = loss_kl + loss_rec + loss_diag
          
                    ###########
                    dx_metrics_valid.measure(data_dx_hat, data[3][:,1:], mask[3][:,1:])
                    recon_metrics_valid.measure(data_i_hat, data[2], mask[2])
                    ##########
                    loss_kl_valid    += loss_kl.item()
                    loss_rec_valid   += loss_rec.item()
                    loss_diag_valid  += loss_diag.item()
                    loss_total_valid += loss_total.item() 
                
                loss_kl_valid    = loss_kl_valid / len(dataloaders['valid'])
                loss_rec_valid   = loss_rec_valid / len(dataloaders['valid'])
                loss_diag_valid  = loss_diag_valid / len(dataloaders['valid'])
                loss_total_valid = loss_total_valid / len(dataloaders['valid'])

                acc_valid, pre_valid, rec_valid, auc_valid = dx_metrics_valid.average()
                ssim_valid, psnr_valid, mse_valid = recon_metrics_valid.average()

            # ###################################################
                     
            # # Save model
            # if min_loss_dx > loss_diag_valid:
            #     min_loss = loss_diag_valid
            #     if os.path.exists(filename):
            #         os.remove(filename)    
                
            #     filename = "./checkpoint/{}/{}/Fold_{}_Acc_{:.04f}_Pre_{:.04f}_Rec_{:.04f}_mauc_{:.04f}_ssim_{:.04f}_psnr_{:.04f}_mse_{:.04f}.pt"\
            #                 .format(model_config.MODEL_NAME,
            #                 model_config.DATA_TYPE,
            #                 fold,
            #                 acc_valid, pre_valid, rec_valid, auc_valid,
            #                 ssim_valid, psnr_valid, mse_valid)
            #     if not os.path.exists("./checkpoint/{}/{}".format(model_config.MODEL_NAME, model_config.DATA_TYPE)):
            #         os.makedirs("./checkpoint/{}/{}".format(model_config.MODEL_NAME, model_config.DATA_TYPE))
            #     torch.save(model, filename)
            #     print("Saving model: ", filename)
            # print("Traing")
            # print("Epoch: {} | Loss_train: {:.04f} | Loss_KL: {:.04f}    | Loss_Recon: {:.04f} | Loss_DX {:.04f}".format(epoch, loss_total_train, loss_kl_train, loss_rec_train, loss_diag_train))
            # print("Epoch: {} | Acc_train: {:.04f}  | Pre_train: {:.04f}  | Rec_train: {:.04f}  | mAUC_train {:.04f}".format(epoch, acc_train, pre_train, rec_train, auc_train))
            # print("Epoch: {} | SSIM_train: {:.04f} | PSNR_train: {:.04f} | MSE_train: {:.04f}".format(epoch, ssim_train, psnr_train, mse_train))
            # print("Valid")
            # print("Epoch: {} | Loss_valid: {:.04f} | Loss_KL: {:.04f}    | Loss_Recon: {:.04f} | Loss_DX {:.04f}".format(epoch, loss_total_valid, loss_kl_valid, loss_rec_valid, loss_diag_valid))
            # print("Epoch: {} | Acc_valid: {:.04f}  | Pre_valid: {:.04f}  | Rec_valid: {:.04f}  | mAUC_valid {:.04f}".format(epoch, acc_valid, pre_valid, rec_valid, auc_valid))
            # print("Epoch: {} | SSIM_valid: {:.04f} | PSNR_valid: {:.04f} | MSE_valid: {:.04f}".format(epoch, ssim_valid, psnr_valid, mse_valid))

            # # Save log
            # data_training.append([epoch, loss_total_train, loss_kl_train, loss_rec_train, loss_diag_train, acc_train, pre_train, rec_train, auc_train,  ssim_train, psnr_train, mse_train,
            #                             loss_total_valid, loss_kl_valid, loss_rec_valid, loss_diag_valid, acc_valid, pre_valid, rec_valid, auc_valid, ssim_valid, psnr_valid, mse_valid])
            # log_data_frame = np.asarray(data_training)
            # log_data_frame = pd.DataFrame(log_data_frame, columns=[['Epoch', 'Loss_train', 'Loss_KL_train', 'Loss_Recon_train', 'Loss_DX_train', 
            #                                                         'Acc_train', 'Pre_train', 'Rec_train', 'mAUC_train', 'SSIM_train', 'PSNR_train', "MSE_train",
            #                                                         "Loss_valid", "Loss_KL_valid", "Loss_Recon_valid", "Loss_DX_valid",
            #                                                         'Acc_valid', 'Pre_valid', 'Rec_valid', 'mAUC_valid', 'SSIM_valid', 'PSNR_valid', "MSE_valid"]])
            
            # log_data_frame.to_csv("./checkpoint/{}/{}/log_train_fold_{}.csv".format(model_config.MODEL_NAME, model_config.DATA_TYPE, fold), index=False)


