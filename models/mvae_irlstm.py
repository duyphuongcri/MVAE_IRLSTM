import torch
import torch.nn as nn
import numpy as np 
import math 
from config import model_config
from models.mvae import MVAE

class Linear_diagonal_weight(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(Linear_diagonal_weight,self).__init__()
        self.device=device
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.diagonal_mask = torch.eye(output_size, input_size)
    def forward(self, input):
        # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # if use_cuda:
        self.diagonal_mask = self.diagonal_mask.to(self.device)
        return  torch.mm(input, self.weight * self.diagonal_mask) + self.bias 

class Linear_diagonal_weight_z(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(Linear_diagonal_weight_z,self).__init__()
        self.device=device
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        self.diagonal_mask = torch.eye(output_size, input_size)
        self.diagonal_mask = 1 - self.diagonal_mask

    def forward(self, input):
        # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # if use_cuda:
        self.diagonal_mask = self.diagonal_mask.to(self.device)  
        return torch.mm(input, self.weight * self.diagonal_mask) + self.bias 


class MVAE_IRLSTM(torch.nn.Module):
    def __init__(self, config, i_ratio=0.9, h_ratio=0.9):
        super(MVAE_IRLSTM, self).__init__()
        self.config = config
        self.i_ratio=i_ratio # dropout ratio of input = 1 - i_ratio
        self.h_ratio=h_ratio # dropout ratio of hidden state = 1 - h_ratio
        self.pi = torch.Tensor([np.pi])

        ############ MVAE ###########
        self.model_mvae = MVAE(config)

        ####### LSTM #######
        input_size = self.config.INPUT_SIZE
        hidden_size = self.config.HIDDEN_SIZE

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Decay weights
        self.w_dg_x = Linear_diagonal_weight(input_size, input_size, config.DEVICE)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias=True)

        # x weights
        self.w_x = torch.nn.Linear(hidden_size, input_size, bias=True)
        self.w_xz = Linear_diagonal_weight_z(input_size, input_size, config.DEVICE)

        #beta weight
        self.w_b_dg = torch.nn.Linear(input_size*2, input_size, bias=True)

        # i weights
        self.w_ui = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hi = torch.nn.Linear(hidden_size, hidden_size, bias=True)        

        # c weights
        self.w_uc = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hc = torch.nn.Linear(hidden_size, hidden_size, bias=True) 

        # o weights
        self.w_uo = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_ho = torch.nn.Linear(hidden_size, hidden_size, bias=True) 

        # r beta weights
        self.w_br = torch.nn.Linear(input_size, input_size, bias=True)

        # f weights
        self.w_uf = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hf = torch.nn.Linear(hidden_size, hidden_size, bias=True)

        # r weights
        self.w_ur = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=True)   

        # PREDICTION MODULE
        self.w_dxh = torch.nn.Linear(hidden_size, self.config.NUM_CLASSES, bias=True)
        self.w_gh = torch.nn.Linear(hidden_size, self.config.OUTPUT_SIZE, bias=True)
        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def imputation_module(self, x_t, m_t, delta_t, x_t_hat):
        gamma_x = torch.exp(-1* self.relu(self.w_dg_x(delta_t))) 
        
        x_t[x_t != x_t] = 0
        x_t_c = m_t*x_t + (1-m_t)*x_t_hat 
        z_t_hat = self.w_xz(x_t_c)
        beta = self.sigmoid(self.w_b_dg(torch.cat((gamma_x,m_t), dim=1)))
        u_t_hat = beta*z_t_hat + (1-beta)*x_t_hat  
        u_t = m_t*x_t + (1-m_t)*u_t_hat    
        return u_t, u_t_hat

    def encoder_module(self, delta_t, u_t, h_t, c_t):
        # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # if use_cuda:
        self.pi = self.pi.to(self.config.DEVICE)

        gamma_h = torch.exp(-1* self.relu( self.w_dg_h(delta_t)))
        
        h_t_hat = gamma_h * h_t
        f_t = self.sigmoid(self.w_uf(u_t) + self.w_hf(h_t_hat))
        r_t = self.sigmoid(self.w_ur(u_t) + self.w_hr(h_t_hat))
        g_t = f_t + torch.sin(f_t * self.pi) * torch.cos(r_t * self.pi) / self.pi
        c_t_hat = self.tanh(self.w_uc(u_t) + self.w_hc(h_t_hat))
        c_t = g_t*c_t + (1-g_t)*c_t_hat
        o_t = self.sigmoid(self.w_uo(u_t) + self.w_ho(h_t_hat))
        h_t = o_t * self.tanh(c_t)  
        return h_t, c_t

    def prediction_module(self, h_t):
        x_t_plus_1_hat = self.w_gh(h_t)
        x_dx_t_plus_1_hat = self.w_dxh(h_t)
        return x_t_plus_1_hat, x_dx_t_plus_1_hat

    def forward(self, data, mask, delta):
        """
        data_i :  B x L x C x D x H x W
        mask_i:   B x L
        delta_i:  B x L
        data_dx:  B x L x 3
        mask_i:   B x L x 3
        delta_i:  B x L x 3
        """
        # device = next(self.parameters()).device
        data_cs, data_dg, data_i, data_dx = data
        mask_cs, mask_dg, mask_i, mask_dx = mask
        delta_cs, delta_dg, delta_i, delta_dx =delta

        B = data_i.shape[0]
        L = data_i.shape[1]
        ############# 
        data_dx_hat  = [] #torch.empty([B, L-1, self.config.NUM_CLASSES])
        data_i_hat = [] #torch.empty(data_i.shape)
        data_cs_hat = [] #torch.empty([B, L-1, len(self.config.FEATURES_COGNITIVE_SCORE)])
        multi_mu_logvar = [] #torch.empty([B, L, int(len(self.config.MODALITY)*2+2), self.config.Z_DIM])

        ### Initialize h_t and c_t in LSTM
        h_t = torch.zeros((B, self.config.HIDDEN_SIZE), dtype=torch.float, device=self.config.DEVICE)
        c_t = torch.zeros((B, self.config.HIDDEN_SIZE), dtype=torch.float, device=self.config.DEVICE)
        for tps in range(L):
            x_i_t, m_i_t, delta_i_t = data_i[:, tps], mask_i[:, tps], delta_i[:, tps]
            x_dx_t, m_dx_t, delta_dx_t = data_dx[:, tps], mask_dx[:, tps], delta_dx[:, tps]
            # Extract image features
            out_i_t, mu, logvar, mri_mu, mri_logvar, pet_mu, pet_logvar, dti_mu, dti_logvar  = self.model_mvae(x_i_t, m_i_t)
            z_t = self.model_mvae.reparameterize(mu, logvar) # B x Z_DIM
            # Convert modality mask to image feature mask
            m_z_t = torch.any(m_i_t==True, dim=-1).reshape(-1, 1).repeat(1, z_t.shape[1])

            x_t = torch.cat((z_t, x_dx_t), dim=1)
            m_t = torch.cat((m_z_t, m_dx_t), dim=1).float()
            delta_t = torch.cat((delta_i_t, delta_dx_t), dim=1)
            if "DG" in self.config.DATA_TYPE:
                x_t = torch.cat((data_dg[:, tps], x_t), dim=1)
                m_t = torch.cat((mask_dg[:, tps], m_t), dim=1).float()
                delta_t = torch.cat((delta_dg[:, tps], delta_t), dim=1)
            if "CS" in self.config.DATA_TYPE:
                x_t = torch.cat((data_cs[:, tps], x_t), dim=1)
                m_t = torch.cat((mask_cs[:, tps], m_t), dim=1).float()
                delta_t = torch.cat((delta_cs[:, tps], delta_t), dim=1)

            # Imputation Module
            if tps == 0:
                u_t = x_t
            else:
                u_t, u_t_hat = self.imputation_module(x_t, m_t, delta_t, x_t_hat)
            # Encoder Module
            h_t, c_t = self.encoder_module(delta_t, u_t, h_t, c_t)
            # Prediction Module
            x_t_plus_1_hat, x_dx_t_plus_1_hat = self.prediction_module(h_t)
            x_t_hat = torch.cat((x_t_plus_1_hat, self.softmax(x_dx_t_plus_1_hat)), dim=1)

            if tps < L-1:
                data_dx_hat.append(x_dx_t_plus_1_hat)
                if "CS" in self.config.DATA_TYPE:
                    data_cs_hat.append(u_t_hat[:, :len(self.config.FEATURES_COGNITIVE_SCORE)])

            data_i_hat.append(out_i_t) 

            multi_mu_logvar.append(torch.stack((mu, logvar, mri_mu, mri_logvar, pet_mu, pet_logvar, dti_mu, dti_logvar), dim=1))

        data_i_hat      = torch.stack(data_i_hat, dim=1)
        data_dx_hat     = torch.stack(data_dx_hat, dim=1)
        multi_mu_logvar = torch.stack(multi_mu_logvar, dim=1)
        if "CS" in self.config.DATA_TYPE:
            data_cs_hat     = torch.stack(data_cs_hat, dim=1)
        return data_i_hat, multi_mu_logvar, data_dx_hat, data_cs_hat
    
    def predict(self, data, mask, delta):
        """
        data_i :  B x L x C x D x H x W
        mask_i:   B x L
        delta_i:  B x L
        data_dx:  B x L x 3
        mask_i:   B x L x 3
        delta_i:  B x L x 3
        """
        # device = next(self.parameters()).device
        data_cs, data_dg, data_i, data_dx = data
        mask_cs, mask_dg, mask_i, mask_dx = mask
        delta_cs, delta_dg, delta_i, delta_dx =delta

        B = data_i.shape[0]
        L = data_i.shape[1]
        ############# 
        data_dx_hat  = [] #torch.empty([B, L-1, self.config.NUM_CLASSES])
        data_i_hat = [] #torch.empty(data_i.shape)
        data_cs_hat = [] #torch.empty([B, L-1, len(self.config.FEATURES_COGNITIVE_SCORE)])
        multi_mu_logvar = [] #torch.empty([B, L, int(len(self.config.MODALITY)*2+2), self.config.Z_DIM])

        ### Initialize h_t and c_t in LSTM
        h_t = torch.zeros((B, self.config.HIDDEN_SIZE), dtype=torch.float, device=self.config.DEVICE)
        c_t = torch.zeros((B, self.config.HIDDEN_SIZE), dtype=torch.float, device=self.config.DEVICE)
        for tps in range(L):
            x_i_t, m_i_t, delta_i_t = data_i[:, tps], mask_i[:, tps], delta_i[:, tps]
            x_dx_t, m_dx_t, delta_dx_t = data_dx[:, tps], mask_dx[:, tps], delta_dx[:, tps]
            # Extract image features
            out_i_t, mu, logvar, mri_mu, mri_logvar, pet_mu, pet_logvar, dti_mu, dti_logvar  = self.model_mvae(x_i_t, m_i_t)
            z_t = self.model_mvae.reparameterize(mu, logvar) # B x Z_DIM
            # Convert modality mask to image feature mask
            m_z_t = torch.any(m_i_t==True, dim=-1).reshape(-1, 1).repeat(1, z_t.shape[1])

            x_t = torch.cat((z_t, x_dx_t), dim=1)
            m_t = torch.cat((m_z_t, m_dx_t), dim=1).float()
            delta_t = torch.cat((delta_i_t, delta_dx_t), dim=1)
            if "DG" in self.config.DATA_TYPE:
                x_t = torch.cat((data_dg[:, tps], x_t), dim=1)
                m_t = torch.cat((mask_dg[:, tps], m_t), dim=1).float()
                delta_t = torch.cat((delta_dg[:, tps], delta_t), dim=1)
            if "CS" in self.config.DATA_TYPE:
                x_t = torch.cat((data_cs[:, tps], x_t), dim=1)
                m_t = torch.cat((mask_cs[:, tps], m_t), dim=1).float()
                delta_t = torch.cat((delta_cs[:, tps], delta_t), dim=1)

            # Imputation Module
            if tps == 0:
                u_t = x_t
            else:
                u_t, u_t_hat = self.imputation_module(x_t, m_t, delta_t, x_t_hat)
            # Encoder Module
            h_t, c_t = self.encoder_module(delta_t, u_t, h_t, c_t)
            # Prediction Module
            x_t_plus_1_hat, x_dx_t_plus_1_hat = self.prediction_module(h_t)
            x_t_hat = torch.cat((x_t_plus_1_hat, self.softmax(x_dx_t_plus_1_hat)), dim=1)

            if tps < L-1:
                data_dx_hat.append(x_dx_t_plus_1_hat)
                if "CS" in self.config.DATA_TYPE:
                    data_cs_hat.append(u_t_hat[:, :len(self.config.FEATURES_COGNITIVE_SCORE)])

            data_i_hat.append(out_i_t) 

            multi_mu_logvar.append(torch.stack((mu, logvar, mri_mu, mri_logvar, pet_mu, pet_logvar, dti_mu, dti_logvar), dim=1))

        data_i_hat      = torch.stack(data_i_hat, dim=1)
        data_dx_hat     = torch.stack(data_dx_hat, dim=1)
        multi_mu_logvar = torch.stack(multi_mu_logvar, dim=1)
        if "CS" in self.config.DATA_TYPE:
            data_cs_hat     = torch.stack(data_cs_hat, dim=1)
        return data_i_hat, multi_mu_logvar, data_dx_hat, data_cs_hat
    

# if __name__=="__main__":
#     import sys
#     sys.path.append("../") 
#     from config import model_config
#     device = torch.device("cuda:0")
#     L = model_config.NUM_PRED_YEAR
#     D_cs = len(model_config.FEATURES_COGNITIVE_SCORE)
#     D_dg = len(model_config.FEATURES_DEMOGRAPHIC)

#     data = (torch.zeros(N, L, D_cs).to(device), torch.zeros(N, L, D_dg).to(device), torch.zeros(N, L, 4, 80, 96, 80).to(device), torch.zeros(N, L, 3).to(device))
#     mask = (torch.ones((N, L, D_cs), dtype=torch.bool).to(device), \
#             torch.ones((N, L,  D_dg), dtype=torch.bool).to(device),\
#             torch.ones((N, L, 3), dtype=torch.bool).to(device), \
#             torch.ones((N, L, 3 ), dtype=torch.bool).to(device))
#     delta = (torch.zeros((N, L, D_cs)).to(device), torch.zeros((N, L, D_dg)).to(device), torch.zeros((N, L, 256)).to(device), torch.zeros((N, L, 3)).to(device))
    
#     model = MVAE_IRLSTM(model_config).to(device)
#     data_i_hat, data_y_hat = model(data, mask, delta)
#     print(data_i_hat.shape, data_y_hat.shape)