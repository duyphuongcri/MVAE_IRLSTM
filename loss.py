import torch.nn as nn
import torch.nn.functional as F
import torch

def dxLoss(logits, target, mask):
    """
    logits: B x L x 3
    target: B x L x 3
    mask  : B x L x 3
    """
    return torch.nn.functional.cross_entropy(logits[mask], target[mask], reduction='mean')

def KLDivergence(z_mu_logvar, mask):
    """
    z_mu_logvar :  B x L x 8 x Z_DIM
    mask        :  B x L x 3
    """
    m_z   = torch.any(mask==True, dim=-1)
    m_mri = mask[:,:, 0]
    m_pet = mask[:,:, 1]
    m_dti = mask[:,:, 2]

    mu, logvar = z_mu_logvar[:, :, 0][m_z],   z_mu_logvar[:, :, 1][m_z]
    loss_kl   = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1) / mu.shape[0]
    if m_mri.sum() > 0:
        mri_mu, mri_logvar = z_mu_logvar[:, :, 2][m_mri], z_mu_logvar[:, :, 3][m_mri]
        loss_kl +=  0.5 * torch.sum(mri_mu**2 + mri_logvar.exp() - mri_logvar - 1) / mri_mu.shape[0]
    if m_pet.sum() > 0:
        pet_mu, pet_logvar = z_mu_logvar[:, :, 4][m_pet], z_mu_logvar[:, :, 5][m_pet]
        loss_kl += 0.5 * torch.sum(pet_mu**2 + pet_logvar.exp() - pet_logvar - 1) / pet_mu.shape[0]
    if m_dti.sum() > 0:
        dti_mu, dti_logvar = z_mu_logvar[:, :, 6][m_dti], z_mu_logvar[:, :, 7][m_dti]
        loss_kl += 0.5 * torch.sum(dti_mu**2 + dti_logvar.exp() - dti_logvar - 1) / dti_mu.shape[0]
    return loss_kl
    

def recontructLoss(logits, target, mask): 
    """
    y    : B x L x C x D x H x W
    mask : B x L x 3
    MRI  : y[:, :, 0]
    PET  : y[:, :, 1]
    DTI  : y[:, :, 2:]
    """
    mask = torch.cat((mask, mask[:, :, 2:]), dim=2) ## duplicate mask of DTI due to its multiple channels (2 channels: FA and MD)
    
    return ((logits - target)[mask]**2).mean()

def imputationLoss(logits, target, mask): 
    """
    y    : B x L x D_cs
    mask : B x L x D_cs
    """    
    return (abs(logits - target)[mask]).mean()
    

############### CONSTRASTIVE LOSS #####################

def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
    
class MMS_Loss(nn.Module):
    def __init__(self, margin=0.001):
        super(MMS_Loss, self).__init__()
        self.margin = margin

    def forward(self, S, ):
        deltas = self.margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss


class CombinatorialLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05,
                 tv_weight=0, ta_weight=0, va_weight=0,
                 t_va_weight=0, v_ta_weight=0, a_tv_weight=0):
        super().__init__()

        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()

        self.tv_weight = tv_weight
        self.ta_weight = ta_weight
        self.va_weight = va_weight
        self.t_va_weight = t_va_weight
        self.v_ta_weight = v_ta_weight
        self.a_tv_weight = a_tv_weight

    def forward(self, input_data):

        nonempty = {}
        nonempty['tv'] = input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']
        nonempty['ta'] = input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
        nonempty['va'] = input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']

        nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (
                    input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])

        loss_sum = 0
        weight_sum = 0
        loss_info = {}

        for name, embed_name1, embed_name2, weight in [
            ('tv', 'text_embed', 'video_embed', self.tv_weight),
            ('ta', 'text_embed', 'audio_embed', self.ta_weight),
            ('va', 'video_embed', 'audio_embed', self.va_weight),
            ('t_va', 'text_embed', 'va_embed', self.t_va_weight),
            ('v_ta', 'video_embed', 'ta_embed', self.v_ta_weight),
            ('a_tv', 'audio_embed', 'tv_embed', self.a_tv_weight),
        ]:
            if (embed_name1 in input_data) and (embed_name2 in input_data) and (weight != 0):
                nonempty_mask = nonempty[name]
                embed1 = input_data[embed_name1][nonempty_mask]
                embed2 = input_data[embed_name2][nonempty_mask]

                loss = self.contrastive_loss(sim_matrix(embed1, embed2))
                loss_info[name] = loss.item()
                loss_sum += weight * loss
                weight_sum += weight

        final_loss = loss_sum / weight_sum
        loss_info['Retrieval'] = final_loss.item()
        return final_loss, loss_info