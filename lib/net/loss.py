#=========================================
# Written by Yude Wang
#=========================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class BAC_loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(BAC_loss, self).__init__()
        self.eps = eps

    def forward(self, features, edge_map):
        B, C, H, W = features.size()
        subregion_representations = []
        for subregion_idx in range(6): 
            mask = edge_map[:, subregion_idx:subregion_idx + 1, :, :] 
            mask_sum = mask.sum(dim=(2, 3), keepdim=True) + self.eps  
            subregion_feat = (features * mask).sum(dim=(2, 3), keepdim=True) / mask_sum 
            subregion_representations.append(subregion_feat)
        subregion_representations = torch.cat(subregion_representations, dim=2).squeeze(-1)
        bg_edge, bg_inner, pz_edge, pz_inner, cg_edge, cg_inner = torch.chunk(subregion_representations, 6, dim=2)
        intra_bg = torch.bmm(bg_edge, bg_inner.permute(0, 2, 1))  
        intra_pz = torch.bmm(pz_edge, pz_inner.permute(0, 2, 1))  
        intra_cg = torch.bmm(cg_edge, cg_inner.permute(0, 2, 1))
        identity_matrix = torch.eye(C, device=features.device).expand(B, C, C)
        intra_loss_bg = F.mse_loss(intra_bg, identity_matrix)
        intra_loss_pz = F.mse_loss(intra_pz, identity_matrix)
        intra_loss_cg = F.mse_loss(intra_cg, identity_matrix)
        inter_bg_pz = torch.bmm(bg_inner, pz_inner.permute(0, 2, 1))  
        inter_pz_cg = torch.bmm(pz_inner, cg_inner.permute(0, 2, 1)) 
        inter_cg_bg = torch.bmm(cg_inner, bg_inner.permute(0, 2, 1)) 
        zero_matrix = torch.zeros_like(identity_matrix)
        inter_loss_bg_pz = F.mse_loss(inter_bg_pz * identity_matrix, zero_matrix)
        inter_loss_pz_cg = F.mse_loss(inter_pz_cg * identity_matrix, zero_matrix)
        inter_loss_cg_bg = F.mse_loss(inter_cg_bg * identity_matrix, zero_matrix)
        total_loss = (intra_loss_bg + intra_loss_pz + intra_loss_cg + inter_loss_bg_pz + inter_loss_pz_cg + inter_loss_cg_bg)/3.
        return total_loss
