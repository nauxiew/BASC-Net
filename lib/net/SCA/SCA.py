import torch
from torch import nn
from torch.nn import functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from kmeans_pytorch import kmeans
import math


class SCA_attention(nn.Module):
    def __init__(self, in_channels, out_channels, K): 
        super(SCA_attention, self).__init__()
        self.num_subregion = 2*3
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.K = K
        centers = torch.Tensor(1, self.out_channels, self.num_subregion*self.K)
        centers.normal_(0, math.sqrt(2. / (self.num_subregion*self.K)))
        centers = self._l2norm(centers, dim=1)
        self.register_buffer('centers', q_centers)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            SynchronizedBatchNorm2d(self.in_channels, momentum = 0.0003),
            nn.ReLU(inplace=True),
            )        
        

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0),
            SynchronizedBatchNorm2d(self.in_channels, momentum = 0.0003),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def GetK_feature_centroids(self, features, edge, K, initial_centers=[]):
        b,c,h,w = features.size()
        assert edge.size()[1] == h
        assert edge.size()[2] == w
        features_T = features.permute(1,0,2,3) 
        selected_features = features_T[:,(edge==1).squeeze()]
        selected_features = selected_features.T 
        device = torch.device('cuda:0')
        cluster_pred, centers = kmeans(X=selected_features.detach(), num_clusters=K, distance='cosine', device=device, cluster_centers=initial_centers)
        cluster_centers = torch.zeros_like(centers)
        del centers
        for index in range(K):
            selected = torch.nonzero(cluster_pred == index).squeeze().to(device)
            selected = torch.index_select(selected_features, 0, selected)
            if selected.shape[0] == 0:
                selected = selected_features[torch.randint(len(selected_features), (1,))]
            cluster_centers[index] = selected.mean(dim=0)
        return cluster_centers


    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    def aux_loss(self, label, pred):
        loss = 0.
        b,n,h,w = label.size()
        assert pred.size()[1] == n*self.K     
        for i in range(n):
            edge_i = label[:,i,:,:]
            pred_i = torch.sum(pred[:, i*self.K:(i+1)*self.K, :, :], dim=1)
            intersection = torch.sum(edge_i*pred_i, dim=(1,2))
            union = torch.sum(edge_i+pred_i, dim=(1,2))
            loss_i = 2*(intersection+1e-6)/(1e-6+union)
            loss += (1.-torch.mean(loss_i))
        loss = loss/n
        return loss

      
    def forward(self, x, edge=None, period=None, scale_factor=None):
        if scale_factor is not None and period is not None:
            downsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
            edge = downsample(edge)
        x1 = self.conv1(x)
        batch_size, c, h, w = x1.size()
        if period is not None:
            bases = torch.zeros_like(self.centers)
            for i in range(self.num_subregion):
                y = edge[:,i,:,:]
                initial_centers = self.centers[0, :, i*self.K:(i+1)*self.K]
                bases_kmeans = self.GetK_feature_centroids(features=x1, edge=y, K=self.K, initial_centers=initial_centers.T)
                bases_kmeans = self._l2norm(bases_kmeans, dim=1)
                bases[0,:,i*self.K:(i+1)*self.K] = bases_kmeans.T
        else:
            bases = self.centers
        bases_K = bases.expand(batch_size, -1, -1)
        x1 = x1.view(batch_size, c, h*w)
        x1_t = x.permute(0, 2, 1)
        att_map = torch.bmm(x1_t, bases_K)
        att_map = F.softmax(att_map, dim=2)
        att_map = att_map / (1e-6 + att_map.sum(dim=1, keepdim=True))
        out_att = att_map.permute(0,2,1).view(batch_size, -1, h, w)
        x_re = torch.einsum('bnk, bck->bnc', att_map, bases_K) 
        x_re = x_re.permute(0, 2, 1).contiguous()
        x_re = x_re.view(batch_size, -1, h,w)
        x_re = self.conv2(x_re)
        output = x_re + x1
        output = self.conv_bn_dropout(output)
        if period is not None:
            aux_loss = self.aux_loss(label=edge, pred=out_att)
            with torch.no_grad():
                self.centers = self.centers*0.95 + bases*0.05
            return output, aux_loss
        else:
            return output, 0.
        
        
       