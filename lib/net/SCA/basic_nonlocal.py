import torch
from torch import nn
from torch.nn import functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from kmeans_pytorch import kmeans
import math


class Non_local_module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels): 
        super(Non_local_module, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, momentum = 0.0003),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, momentum = 0.0003),
            nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
      
    def forward(self, x, period=None):
        batch_size, c, h, w = x.size()
        query = self.f_query(x)
        query = query.view(batch_size, self.key_channels, -1) 
        key = self.f_key(x).view(batch_size, self.key_channels, -1) 
        key = key.permute(0,2,1) 
        value = self.f_value(x) 
        value = value.view(batch_size, self.value_channels, -1) 
        value = value.permute(0,2,1) 

        att_map = torch.matmul(key, query)
        att_map = (self.key_channels ** -.5) * att_map
        att_map = F.softmax(att_map, dim=2)

        context = torch.matmul(att_map, value) 
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        out = context + x
        return out

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class AS_block(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, psp_size=(1,3,6,8)):
        super(AS_block, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, momentum = 0.0003),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, momentum = 0.0003),
            nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.psp(self.f_value(x))

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        out = context + x
        return out
        
class EMAU(nn.Module):

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))   
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            SynchronizedBatchNorm2d(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x, period=None):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               
        mu = self.mu.repeat(b, 1, 1)       
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)   
                z = torch.bmm(x_t, mu)     
                z = F.softmax(z, dim=2)    
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       
                mu = self._l2norm(mu, dim=1)
        z_t = z.permute(0, 2, 1)           
        x = mu.matmul(z_t)                 
        x = x.view(b, c, h, w)             
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)
        if period is not None:
            with torch.no_grad():
                mu = mu.mean(dim=0, keepdim=True)
                self.mu = self.mu*0.9+mu*0.1
        return x

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

