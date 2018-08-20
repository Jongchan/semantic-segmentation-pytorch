import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, version=None):
        super(ChannelGate, self).__init__()
        self.version = version
        self.se = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        if self.version == 'v8':
            # LSE pool only
            lse_pool = logsumexp_2d(x)
            scale = F.sigmoid( self.se( lse_pool )).unsqueeze(2).unsqueeze(3).expand_as(x)
            return x * scale

        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)) )
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)) )
        if self.version == 'v7':
            return (self.se( avg_pool ) + self.se( max_pool )).unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            scale = F.sigmoid( (self.se( avg_pool ) + self.se( max_pool )).unsqueeze(2).unsqueeze(3).expand_as(x) )
            return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, size_reduction_ratio=2, version=None):
        super(SpatialGate, self).__init__()
        self.version = version
        if version in ['v1', 'v2']:
            if version == 'v1':
                kernel_size = 7
            elif version == 'v2':
                kernel_size = 3
            self.compress = BasicConv(gate_channels, 1, 1, bn=False)
            self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        elif version in ['v3', 'v4', 'v7', 'v8']:
            if version in ['v3', 'v7', 'v8'] :
                kernel_size = 7
            elif version in ['v4']:
                kernel_size = 3
            self.compress = ChannelPool()
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        elif version == 'v5':
            self.compress = BasicConv(gate_channels, gate_channels // reduction_ratio, 1, bn=False)
            gate_channels = gate_channels // reduction_ratio
            self.spatial = nn.Sequential(
                BasicConv(gate_channels, gate_channels, 3, stride=1, padding=1, groups=gate_channels, bn=False),
                BasicConv(gate_channels, gate_channels, 1, bn=False)
                )
            self.expand = BasicConv(gate_channels, 1, 1, relu=False)
        elif version == 'v6':
            self.size_reduction_ratio = size_reduction_ratio
            self.compress = BasicConv(gate_channels, gate_channels // reduction_ratio, 1, bn=False)
            gate_channels = gate_channels // reduction_ratio
            self.spatial = nn.Sequential(
                BasicConv(gate_channels, gate_channels, 3, stride=1, padding=1, groups=gate_channels, bn=False),
                BasicConv(gate_channels, gate_channels, 1, bn=False)
                )
            self.expand = BasicConv(gate_channels, gate_channels*reduction_ratio, 1, relu=False)
    def forward(self, x):
        if self.version in ['v1', 'v2']:
            x_compress = self.compress(x)     
            x_out = self.spatial(x_compress)
            scale = F.sigmoid(x_out) # broadcasting
        elif self.version in ['v3', 'v4', 'v7', 'v8']:
            x_compress = self.compress(x)
            x_out = self.spatial(x_compress)
            if not self.version == 'v7':
                scale = F.sigmoid(x_out) # broadcasting
        elif self.version == 'v5':
            x_compress = self.compress(x)
            x_ = self.spatial(x_compress)
            x_out = self.expand(x_) 
            scale = F.sigmoid(x_out) # broadcasting
        elif self.version == 'v6':
            x_pool = F.adaptive_max_pool2d( x, x.size(2) // self.size_reduction_ratio )
            x_ = self.compress(x_pool)
            x_ = self.spatial(x_)
            x_ = self.expand(x_)
            x_up = F.upsample( x_, x.size()[2:], mode='bilinear')
            scale = F.sigmoid(x_up) # broadcasting
        
        if self.version == 'v7':
            return x_out.expand_as(x)
        else: 
            return x * scale

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, size_reduction_ratio=2, reduction_ratio=16, version=None):
        super(AttentionGate, self).__init__()
        assert version in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']
        self.version = version
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, version) if version != 'v6' else None 
        self.SpatialGate = SpatialGate(gate_channels, reduction_ratio, size_reduction_ratio, version)
    def forward(self, x):
        if self.version in ['v1', 'v2', 'v3', 'v4', 'v5', 'v8']:
            x_out = self.ChannelGate(x)
            x_out = self.SpatialGate(x_out)
        elif self.version in ['v6']:
            x_out = self.SpatialGate(x)
        elif self.version == 'v7':
            x_c = self.ChannelGate(x)
            x_s = self.SpatialGate(x)
            scale = F.sigmoid(x_c + x_s)
            x_out = x * scale
        return x_out



