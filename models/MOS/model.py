import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .radar_transformer import Self_attention_block, Cross_attention_block
import torch.nn.functional as F
import numpy as np



class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = Self_attention_block(32, cfg.model.transformer_dim, nneighbor, knn=False)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            knn = True if i > 0 else False
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(Self_attention_block(channel, cfg.model.transformer_dim, nneighbor, knn))
        self.cross_attention = Cross_attention_block(channel, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
    
    def forward(self, x1, x2):
        xyz1 = x1[..., :3]
        xyz2 = x2[..., :3]
        points1 = self.transformer1(xyz1, self.fc1(x1))[0]
        points2 = self.transformer1(xyz2, self.fc1(x2))[0]
        xyz_and_feats1 = [(xyz1, points1)]
        xyz_and_feats2 = [(xyz2, points2)]
        for i in range(self.nblocks):
            xyz1, points1 = self.transition_downs[i](xyz1, points1)
            xyz2, points2 = self.transition_downs[i](xyz2, points2)
            points1 = self.transformers[i](xyz1, points1)[0]
            points2 = self.transformers[i](xyz2, points2)[0]
            xyz_and_feats1.append((xyz1, points1))
            xyz_and_feats2.append((xyz2, points2))
        points1 = self.cross_attention(xyz1, points1, xyz2, points2)[0]
        
        return points1, xyz_and_feats1, xyz_and_feats2


class MOS_module(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = Self_attention_block(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor, knn=True)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            knn = True if i > 0 else False
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(Self_attention_block(channel, cfg.model.transformer_dim, nneighbor, knn=knn))

        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x1, x2):
        # encoder module
        points, xyz_and_feats1, _ = self.encoder(x1, x2)
        
        # decoder module
        xyz = xyz_and_feats1[-1][0]
        points1 = self.transformer2(xyz, self.fc2(points))[0]
        for i in range(self.nblocks):
            points1 = self.transition_ups[i](xyz, points1, xyz_and_feats1[- i - 2][0], xyz_and_feats1[- i - 2][1])
            xyz = xyz_and_feats1[- i - 2][0]
            points1 = self.transformers[i](xyz, points1)[0]
        seg = self.mlp(points1)

        return seg