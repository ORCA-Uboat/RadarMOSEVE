import torch
import torch.nn as nn
from pointnet_util import PointNetSetAbstraction
from .radar_transformer import Self_attention_block, Cross_attention_block
from pointnet_util import index_points, square_distance
import torch.nn.functional as F
import numpy as np

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.input_dim
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


class EVE_module(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        nblocks, output_dim = cfg.model.nblocks, cfg.output_dim
        self.fc_speed2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU()
        )
        self.fc_speed3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )


    def forward(self, x1, x2):
        points, _, _ = self.backbone(x1, x2)
        # speed
        res1 = self.fc_speed2(points.mean(1))
        speed = self.fc_speed3(res1)
        return speed


class Doppler_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, v, target, point_set, seg):
        yaw_p = torch.rad2deg(torch.atan2(point_set[:, :, 1], point_set[:, :, 0]))
        deg_m1 = torch.deg2rad(90-yaw_p)
        vr_speed= torch.multiply(v[:, 0].view(-1, 1).repeat(1, deg_m1.size()[1]), torch.cos(deg_m1)) 
        vr = point_set[:, :, 3]
        vr = abs(vr + vr_speed)
        vr = vr[seg==0]
        loss_points = torch.mean(vr)
        loss_eve = self.mse(v, target)
        loss = loss_eve + loss_points
        return loss, loss_eve, loss_points