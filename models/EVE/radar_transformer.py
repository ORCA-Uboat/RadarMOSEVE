from pointnet_util import index_points, square_distance, query_ball_point
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Self_attention_block(nn.Module):
    def __init__(self, d_points, d_model, k, knn=True) -> None:
        super().__init__()
        DROP_RATIO = 0.2
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.Dropout(DROP_RATIO)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(d_model, d_points),
            nn.Dropout(DROP_RATIO)
        ) 
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(DROP_RATIO),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(DROP_RATIO)
        )
        self.w_qs = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(DROP_RATIO)
        ) 
        self.w_ks = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(DROP_RATIO)
        ) 
        self.w_vs = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(DROP_RATIO)
        ) 
        self.k = k
        self.sample_dis = 2
        self.knn = knn
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        b, n, _ = xyz.size()
        if self.knn:
            dists = square_distance(xyz, xyz)
            # knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
            knn_idx = dists.argsort()[:, :, :self.k*2:2]  # b x n x k
        else:
            knn_idx = query_ball_point(self.sample_dis, self.k, xyz, xyz)
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

class Cross_attention_block(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.Dropout(0.2)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(d_model, d_points),
            nn.Dropout(0.2)
        ) 
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.2)
        )
        self.w_qs = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(0.2)
        ) 
        self.w_ks = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(0.2)
        ) 
        self.w_vs = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(0.2)
        ) 
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz1, features1, xyz2, features2):
        '''
        1为当前帧，2为上一帧
        '''
        dists = square_distance(xyz1, xyz2)
        knn_idx = dists.argsort()[:, :, :self.k*2:2]  # b x n x k
        knn_xyz = index_points(xyz2, knn_idx)
        pre = features1
        x1 = self.fc1(features1)
        x2 = self.fc1(features2)
        q, k, v = self.w_qs(x1), index_points(self.w_ks(x2), knn_idx), index_points(self.w_vs(x2), knn_idx)

        pos_enc = self.fc_delta(xyz1[:, :, None] - knn_xyz)  # b x n x k x f
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
