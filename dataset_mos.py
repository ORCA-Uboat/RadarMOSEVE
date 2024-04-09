import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json
import tqdm

class MovingDataset(Dataset):
    def __init__(self, root='', npoints=512, split='train',normal_channel=True):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.datapath = []
        self.datapath1 = []
        self.velocity_path = []
        self.velocity_path1 = []
        self.velocity_path2 = []
        self.split = split
        self.mean_speed = []
        gap = 10
        if split == 'train':
            with open(os.path.join(self.root, 'train.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                
                for data_path in tqdm.tqdm(data_nums):
                    past_data_path = int(data_path)-gap
                    velocity_path = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path))
                    velocity_path1 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap))
                    velocity_path2 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path))
                    
                    if past_data_path in data_nums and (past_data_path - gap) in data_nums and os.path.exists(velocity_path) and os.path.exists(velocity_path1) and os.path.exists(velocity_path2):
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                        
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%past_data_path))
                            self.velocity_path1.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap)))
                            self.velocity_path2.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path)))
        elif split == 'test':
            with open(os.path.join(self.root, 'test.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                for data_path in tqdm.tqdm(data_nums):
                    past_data_path = int(data_path)-gap
                    velocity_path = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path))
                    velocity_path1 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap))
                    velocity_path2 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path))
                    if past_data_path in data_nums and (past_data_path - gap) in data_nums and os.path.exists(velocity_path) and os.path.exists(velocity_path1) and os.path.exists(velocity_path2):    
                    # if past_data_path in data_nums and (past_data_path - gap) in data_nums :
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                        # if np.sum(data[:, -1]==0) < 5 and np.sum(data1[:, -1]==0) < 5 and np.sum(data[:, -1]==252) > 5 and np.sum(data1[:, -1]==252) > 5:
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%past_data_path))
                            self.velocity_path1.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap)))
                            self.velocity_path2.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path)))
        elif split == 'val':
            with open(os.path.join(self.root, 'val.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                for data_path in tqdm.tqdm(data_nums):
                    past_data_path = int(data_path)-gap
                    velocity_path = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path))
                    velocity_path1 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap))
                    velocity_path2 = os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path))
                    if past_data_path in data_nums and (past_data_path - gap) in data_nums and os.path.exists(velocity_path) and os.path.exists(velocity_path1) and os.path.exists(velocity_path2):    
                    # if past_data_path in data_nums and (past_data_path - gap) in data_nums :
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                        # if np.sum(data[:, -1]==0) < 5 and np.sum(data1[:, -1]==0) < 5 and np.sum(data[:, -1]==252) > 5 and np.sum(data1[:, -1]==252) > 5:
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%past_data_path))
                            self.velocity_path1.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(past_data_path-gap)))
                            self.velocity_path2.append(os.path.join(self.root, 'velocity_pred', '%06d.txt'%(data_path)))
        else:
            exit
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):


        fn = self.datapath[index]
        fn1 = self.datapath1[index]
        vn = self.velocity_path[index]
        vn1 = self.velocity_path2[index]

        data = np.loadtxt(fn).astype(np.float32).reshape(-1, 5)
        data1 = np.loadtxt(fn1).astype(np.float32).reshape(-1, 5)
        velocity = np.loadtxt(vn).astype(np.float32)
        velocity1 = np.loadtxt(vn1).astype(np.float32)
        if self.normal_channel:
            point_set = data[:, 0:3]
            point_set1 = data1[:, 0:3]
        else:
            point_set = data[:, 0:4]
            point_set1 = data1[:, 0:4]
        seg = data[:, -1].astype(np.int32)
        seg[seg==252] = 1
        # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        choice1 = np.random.choice(len(point_set1), self.npoints, replace=True)
        
        # resample
        point_set = point_set[choice, :]
        point_set1 = point_set1[choice1, :]
        seg = seg[choice]

        if not self.normal_channel:
            point_set = self.calc_speed(point_set, velocity1)
            point_set1 = self.calc_speed(point_set1, velocity)
        
        return point_set, point_set1, seg, int(fn[-10:-4])

    def calc_speed(self, point_set, speed):
        yaw_p = np.degrees(np.arctan2(point_set[:, 1], point_set[:, 0]))
        deg_m1 = np.radians(90-yaw_p)

        if self.split=='train':
            randomscale = 0.4
            randombias  = 0.2
        else:
            randomscale = 0
            randombias  = 0
        rand_value = np.random.random()
        vr_speed= (speed+rand_value*randomscale-randombias) * np.cos(deg_m1)
        vr = point_set[:, 3]
        cos_phi = np.linalg.norm(point_set[:, :2], axis=1) / np.linalg.norm(point_set[:, :3], axis=1)
        vr = vr / cos_phi
        vr = vr + vr_speed 
        point_set[:, 3] = vr
        return point_set

    def __len__(self):
        return len(self.datapath)