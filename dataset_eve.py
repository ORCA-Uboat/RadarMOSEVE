import numpy as np
import os
from torch.utils.data import Dataset
import torch
import tqdm

class MovingDataset(Dataset):
    def __init__(self, root='', npoints=512, split='train', normal_channel=True):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.datapath = []
        self.datapath1 = []
        self.velocity_path = []
        self.split = split
        self.mean_speed = []

        if split == 'train':
            with open(os.path.join(self.root, 'train.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                for data_path in data_nums:
                    past_data_path = int(data_path)-10
                    if past_data_path in data_nums and (past_data_path + 10) in data_nums :
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'speed', '%06d.txt'%data_path))
        elif split == 'test':
            with open(os.path.join(self.root, 'test.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                i = 0
                for data_path in tqdm.tqdm(data_nums):
                    past_data_path = int(data_path)-10
                    if past_data_path in data_nums and (past_data_path + 10) in data_nums :
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'speed', '%06d.txt'%data_path))
        elif split == 'val':
            with open(os.path.join(self.root, 'val.txt')) as f:
                data_name = f.readlines()
                np.sort(data_name)
                data_nums = [int(num[:-5]) for num in data_name]
                for data_path in data_nums:
                    past_data_path = int(data_path)-10
                    if past_data_path in data_nums and (past_data_path + 10) in data_nums :
                        data = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%data_path)).reshape(-1, 5)
                        data1 = np.loadtxt(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path)).reshape(-1, 5)
                        if np.sum(data[:, -1]==0) > 5 and np.sum(data1[:, -1]==0) > 5:
                            self.datapath.append(os.path.join(self.root, 'points', '%06d.xyz'%data_path))
                            self.datapath1.append(os.path.join(self.root, 'points', '%06d.xyz'%past_data_path))
                            self.velocity_path.append(os.path.join(self.root, 'speed', '%06d.txt'%data_path))
        else:
            exit
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):

        fn = self.datapath[index]
        fn1 = self.datapath1[index]
        vn1 = self.velocity_path[index]

        data = np.loadtxt(fn).astype(np.float32).reshape(-1, 5)
        data1 = np.loadtxt(fn1).astype(np.float32).reshape(-1, 5)
        velocity = np.loadtxt(vn1).astype(np.float32)
            
        if self.normal_channel:
            point_set = data[:, 0:3]
            point_set1 = data1[:, 0:3]
        else:
            point_set = data[:, 0:4]
            point_set1 = data1[:, 0:4]
        seg = data[:, -1].astype(np.int32)
        seg[seg==252] = 1
        
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        choice1 = np.random.choice(len(point_set1), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        
        seg = seg[choice]
        point_set1 = point_set1[choice1, :]

        vr = point_set[:, 3]
        cos_phi = np.linalg.norm(point_set[:, :2], axis=1) / np.linalg.norm(point_set[:, :3], axis=1)
        vr = vr / cos_phi
        point_set[:, 3] = vr

        vr = point_set1[:, 3]
        cos_phi = np.linalg.norm(point_set1[:, :2], axis=1) / np.linalg.norm(point_set1[:, :3], axis=1)
        vr = vr / cos_phi 
        point_set1[:, 3] = vr
        return point_set, point_set1, velocity[[0]], seg, int(fn[-10:-4])


    def __len__(self):
        return len(self.datapath)