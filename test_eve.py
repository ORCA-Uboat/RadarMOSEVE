import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset_eve import MovingDataset
import hydra
import omegaconf
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# random seed
setup_seed(3108)

@hydra.main(config_path='config', config_name='eve')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    data_path = args.data_path
    TEST_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)
    output_dim = args.output_dim

    eve_module = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'EVE_module')(args).cuda()

    checkpoint = torch.load('eve.pth')
    eve_module.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        eve_module = eve_module.eval()

        for batch_id, (points, points1, velocity, _, idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, points1 = points.float().cuda(), points1.float().cuda()
            velocity_pred = eve_module(points, points1)
            velocity_pred = velocity_pred.cpu().data.numpy()

            for curr_name, pred in zip(idx, velocity_pred):
                np.savetxt('velocity_pred/%06d.txt'%curr_name, pred)



if __name__ == '__main__':
    main()