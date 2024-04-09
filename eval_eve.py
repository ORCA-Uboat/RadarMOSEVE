import argparse
import os
import torch
import torch.nn as nn
import datetime
import logging
import sys
import random
import importlib
import shutil
import provider
import numpy as np
import time

from pathlib import Path
from tqdm import tqdm
from dataset_eve import MovingDataset
import hydra
import omegaconf


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
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)
    args.output_dim = 1
    output_dim = args.output_dim

    eve_module = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'EVE_module')(args).cuda()
    criterion = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'Doppler_loss')().cuda()

    checkpoint = torch.load('eve.pth')

    eve_module.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Use pretrain model')

    with torch.no_grad():
        test_metrics = {}
        total_mean_speed = []
        total_loss = []
        total_loss_eve = []
        total_loss_points = []

        eve_module = eve_module.eval()
        all_t = []
        for batch_id, (points, points1, target, seg, idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, points1, target = points.float().cuda(), points1.float().cuda(), target.float().cuda()

            velocity_pred = eve_module(points, points1)
            velocity_pred = velocity_pred.contiguous().view(-1, output_dim)
            target = target.view(-1, output_dim)
            loss, loss_eve, loss_points = criterion(velocity_pred, target, points, seg)
            print(velocity_pred, target)
            dis_speed = abs(velocity_pred[:, 0]-target[:, 0]).cpu().detach().numpy().tolist()
            total_mean_speed.extend(dis_speed)
            total_loss.append(loss.cpu().detach().numpy())
            total_loss_eve.append(loss_eve.cpu().detach().numpy())
            total_loss_points.append(loss_points.cpu().detach().numpy())
            # recording data
            # cur_pred_val = velocity_pred.cpu().data.numpy()
            # for curr_name, pred in zip(idx, cur_pred_val):
                # pred1 = np.concatenate((np.zeros(2), pred, np.zeros(1).reshape(-1)), axis=0)
                # np.savetxt('velocity_pred/%06d.txt'%curr_name, pred)

        test_metrics['total_mean_speed'] = np.mean(total_mean_speed)
        test_metrics['total_loss'] = np.mean(total_loss)
        test_metrics['total_loss_ve'] = np.mean(total_loss_eve)
        test_metrics['total_loss_points'] = np.mean(total_loss_points)

    logger.info('Epoch %d test_mean_speed: %.5f  total_loss: %.5f total_loss_eve: %.5f total_loss_points: %.5f' % (
        0, test_metrics['total_mean_speed'], test_metrics['total_loss'], 
        test_metrics['total_loss_ve'], test_metrics['total_loss_points']))



if __name__ == '__main__':
    main()