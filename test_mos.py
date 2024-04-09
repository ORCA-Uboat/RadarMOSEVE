import os
import torch
import logging
import importlib
import numpy as np

from tqdm import tqdm
from dataset_mos import MovingDataset
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
setup_seed(3325)

seg_classes = {'move': [1], 'static': [0]}

@hydra.main(config_path='config', config_name='mos')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())


    data_path = args.data_path
    TEST_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)

    mos_network = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'MOS_module')(args).cuda()

    checkpoint = torch.load('mos.pth')
    mos_network.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Use pretrain model')


    with torch.no_grad():
        mos_network = mos_network.eval()

        for batch_id, (points1, points2, target, idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points1.size()
            points1, target = points1.float().cuda(), target.long().cuda()
            points2 = points2.float().cuda()
            seg_pred = mos_network(points1, points2)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT, 1)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i] = np.argmax(logits, axis=1).reshape(-1,1)
            points1 = points1.cpu().numpy()
            points1 = np.concatenate((points1, cur_pred_val.reshape(cur_batch_size, NUM_POINT, 1)), axis=2)
            points1 = np.concatenate((points1, target.reshape(cur_batch_size, NUM_POINT, 1)), axis=2)
            for i in range(cur_batch_size):
                np.savetxt('/home/orca/pcs/Radar_MOSEVE/seg/%06d.txt'%(idx[i]), points1[i])


if __name__ == '__main__':
    main()