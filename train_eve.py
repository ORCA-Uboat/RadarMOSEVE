import os
import torch
import torch.nn as nn
import logging
import random
import importlib
import shutil
import provider
import numpy as np

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

    # print(args.pretty())

    data_path = args.data_path
    TRAIN_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    VAL_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='val', normal_channel=args.normal)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)
    output_dim = args.output_dim
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    shutil.copy(hydra.utils.to_absolute_path('models/{}/radar_transformer.py'.format(args.model.name)), '.')
    shutil.copy(hydra.utils.to_absolute_path('dataset_eve.py'), '.')
    shutil.copy(hydra.utils.to_absolute_path('train_eve.py'), '.')

    eve_module = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'EVE_module')(args).cuda()
    criterion = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'Doppler_loss')().cuda()

    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            eve_module.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(eve_module.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_mean_speed = 1000
    best_mean_points = 1000

    for epoch in range(start_epoch, args.epoch):
        mean_speed = []
        mean_loss = []
        mean_loss_eve = []
        mean_loss_points = []

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        eve_module = eve_module.apply(lambda x: bn_momentum_adjust(x, momentum))
        eve_module = eve_module.train()
        

        '''learning one epoch'''
        for i, (points, points1, velocity, seg, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, points1 = points.data.numpy(), points1.data.numpy()
            points[:, :, 0:3], points1[:, :, :3] = provider.random_scale_point_cloud1(points[:, :, 0:3], points1[:, :, :3])
            points = torch.Tensor(points)
            points1 = torch.Tensor(points1)
            points, points1, seg, velocity = points.float().cuda(), points1.float().cuda(), seg.float().cuda(), velocity.float().cuda()
            optimizer.zero_grad()
            velocity_pred = eve_module(points, points1)
            velocity_pred = velocity_pred.contiguous().view(-1, output_dim)

            velocity = velocity.view(-1, 1)
            loss, loss_eve, loss_points = criterion(velocity_pred, velocity, points, seg)
            loss.backward()
            optimizer.step()

            dis_speed = abs(velocity_pred[:, 0]-velocity[:, 0]).cpu().detach().numpy().tolist()
            mean_loss.append(loss.cpu().detach().numpy())
            mean_loss_eve.append(loss_eve.cpu().detach().numpy())
            mean_loss_points.append(loss_points.cpu().detach().numpy())
            mean_speed.extend(dis_speed)

        train_instance_acc_speed = np.mean(mean_speed)
        train_instance_loss = np.mean(mean_loss)
        train_instance_loss_eve= np.mean(mean_loss_eve)
        train_instance_loss_points = np.mean(mean_loss_points)
        logger.info('Train accuracy speed is: %.5f Loss is: %.5f Loss eve is: %.5f Loss points is: %.5f'  % 
                (train_instance_acc_speed, train_instance_loss, train_instance_loss_eve, train_instance_loss_points))

        with torch.no_grad():
            test_metrics = {}
            total_mean_speed = []
            total_loss = []
            total_loss_eve = []
            total_loss_points = []

            eve_module = eve_module.eval()

            for batch_id, (points, points1, velocity, seg, _) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                points, points1, velocity = points.float().cuda(), points1.float().cuda(), velocity.float().cuda()
                velocity_pred = eve_module(points, points1)
                loss, loss_eve, loss_points = criterion(velocity_pred, velocity, points, seg)

                velocity_pred = velocity_pred.contiguous().view(-1, output_dim)
                velocity = velocity.view(-1, output_dim)

                dis_speed = abs(velocity_pred[:, 0]-velocity[:, 0]).cpu().detach().numpy().tolist()
                total_mean_speed.extend(dis_speed)
                total_loss.append(loss.cpu().detach().numpy())
                total_loss_eve.append(loss_eve.cpu().detach().numpy())
                total_loss_points.append(loss_points.cpu().detach().numpy())
       
            test_metrics['total_mean_speed'] = np.mean(total_mean_speed)
            test_metrics['total_loss'] = np.mean(total_loss)
            test_metrics['total_loss_eve'] = np.mean(total_loss_eve)
            test_metrics['total_loss_points'] = np.mean(total_loss_points)

        logger.info('Epoch %d test_mean_speed: %.5f total_loss: %.5f total_loss_eve: %.5f total_loss_points: %.5f' % (
            epoch + 1, test_metrics['total_mean_speed'],  test_metrics['total_loss'], 
            test_metrics['total_loss_eve'], test_metrics['total_loss_points']))

        if (test_metrics['total_mean_speed'] <= best_mean_speed):
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc_speed': train_instance_acc_speed,
                'test_mean_speed': test_metrics['total_mean_speed'],
                'total_loss': test_metrics['total_loss'],
                'total_loss_eve': test_metrics['total_loss_eve'],
                'total_loss_points': test_metrics['total_loss_points'],
                'model_state_dict': eve_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

            best_test_mean_speed = np.mean(total_mean_speed)

        torch.save(state, 'last.pth')
        if test_metrics['total_mean_speed'] < best_mean_speed:
            best_mean_speed = test_metrics['total_mean_speed']
        if test_metrics['total_loss_points'] < best_mean_points:
            best_mean_points = test_metrics['total_loss_points']
        logger.info('Best total mean speed is: %.5f' % best_mean_speed)
        logger.info('Best total test mean speed is: %.5f' % best_test_mean_speed)
        global_epoch += 1


if __name__ == '__main__':
    main()