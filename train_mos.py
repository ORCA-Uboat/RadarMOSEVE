import argparse
import os
import torch
import datetime
import logging
import sys
import random
import importlib
import shutil
import provider
import numpy as np

from tqdm import tqdm
from dataset_mos import MovingDataset
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

seg_classes = {'move': [1], 'static': [0]}

@hydra.main(config_path='config', config_name='mos')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    data_path = args.data_path
    TRAIN_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    VAL_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='val', normal_channel=args.normal)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    shutil.copy(hydra.utils.to_absolute_path('models/{}/radar_transformer.py'.format(args.model.name)), '.')
    shutil.copy(hydra.utils.to_absolute_path('dataset_mos.py'), '.')
    shutil.copy(hydra.utils.to_absolute_path('train_mos.py'), '.')

    mos_network = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'MOS_module')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        mos_network.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            mos_network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(mos_network.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        mean_loss = []

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
        mos_network = mos_network.apply(lambda x: bn_momentum_adjust(x, momentum))
        mos_network = mos_network.train()
        

        '''learning one epoch'''
        for i, (points1, points2, target, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points1 = points1.data.numpy()
            points2 = points2.data.numpy()
            points1[:, :, 0:3], points2[:, :, 0:3] = provider.shift_point_cloud1(points1[:, :, 0:3], points2[:, :, 0:3])
            points1, points2 = torch.Tensor(points1), torch.Tensor(points2)

            points1, points2, target = points1.float().cuda(), points2.float().cuda(), target.long().cuda()
            optimizer.zero_grad()
            seg_pred = mos_network(points1, points2)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, target)
            mean_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        train_instance_loss = np.mean(mean_loss)
        logger.info('Train accuracy is: %.5f Loss is: %.5f'  % (train_instance_acc, train_instance_loss))

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            mos_network = mos_network.eval()

            for batch_id, (points1, points2, target, _) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points1.size()
                points1, target, points2 = points1.float().cuda(), target.long().cuda(), points2.float().cuda()
                seg_pred = mos_network(points1, points2)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits, 1)

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = 'move'
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):
                            iou = 1.0
                        else:
                            iou = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(iou)
                    cat = 'static'
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and ( np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            iou = 1.0
                        else:
                            iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(iou)

            for cat in shape_ious.keys():
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
            for cat in sorted(shape_ious.keys()):
                logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious

        logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou']))
        if (test_metrics['class_avg_iou'] >= best_class_avg_iou):
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'model_state_dict': mos_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            best_class_avg_iou = test_metrics['class_avg_iou']
            best_acc = test_metrics['accuracy']
            torch.save(state, savepath)
            logger.info('Saving model....')
        torch.save(state, 'last.pth')
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()