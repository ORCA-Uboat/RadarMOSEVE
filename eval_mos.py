import os
import torch
import logging
import importlib
import shutil
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


seg_classes = {'move': [1], 'static': [0]}
setup_seed(3325)

@hydra.main(config_path='config', config_name='mos')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(args)

    data_path = args.data_path
    TEST_DATASET = MovingDataset(root=data_path, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = (3 if args.normal else 4)
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    mos_network = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'MOS_module')(args).cuda()

    # try:
    checkpoint = torch.load('mos.pth')
    mos_network.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        mos_network = mos_network.eval()
        for batch_id, (points, points1, target, idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, target = points.float().cuda(), target.long().cuda()
            points1 = points1.float().cuda()
            seg_pred = mos_network(points, points1)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i] = np.argmax(logits, axis=1)

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

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious

    logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f' % (
        1, test_metrics['accuracy'], test_metrics['class_avg_iou']))
        


if __name__ == '__main__':
    main()