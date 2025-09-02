"""ACDC: total 1356 samples; 30 samples for vadilation;
57 iterations per epoch; max epoch: 527.
UAMT: Uncertainty-Aware Mean Teacher
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from val_2D import test_all_case_2D
from utils import losses, ramps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../../data/ACDC', help='Data root path')
    parser.add_argument('--data_name', type=str,
                        default='ACDC', help='Data name')  
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name, select: unet')
    parser.add_argument('--exp', type=str,
                        default='UAMT', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='MAAGfold', help='cross validation fold')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--ES_interval', type=int,
                        default=10000, help='maximum iteration iternal for early-stopping')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input. Specially, [224, 224] for swinunet')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    args = parser.parse_args()
    return args

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * ramps.sigmoid_rampup(epoch, 60)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    ES_interval = args.ES_interval

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # Create model
    model = create_model()
    ema_model = create_model(ema=True)

    # create Dataset
    db_train = BaseDataSets( base_dir=args.root_path, split="train", transform=transforms.Compose(
                            [RandomGenerator(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Data loader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0

    # Training
    model.train()
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for iter, sampled_batch in enumerate(trainloader):

            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()

            outputs = model(img)
            loss_ce = ce_loss(outputs, label.long())
            supervised_loss = loss_ce

            rot_times = random.randrange(0, 4)
            rotated_volume_batch = torch.rot90(img, rot_times, [2, 3])
            noise = torch.clamp(torch.randn_like(
                rotated_volume_batch) * 0.1, -0.2, 0.2)
            
            with torch.no_grad():
                ema_inputs = rotated_volume_batch + noise
                ema_output = ema_model(ema_inputs)
            T = 8
            _, _, w, h = img.shape
            volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * \
                torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            consistency_weight = get_current_consistency_weight(
                iter_num // 1000)
            
            rotated_ouputs = torch.rot90(outputs, rot_times, [2, 3])
            consistency_dist = losses.softmax_mse_loss(
                rotated_ouputs, ema_output)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num,
                                                            max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(
                mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            
            # Validation
            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f' 
                    %(iter_num, loss.item(), loss_ce.item()))
                
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i], iter_num)
             
                if metric_list[:, 0].mean() > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = metric_list[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', metric_list[:, 0].mean(), iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, metric_list[:, 0].mean()))

                model.train()


            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num - fresh_iter_num >= ES_interval:
                logging.info("early stooping since there is no model updating over 1w \
                    iteration, iter:{} ".format(iter_num))
                break

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations or (iter_num - fresh_iter_num >= ES_interval):
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = parse_args()
    snapshot_path = "../../checkpoints/{}_{}".format(args.data_name, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(snapshot_path+"/train_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s')) 
    logger.addHandler(console_handler)
    logger.info(str(args))
    start_time = time.time()
    train(args, snapshot_path)
    time_s = time.time()-start_time
    logging.info("time cost: {} s, i.e, {} h".format(time_s,time_s/3600))