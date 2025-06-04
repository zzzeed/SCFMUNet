
import cv2
import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time
import torch, gc
import numpy as np
from tqdm import tqdm
from medpy.metric import dc, hd95
from scipy.ndimage import zoom

from utils_ACDC.utils import powerset
from utils_ACDC.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils_ACDC.dataset_ACDC import ACDCdataset, RandomGenerator

from model.ours import SCFMUNet

# In[2]:


gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                                             'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                                             'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching', )
parser.add_argument("--batch_size", default=4, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=200)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default="./test_pth")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./best', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
args = parser.parse_args("AAA".split())
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

from utils_ACDC.utils import test_single_volume

def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
            print('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
            print('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info(
            'Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (
            performance, mean_hd95, mean_jacard, mean_asd))
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (
        performance, mean_hd95, mean_jacard, mean_asd))

        logging.info("Testing Finished!")
        print("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    args.is_pretrain = True
    args.is_savenii = True
    args.exp = 'tri'+ str(args.img_size)
    snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'tri')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    net = SCFMUNet(num_classes=4,
            input_channels=3,
            depths=[2,2,2,2],
            depths_decoder=[2,2,2,2],
            drop_path_rate= 0.2,
            load_ckpt_path='./pre_trained_weights/vmamba_small_e238_ema.pth').cuda(0)

    snapshot = './best.pth'


    print(snapshot)
    print(os.path.exists(snapshot))
    if not os.path.exists(snapshot): snapshot = snapshot.replace('last', 'epoch_' + str(args.max_epochs - 1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_best/test_log_bestours' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    if args.is_savenii:
        args.test_save_dir = './best/predictions/best'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    os.makedirs(test_save_path, exist_ok=True)

    db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    results = inference(args, net, testloader, test_save_path)

# In[ ]:


img = cv2.imread()

