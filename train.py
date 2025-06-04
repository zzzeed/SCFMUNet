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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                                             'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                                             'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching', )
parser.add_argument("--batch_size", default=4, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=200)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default="./ACDC/model_pth/")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./ACDC/predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
args = parser.parse_args("AAA".split())
# In[4]:
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
args.exp = 'ACDC' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'ACDC')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

current_time = time.strftime("%H%M%S")
print("The current time is", current_time)
snapshot_path = snapshot_path + '_run' + current_time

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

# In[5]:
net = SCFMUNet(num_classes=4,
            input_channels=3,
            depths=[2,2,2,2],
            depths_decoder=[2,2,2,2],
            drop_path_rate= 0.2,
            load_ckpt_path='./pre_trained_weights/vmamba_small_e238_ema.pth').cuda(0)
net.load_from()
if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_dcs_th = 0.865

logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
def val():
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(
            0).cpu().detach().numpy()
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y),
                                   order=3)  # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        P = net(val_image_batch)
        val_outputs = P
        val_outputs = torch.softmax(val_outputs, dim=1)

        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum += dc(val_outputs, val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: val_mean_dice : %f, val_best_dice : %f' % (performance, Best_dcs))

    return performance
iter_num = 0
best_state_dict = net.state_dict()
for epoch in tqdm(range(args.max_epochs)):
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        P = net(image_batch)
        loss = 0.0
        lc1, lc2 = 0.4, 0.6
        iout = P
        loss_ce = ce_loss(iout, label_batch[:].long())
        loss_dice = dice_loss(iout, label_batch, softmax=True)
        loss += (lc1 * loss_ce + lc2 * loss_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num % 50 == 0:
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    save_model_path1 = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path1)

    avg_dcs = val()
    if avg_dcs > Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'val_best.pth')
        #best_state_dict = net.state_dict()
        torch.save(net.state_dict(), save_model_path)
        # logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))

        Best_dcs = avg_dcs

    if epoch == 109:
        save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))

    # if epoch == 119:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 129:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 139:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 149:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 159:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 169:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 179:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))
    # if epoch == 189:
    #     save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
    #     torch.save(net.state_dict(), save_model_path)
    #     logging.info("save model to {}".format(save_model_path))
    #     print("save model to {}".format(save_model_path))

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}.pth'.format(epoch))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        break
save_model_path1 = os.path.join(snapshot_path, 'last.pth')
torch.save(best_state_dict, save_model_path1)
