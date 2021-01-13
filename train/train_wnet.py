"""
    This is a script that trains the W-Net in a semi-supervised fashion
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os
import json

import torch.optim as optim
from neuralnets.util.augmentation import *
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import validate
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from networks.wnet import WNet2D, JOINT, RECONSTRUCTION, SEGMENTATION
from data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--device", help="GPU device for computations", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="wnet")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=10)

# network parameters
parser.add_argument("--data_file_src", help="Path to the JSON source data file", type=str, default="epfl.json")
parser.add_argument("--data_file_tar_ul", help="Path to the JSON unlabeled target data file", type=str,
                    default="vnc_ul.json")
parser.add_argument("--data_file_tar_l", help="Path to the JSON labeled data file", type=str, default="vnc_l.json")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="256,256")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)",
                    type=int, default=4)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.00)
parser.add_argument("--norm", help="Normalization in the network (batch or instance)", type=str, default="instance")
parser.add_argument("--activation", help="Non-linear activations in the network", type=str, default="relu")
parser.add_argument("--classes_of_interest", help="List of indices that correspond to the classes of interest",
                    type=str, default="0,1")
parser.add_argument("--available_target_labels", help="Amount of available target labels", type=int, default=-1)
parser.add_argument("--end2end", help="Train the network end to end", action="store_true")

# regularization parameters
parser.add_argument('--lambda_rec', help='Regularization parameter for W-Net reconstruction', type=float, default=1e-1)
parser.add_argument('--lambda_dc', help='Regularization parameter for W-Net domain confusion', type=float, default=1e-1)
parser.add_argument("--conv_channels", help="Amount of convolutional channels", type=str, default="16,16,16,16,16")
parser.add_argument("--fc_channels", help="Amount of fully connected channels", type=str, default="128,32")

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs_src", help="Total number of epochs to pretrain on source", type=int, default=0)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=250)
parser.add_argument("--len_epoch", help="Number of iteration in each epoch", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=2)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)

args = parser.parse_args()
print('[%s] Arguments: ' % (datetime.datetime.now()))
print('[%s] %s' % (datetime.datetime.now(), args))
args.input_size = [int(item) for item in args.input_size.split(',')]
args.classes_of_interest = [int(c) for c in args.classes_of_interest.split(',')]
args.conv_channels = [int(c) for c in args.conv_channels.split(',')]
args.fc_channels = [int(c) for c in args.fc_channels.split(',')]

"""
Fix seed (for reproducibility)
"""
set_seed(args.seed)

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

"""
    Load the data
"""
df_src = json.load(open(args.data_file_src))
df_tar_ul = json.load(open(args.data_file_tar_ul))
df_tar_l = json.load(open(args.data_file_tar_l))
input_shape = (1, args.input_size[0], args.input_size[1])
print('[%s] Loading data' % (datetime.datetime.now()))
augmenter = Compose([ToFloatTensor(device=args.device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
                     ContrastAdjust(adj=0.1, include_segmentation=True),
                     RandomDeformation_2D(input_shape[1:], grid_size=(64, 64), sigma=0.01, device=args.device,
                                          include_segmentation=True)])
train_src = StronglyLabeledVolumeDataset(df_src['raw'], df_src['labels'], split_orientation=df_src['split-orientation'],
                                         split_location=df_src['split-location'], input_shape=input_shape,
                                         len_epoch=args.len_epoch, type=df_src['type'], train=True)
test_src = StronglyLabeledVolumeDataset(df_src['raw'], df_src['labels'], split_orientation=df_src['split-orientation'],
                                        split_location=df_src['split-location'], input_shape=input_shape,
                                        len_epoch=args.len_epoch, type=df_src['type'], train=False)
train_tar_ul = UnlabeledVolumeDataset(df_tar_ul['raw'], split_orientation=df_tar_ul['split-orientation'],
                                      split_location=df_tar_ul['split-location'], input_shape=input_shape,
                                      len_epoch=args.len_epoch, type=df_tar_ul['type'], train=True)
test_tar_ul = UnlabeledVolumeDataset(df_tar_ul['raw'], split_orientation=df_tar_ul['split-orientation'],
                                     split_location=df_tar_ul['split-location'], input_shape=input_shape,
                                     len_epoch=args.len_epoch, type=df_tar_ul['type'], train=False)
train_tar_l = StronglyLabeledVolumeDataset(df_tar_l['raw'], df_tar_l['labels'],
                                           split_orientation=df_tar_l['split-orientation'],
                                           split_location=df_tar_l['split-location'], input_shape=input_shape,
                                           len_epoch=args.len_epoch, type=df_tar_l['type'], train=True,
                                           available=args.available_target_labels)
test_tar_l = StronglyLabeledVolumeDataset(df_tar_l['raw'], df_tar_l['labels'],
                                          split_orientation=df_tar_l['split-orientation'],
                                          split_location=df_tar_l['split-location'], input_shape=input_shape,
                                          len_epoch=args.len_epoch, type=df_tar_l['type'],
                                          train=False)
train_loader_src = DataLoader(train_src, batch_size=args.train_batch_size)
test_loader_src = DataLoader(test_src, batch_size=args.test_batch_size)
train_loader_tar_ul = DataLoader(train_tar_ul, batch_size=args.train_batch_size)
test_loader_tar_ul = DataLoader(test_tar_ul, batch_size=args.train_batch_size)
if args.available_target_labels == 0:
    train_loader_tar_l = None
else:
    train_loader_tar_l = DataLoader(train_tar_l, batch_size=args.train_batch_size)
test_loader_tar_l = DataLoader(test_tar_l, batch_size=args.test_batch_size)

"""
    Build the network
"""
print('[%s] Building the network' % (datetime.datetime.now()))
net = WNet2D(feature_maps=args.fm, levels=args.levels, norm=args.norm, lambda_rec=args.lambda_rec,
             lambda_dc=args.lambda_dc, dropout_enc=args.dropout, activation=args.activation,
             coi=args.classes_of_interest, input_size=args.input_size, conv_channels=args.conv_channels,
             fc_channels=args.fc_channels)

"""
    Setup optimization for training
"""
print('[%s] Setting up optimization for training' % (datetime.datetime.now()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Pretrain the network
"""
if args.epochs_src > 0:
    print('[%s] Starting pre-training' % (datetime.datetime.now()))
    net.train_net(train_loader_src, train_loader_tar_ul, None, test_loader_src, test_loader_tar_ul, test_loader_tar_l,
                  optimizer, args.epochs_src, scheduler=scheduler, augmenter=augmenter, print_stats=args.print_stats,
                  log_dir=args.log_dir, device=args.device)

"""
    Train the network
"""
if args.end2end:
    print('[%s] Starting end-to-end training' % (datetime.datetime.now()))
    net.train_mode = JOINT
    net.train_net(train_loader_src, train_loader_tar_ul, train_loader_tar_l, test_loader_src, test_loader_tar_ul,
                  test_loader_tar_l, optimizer, args.epochs, scheduler=scheduler, augmenter=augmenter,
                  print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)
else:
    print('[%s] Starting reconstruction training' % (datetime.datetime.now()))
    net.train_mode = RECONSTRUCTION
    net.train_net(train_loader_src, train_loader_tar_ul, train_loader_tar_l, test_loader_src, test_loader_tar_ul,
                  test_loader_tar_l, optimizer, args.epochs, scheduler=scheduler, augmenter=augmenter,
                  print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)
    print('[%s] Starting segmentation training' % (datetime.datetime.now()))
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    net.train_mode = SEGMENTATION
    net.train_net(train_loader_src, train_loader_tar_ul, train_loader_tar_l, test_loader_src, test_loader_tar_ul,
                  test_loader_tar_l, optimizer, args.epochs, scheduler=scheduler, augmenter=augmenter,
                  print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)

"""
    Validate the trained network
"""
validate(net.get_unet(), test_tar_l.data, test_tar_l.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.log_dir, 'segmentation_final'),
         val_file=os.path.join(args.log_dir, 'validation_final.npy'), classes_of_interest=args.classes_of_interest)
net.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch')))
validate(net.get_unet(), test_tar_l.data, test_tar_l.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.log_dir, 'segmentation_best'),
         val_file=os.path.join(args.log_dir, 'validation_best.npy'), classes_of_interest=args.classes_of_interest)

print('[%s] Finished!' % (datetime.datetime.now()))
