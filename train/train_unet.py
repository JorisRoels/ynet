"""
    This is a script that trains the U-Net
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
from neuralnets.util.losses import DiceLoss
from neuralnets.networks.unet import UNet2D
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.datasets import StronglyLabeledVolumeDataset

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--device", help="GPU device for computations", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="unet")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=50)

# network parameters
parser.add_argument("--data_file", help="Path to the JSON data file", type=str, default="epfl.json")
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

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
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
df = json.load(open(args.data_file))
input_shape = (1, args.input_size[0], args.input_size[1])
print('[%s] Loading data' % (datetime.datetime.now()))
augmenter = Compose([ToFloatTensor(device=args.device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
                     ContrastAdjust(adj=0.1, include_segmentation=True),
                     RandomDeformation_2D(input_shape[1:], grid_size=(64, 64), sigma=0.01, device=args.device,
                                          include_segmentation=True)])
train_src = StronglyLabeledVolumeDataset(df['raw'], df['labels'], split_orientation=df['split-orientation'],
                                         split_location=df['split-location'], input_shape=input_shape,
                                         len_epoch=args.len_epoch, type=df['type'], train=True)
test_src = StronglyLabeledVolumeDataset(df['raw'], df['labels'], split_orientation=df['split-orientation'],
                                        split_location=df['split-location'], input_shape=input_shape,
                                        len_epoch=args.len_epoch, type=df['type'], train=False)
train_loader_src = DataLoader(train_src, batch_size=args.train_batch_size)
test_loader_src = DataLoader(test_src, batch_size=args.test_batch_size)

"""
    Build the network
"""
print('[%s] Building the network' % (datetime.datetime.now()))
net = UNet2D(feature_maps=args.fm, levels=args.levels, norm=args.norm, activation=args.activation,
             coi=args.classes_of_interest)

"""
    Setup optimization for training
"""
print('[%s] Setting up optimization for training' % (datetime.datetime.now()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print('[%s] Starting training' % (datetime.datetime.now()))
net.train_net(train_loader_src, test_loader_src, DiceLoss(), optimizer, args.epochs, scheduler=scheduler,
              augmenter=augmenter, print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)

"""
    Validate the trained network
"""
validate(net, test_src.data, test_src.labels, args.input_size, batch_size=args.test_batch_size,
         classes_of_interest=args.classes_of_interest, write_dir=os.path.join(args.log_dir, 'segmentation_final'),
         val_file=os.path.join(args.log_dir, 'validation_final.npy'))
net.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch')))
validate(net, test_src.data, test_src.labels, args.input_size, batch_size=args.test_batch_size,
         classes_of_interest=args.classes_of_interest, write_dir=os.path.join(args.log_dir, 'segmentation_best'),
         val_file=os.path.join(args.log_dir, 'validation_best.npy'))

print('[%s] Finished!' % (datetime.datetime.now()))
