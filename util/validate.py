"""
    This is a script that validates a U-Net
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
parser.add_argument("--net", help="Path to the network parameters", type=str, default="../train/unet/checkpoint.pytorch")
parser.add_argument("--data_file", help="Path to the JSON data file", type=str, default="../train/epfl.json")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="256,256")
parser.add_argument("--classes_of_interest", help="List of indices that correspond to the classes of interest",
                    type=str, default="0,1")

# optimization parameters
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
test_src = StronglyLabeledVolumeDataset(df['source']['raw'], df['source']['labels'],
                                        split_orientation=df['source']['split-orientation'],
                                        split_location=df['source']['split-location'], type=df['types'], train=False)

"""
    Load the network
"""
print('[%s] Loading the network' % (datetime.datetime.now()))
net = torch.load(args.net, map_location='cuda:' + str(args.device))
if net.__class__ != UNet2D:
    net = net.get_unet()

"""
    Validate the trained network
"""
validate(net, test_src.data, test_src.labels, args.input_size, batch_size=args.test_batch_size, track_progress=True,
         write_dir=os.path.join(args.log_dir, 'segmentation'), val_file=os.path.join(args.log_dir, 'validation.npy'))

print('[%s] Finished!' % (datetime.datetime.now()))
