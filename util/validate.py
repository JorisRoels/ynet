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

from neuralnets.util.augmentation import *
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import validate
from neuralnets.networks.unet import UNet2D

from data.datasets import StronglyLabeledVolumeDataset
from networks.unet_noda import UNetNoDA2D
from networks.unet_dat import UNetDAT2D
from networks.unet_mmd import UNetMMD2D
from networks.unet_ts import UNetTS2D
from networks.wnet import WNet2D
from networks.ynet import YNet2D

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
parser.add_argument("--net", help="Path to the network parameters", type=str,
                    default="../train/unet/checkpoint.pytorch")
parser.add_argument("--data_file", help="Path to the JSON data file", type=str, default="../train/epfl.json")
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
test_src = StronglyLabeledVolumeDataset(df['raw'], df['labels'], split_orientation=df['split-orientation'],
                                        split_location=df['split-location'], type=df['type'], train=False)

"""
    Load the network
"""
print('[%s] Loading the network' % (datetime.datetime.now()))


def _load_net(path, device):
    net_state = torch.load(path, map_location='cuda:' + str(device))

    # attempt to load the various networks
    modules = [UNetNoDA2D, UNetDAT2D, UNetMMD2D, UNetTS2D, WNet2D, YNet2D]
    net = None
    for module in modules:
        try:
            net = module(feature_maps=args.fm, levels=args.levels, norm=args.norm, activation=args.activation,
                         coi=args.classes_of_interest)
            net.load_state_dict(net_state)
            break
        except RuntimeError:
            continue

    return net


net = _load_net(args.net, args.device)
if net.__class__ != UNet2D:
    net = net.get_unet()

"""
    Validate the trained network
"""
validate(net, test_src.data, test_src.labels, args.input_size, batch_size=args.test_batch_size, track_progress=True,
         write_dir=os.path.join(args.log_dir, 'segmentation'), val_file=os.path.join(args.log_dir, 'validation.npy'))

print('[%s] Finished!' % (datetime.datetime.now()))
