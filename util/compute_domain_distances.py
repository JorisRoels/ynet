import json
import os
import numpy as np
import torch
import argparse
import datetime
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader

from neuralnets.util.tools import tensor_to_device, module_to_device, set_seed
from neuralnets.util.io import print_frm, mkdir

from data.datasets import UnlabeledVolumeDataset


def _load_daae(state_dict, device=0):
    """
    Load a pretrained pytorch DAAE2D state dict

    :param state_dict: state dict of a daae
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    :return: a module that corresponds to the trained network
    """
    from networks.daae import DAAE2D

    # extract the hyperparameters of the network
    feature_maps = state_dict['encoder.features.convblock1.conv1.unit.0.weight'].size(0)
    levels = int(list(state_dict.keys())[-15][len('decoder.features.upconv')])
    bottleneck_in_features = state_dict['encoder.bottleneck.0.weight'].size(1)
    bottleneck_dim = state_dict['encoder.bottleneck.0.weight'].size(0)
    x = int(np.sqrt(bottleneck_in_features * 2 ** (3 * levels - 1) / feature_maps))
    norm = 'batch' if 'norm' in list(state_dict.keys())[2] else 'instance'
    lambda_reg = 0.0
    activation = 'relu'
    dropout_enc = 0.0
    n_hidden = state_dict['domain_classifier.linear2.unit.0.weight'].size(1)
    n_domains = state_dict['domain_classifier.linear2.unit.0.weight'].size(0)

    # initialize the network
    net = DAAE2D(lambda_reg=lambda_reg, input_size=[x, x], bottleneck_dim=bottleneck_dim,
                 feature_maps=feature_maps, levels=levels, dropout_enc=dropout_enc, norm=norm,
                 activation=activation, fc_channels=(n_hidden, n_domains))

    # load the parameters in the model
    net.load_state_dict(state_dict)

    # map to the correct device
    module_to_device(net, device=device)

    return net


def _load_net(model_file, device=0):
    """
    Load a pretrained pytorch network, currently only support for U-Net and BVAE

    :param model_file: path to the state dict checkpoint
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    :return: a module that corresponds to the trained network
    """

    # load the state dict to the correct device
    map_location = 'cpu' if not torch.cuda.is_available() or device == 'cpu' else 'cuda:' + str(device)
    state_dict = torch.load(model_file, map_location=map_location)

    net = _load_daae(state_dict, device=device)

    return net


def _dist_data(dl, dl_, net):

    z = np.zeros((n, net.bottleneck_dim))
    z_ = np.zeros((n, net.bottleneck_dim))

    b = dl.batch_size
    for i, data in enumerate(dl):
        x = tensor_to_device(data, device).float()
        z[i * b: i * b + x.size(0)] = net.encoder(x).detach().cpu().numpy()
    b_ = dl_.batch_size
    for i, data in enumerate(dl_):
        x = tensor_to_device(data, device).float()
        z_[i * b_: i * b_ + x.size(0)] = net.encoder(x).detach().cpu().numpy()

    return np.mean(distance_matrix(z, z_))


"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--device", help="GPU device for computations", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="ae")

# network parameters
parser.add_argument("--data_file", help="Path to the JSON data file", type=str, default="joint.json")
parser.add_argument("--net", help="Path to the model file", type=str, default="daae.pytorch")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="64,64")
parser.add_argument("--batch_size", help="Batch size in the testing stage", type=int, default=4)
parser.add_argument("--n", help="Amount of samples used in the source and target domain", type=int, default=16)

args = parser.parse_args()
print('[%s] Arguments: ' % (datetime.datetime.now()))
print('[%s] %s' % (datetime.datetime.now(), args))
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
Fix seed (for reproducibility)
"""
set_seed(args.seed)

# parameters
device = args.device  # computing device
n = args.n  # amount of samples to be extracted per domain
b = args.batch_size  # batch size for processing
input_size = args.input_size

# load the network
print_frm('Loading network')
model_file = args.net
net = _load_net(model_file, device)

# load reference patch
print_frm('Loading data')
data_file = args.data_file
df = json.load(open(data_file))
n_domains = len(df['raw'])
input_shape = (1, input_size[0], input_size[1])

# datasets
dss = []
for d in range(n_domains):
    print_frm('Loading %s' % df['raw'][d])
    dss.append(UnlabeledVolumeDataset(df['raw'][d], split_orientation=df['split-orientation'][d],
                                      split_location=df['split-location'][d], input_shape=input_shape,
                                      type=df['types'][d], train=False, len_epoch=n))

# for all other domains, compute embeddings of randomly selected patches
print_frm('Computing embeddings remaining domains')
sample_dists = np.zeros((n_domains, n_domains))
for d_src in range(n_domains):
    print_frm('Processing src domain %d/%d' % (d_src, n_domains))
    test_src = dss[d_src]
    dl = DataLoader(test_src, batch_size=args.batch_size)
    for d_tar in range(n_domains):
        print_frm('  Processing tar domain %d/%d' % (d_tar, n_domains))
        if d_src != d_tar:
            test_tar = dss[d_tar]
            dl_ = DataLoader(test_tar, batch_size=args.batch_size)
            sample_dists[d_src, d_tar] = _dist_data(dl, dl_, net)

# save results
mkdir(args.log_dir)
np.save(os.path.join(args.log_dir, 'dom_dists.npy'), sample_dists)
