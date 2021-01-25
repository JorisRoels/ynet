import json
import os
import numpy as np
import torch
import argparse
import datetime
from scipy.spatial import distance_matrix

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
parser.add_argument("--domain_id", help="Index of the reference domain", type=int, default=0)
parser.add_argument("--n", help="Amount of samples to be extracted from the reference domain", type=int, default=128)
parser.add_argument("--k", help="Amount of closest samples to save", type=int, default=5)

args = parser.parse_args()
print('[%s] Arguments: ' % (datetime.datetime.now()))
print('[%s] %s' % (datetime.datetime.now(), args))
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
Fix seed (for reproducibility)
"""
set_seed(args.seed)

# parameters
domain_id = args.domain_id  # id of the domain where a reference patch should be selected
device = args.device  # computing device
n = args.n  # amount of samples to be extracted per domain
b = args.batch_size  # batch size for processing
input_size = args.input_size
k = args.k  # amount of closest samples to be extracted

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
dataset_ref = UnlabeledVolumeDataset(df['raw'][domain_id], split_orientation=df['split-orientation'][domain_id],
                                     split_location=df['split-location'][domain_id], input_shape=input_shape,
                                     type=df['types'][domain_id], train=False)
x_ref = dataset_ref[0]

# compute embedding reference sample
print_frm('Computing embedding reference samples')
x_ref_t = tensor_to_device(torch.from_numpy(x_ref[np.newaxis, ...]), device).float()
z_ref = net.encoder(x_ref_t)
z_ref = z_ref.detach().cpu().numpy()

# for all other domains, compute embeddings of randomly selected patches
print_frm('Computing embeddings remaining domains')
z = np.zeros((n_domains, n, net.bottleneck_dim))
doms = np.zeros((n_domains, n), dtype=int)
samples = np.zeros((n_domains, n, input_size[0], input_size[1]))
reconstructions = np.zeros((n_domains, n, input_size[0], input_size[1]))
closest_samples = np.zeros((n_domains, k, input_size[0], input_size[1]))
closest_dists = np.zeros((n_domains, k))
for d in range(n_domains):
    print_frm('Processing domain %d/%d' % (d, n_domains))
    if d != domain_id:
        test = UnlabeledVolumeDataset(df['raw'][d], split_orientation=df['split-orientation'][d],
                                      split_location=df['split-location'][d], input_shape=input_shape,
                                      type=df['types'][d], train=False)
        for i in range(n // b):
            if i % 5 == 0:
                print_frm('    Processed %d/%d batches' % (i, n // b))
            # sample
            x = np.zeros((b, 1, input_size[0], input_size[1]))
            for j in range(b):
                xx = test[i]
                x[j] = xx
            samples[d, i * b: (i + 1) * b] = x[:, 0, ...]
            doms[d, i * b: (i + 1) * b] = d
            x = tensor_to_device(torch.from_numpy(x), device).float()

            # compute encoding
            bottleneck = net.encoder(x)
            bottleneck = bottleneck.detach().cpu().numpy()
            pred, dom_pred = net(x)
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            z[d, i * b: (i + 1) * b] = bottleneck
            reconstructions[d, i * b: (i + 1) * b] = pred[:, 0, ...]

    # for m in range(4):
    #     i = np.random.randint(n)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(samples[d, i], cmap='gray')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(reconstructions[d, i], cmap='gray')
    #     plt.show()

    # find k closest samples
    dists = distance_matrix(z_ref, z[d])
    inds = np.argsort(dists[0])
    for kk in range(k):
        closest_samples[d, kk, ...] = samples[d, inds[kk], ...]
        closest_dists[d, kk, ...] = dists[0, inds[kk]]

    # # apply u-map
    # print_frm('U-Map dimensionality reduction')
    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(z)

    # # show results
    # print_frm('Visualization')
    # for g in np.unique(doms):
    #     i = np.where(doms == g)
    #     plt.scatter(embedding[i, 0], embedding[i, 1], label=g)
    # plt.legend()
    # plt.show()

# save results
mkdir(args.log_dir)
np.save(os.path.join(args.log_dir, 'x_ref.npy'), x_ref)
np.save(os.path.join(args.log_dir, 'z_ref.npy'), z_ref)
np.save(os.path.join(args.log_dir, 'z.npy'), z)
np.save(os.path.join(args.log_dir, 'doms.npy'), doms)
np.save(os.path.join(args.log_dir, 'samples.npy'), samples)
np.save(os.path.join(args.log_dir, 'reconstructions.npy'), reconstructions)
np.save(os.path.join(args.log_dir, 'closest_samples.npy'), closest_samples)
np.save(os.path.join(args.log_dir, 'closest_dists.npy'), closest_dists)
