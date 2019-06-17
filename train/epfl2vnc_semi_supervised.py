
"""
    This is a script that trains the Y-Net in a semi-supervised fashion
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import UnlabeledVolumeDataset, StronglyLabeledVolumeDataset
from networks.ynet import YNetFT
from util.losses import CrossEntropyLoss, MSELoss
from util.preprocessing import get_augmenter
from util.validation import validate

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="epfl2vnc")
parser.add_argument("--write_dir", help="Writing directory", type=str, default="epfl2vnc")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=50)

# network parameters
parser.add_argument("--data_dir", help="Data directory", type=str, default="../../data")
parser.add_argument("--train_frac", help="Fraction of training data to use", type=float, default=0.1)
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="256,256")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)",
                    type=int, default=4)
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.5)

# regularization parameters
parser.add_argument('--lambda_rec', help='Regularization parameters for Y-Net reconstruction', type=float, default=1e-2)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=200)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=5)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=1)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
loss_fn_seg = CrossEntropyLoss()
loss_fn_rec = MSELoss()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
# load source
print('[%s] Loading data' % (datetime.datetime.now()))
# augmenters
augmenter = get_augmenter()
# load data
res_src = (1,5,5)
res_tar = (1,4.6,4.6)
scaling = np.divide(res_src, res_tar)
n_folds = int(1/args.train_frac)
src_train_labeled = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/data.tif'),
                                                 os.path.join(args.data_dir, 'EM/EPFL/labels.tif'),
                                                 input_shape=input_shape, augmenter=augmenter, preprocess='unit', scaling=scaling)
tar_train_unlabeled = UnlabeledVolumeDataset(os.path.join(args.data_dir, 'EM/VNC/data_larger.tif'), input_shape=input_shape,
                                             preprocess='unit')
tar_train_labeled = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/VNC/data.tif'),
                                                 os.path.join(args.data_dir, 'EM/VNC/mito_labels.tif'),
                                                 train=True, fold=0, n_folds=n_folds, input_shape=input_shape,
                                                 augmenter=augmenter, preprocess='unit')
tar_test_labeled = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/VNC/data.tif'),
                                                os.path.join(args.data_dir, 'EM/VNC/mito_labels.tif'),
                                                train=False, fold=0, n_folds=n_folds, input_shape=input_shape, preprocess='unit')
src_train_labeled_loader = DataLoader(src_train_labeled, batch_size=args.train_batch_size)
tar_train_unlabeled_loader = DataLoader(tar_train_unlabeled, batch_size=args.train_batch_size)
tar_train_labeled_loader = DataLoader(tar_train_labeled, batch_size=args.train_batch_size)
tar_test_labeled_loader = DataLoader(tar_test_labeled, batch_size=args.test_batch_size)

"""
    Build the network
"""
print('[%s] Building the network' % (datetime.datetime.now()))
net = YNetFT(feature_maps=args.fm, levels=args.levels, dropout=args.dropout, lambda_rec=args.lambda_rec)

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
net.train_net(train_loader_source_labeled=src_train_labeled_loader, train_loader_target_unlabeled=tar_train_unlabeled_loader,
              train_loader_target_labeled=tar_train_labeled_loader, test_data=tar_test_labeled.data/255, test_labels=tar_test_labeled.labels,
              optimizer=optimizer, loss_seg_fn=loss_fn_seg, loss_rec_fn=loss_fn_rec,
              scheduler=scheduler, epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)


"""
    Validate the trained network
"""
validate(net, tar_test_labeled.data/255, tar_test_labeled.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.write_dir, 'segmentation_final'),
         val_file=os.path.join(args.log_dir, 'validation_final.npy'))
net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
validate(net, tar_test_labeled.data/255, tar_test_labeled.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.write_dir, 'segmentation_best'),
         val_file=os.path.join(args.log_dir, 'validation_best.npy'))

print('[%s] Finished!' % (datetime.datetime.now()))