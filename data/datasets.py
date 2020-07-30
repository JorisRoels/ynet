import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from neuralnets.util.io import read_volume
from neuralnets.util.tools import sample_unlabeled_input, sample_labeled_input, normalize


def _orient(data, orientation=0):
    """
    This function essentially places the desired orientation axis to that of the original Z-axis
    For example:
          (Z, Y, X) -> (Y, Z, X) for orientation=1
          (Z, Y, X) -> (X, Y, Z) for orientation=2
    Note that applying this function twice corresponds to the identity transform

    :param data: assumed to be of shape (Z, Y, X)
    :param orientation: 0, 1 or 2 (respectively for Z, Y or X axis)
    :return: reoriented data sample
    """
    if orientation == 1:
        return np.transpose(data, axes=(1, 0, 2))
    elif orientation == 2:
        return np.transpose(data, axes=(2, 1, 0))
    else:
        return data


def _validate_shape(input_shape, data_shape, orientation=0, in_channels=1, levels=4):
    """
    Validates an input for propagation through a U-Net by taking into account the following:
        - Sampling along different orientations
        - Sampling multiple adjacent slices as channels
        - Maximum size that can be sampled from the data

    :param input_shape: original shape of the sample (Z, Y, X)
    :param data_shape: shape of the data to sample from (Z, Y, X)
    :param orientation: orientation to sample along
    :param in_channels: sample multiple adjacent slices as channels
    :param levels: amount of pooling layers in the network
    :return: the validated input shape
    """

    # make sure input shape can be edited
    input_shape = list(input_shape)

    # sample adjacent slices if necessary
    is2d = input_shape[0] == 1
    if is2d and in_channels > 1:
        input_shape[0] = in_channels

    # transform the data shape and input shape according to the orientation
    if orientation == 1:  # (Z, Y, X) -> (Y, Z, X)
        input_shape = [input_shape[1], input_shape[0], input_shape[2]]
    elif orientation == 2:  # (Z, Y, X) -> (X, Y, Z)
        input_shape = [input_shape[2], input_shape[1], input_shape[0]]

    # make sure the input shape fits in the data shape: i.e. find largest k such that n of the form n=k*2**levels
    for d in range(3):
        if not (is2d and d == orientation) and input_shape[d] > data_shape[d]:
            # 2D case: X, Y - 3D case: X, Y, Z
            # note we assume that the data has a least in_channels elements in each dimension
            input_shape[d] = int((data_shape[d] // (2 ** levels)) * (2 ** levels))

    # and return as a tuple
    return tuple(input_shape)


class VolumeDataset(data.Dataset):
    """
    Dataset for volumes

    :param data_path: path to the dataset
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional split_orientation: orientation to train/test split the data along
    :param optional split_location: location where the train/test data should be split
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional train: train or test data
    """

    def __init__(self, data_path, input_shape, split_orientation='z', split_location=0.50, scaling=None, len_epoch=1000,
                 type='tif3d', in_channels=1, orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit',
                 train=True):
        self.data_path = data_path
        self.input_shape = input_shape
        self.split_orientation = split_orientation
        self.split_location = split_location
        self.scaling = scaling
        self.len_epoch = len_epoch
        self.in_channels = in_channels
        self.orientations = orientations
        self.orientation = 0
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.train = train

        # load the data
        d = 0 if split_orientation == 'z' else 1 if split_orientation == 'y' else 2
        if split_orientation == 'z':
            split = int(len(os.listdir(data_path)) * split_location)
            start = 0 if train else split
            stop = split if train else -1
            self.data = read_volume(data_path, type=type, dtype=dtype, start=start, stop=stop)
        else:
            data = read_volume(data_path, type=type, dtype=dtype)
            split = int(data.shape[d] * split_location)
            if split_orientation == 'y':
                self.data = data[:, :split, :] if train else data[:, split:, :]
            else:
                self.data = data[:, :, :split] if train else data[:, :, split:]

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def _get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

    def _select_orientation(self):
        self.orientation = np.random.choice(self.orientations)


class StronglyLabeledVolumeDataset(VolumeDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional split_orientation: orientation to train/test split the data along
    :param optional split_location: location where the train/test data should be split
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional train: train or test data
    """

    def __init__(self, data_path, label_path, input_shape=None, split_orientation='z', split_location=0.50,
                 scaling=None, len_epoch=1000, type='tif3d', coi=(0, 1), in_channels=1, orientations=(0,), batch_size=1,
                 data_dtype='uint8', label_dtype='uint8', norm_type='unit', train=True):
        super().__init__(data_path, input_shape, split_orientation=split_orientation, split_location=split_location,
                         scaling=scaling, len_epoch=len_epoch, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=data_dtype, norm_type=norm_type,
                         train=train)

        self.label_path = label_path
        self.coi = coi

        # load labels
        d = 0 if split_orientation == 'z' else 1 if split_orientation == 'y' else 2
        if split_orientation == 'z':
            split = int(len(os.listdir(label_path)) * split_location)
            start = 0 if train else split
            stop = split if train else -1
            self.labels = read_volume(label_path, type=type, dtype=label_dtype, start=start, stop=stop)
        else:
            data = read_volume(label_path, type=type, dtype=label_dtype)
            split = int(data.shape[d] * split_location)
            if split_orientation == 'y':
                self.labels = data[:, :split, :] if train else data[:, split:, :]
            else:
                self.labels = data[:, :, :split] if train else data[:, :, split:]

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                        mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)
        target = _orient(target, orientation=self.orientation)

        if self.input_shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if len(np.intersect1d(np.unique(target),
                              self.coi)) == 0:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledVolumeDataset(VolumeDataset):
    """
    Dataset for unlabeled volumes

    :param data_path: path to the dataset
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional split_orientation: orientation to train/test split the data along
    :param optional split_location: location where the train/test data should be split
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional train: train or test data
    """

    def __init__(self, data_path, input_shape=None, split_orientation='z', split_location=0.50, scaling=None,
                 len_epoch=1000, type='tif3d', in_channels=1, orientations=(0,), batch_size=1, dtype='uint8',
                 norm_type='unit', train=True):
        super().__init__(data_path, input_shape, split_orientation=split_orientation, split_location=split_location,
                         scaling=scaling, len_epoch=len_epoch, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=dtype, norm_type=norm_type,
                         train=train)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input = sample_unlabeled_input(self.data, input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)

        if self.input_shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input
