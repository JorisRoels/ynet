
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from util.preprocessing import normalize
from util.ioio import read_data, read_file
from util.tools import sample_unlabeled_input, sample_labeled_input

class StronglyLabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, label_path, input_shape=None, split = None, train = None, len_epoch = 1000,
                 preprocess = 'z', augmenters=None, dtypes = ('uint8', 'uint8'),
                 preload=True, keys=(None, None), scaling=None):

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.augmenters = augmenters
        self.preload = preload
        self.dtypes = dtypes
        self.keys = keys
        self.preprocess = preprocess
        self.scaling = scaling

        if preload:
            # data should be preloaded
            self.data = read_data(data_path, dtype=dtypes[0], key=keys[0])
            self.labels = read_data(label_path, dtype=dtypes[1], key=keys[1])

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
                self.data = F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[0,0,...].numpy()
                self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[0,0,...].numpy()

            self.mu, self.std = self.get_stats()
            self.labels = np.asarray(normalize(self.labels, 0, 255), dtype='uint8')

            if split is not None:
                if train:
                    s = int(split * self.data.shape[2])
                    self.data = self.data[:, :, :s]
                    self.labels = self.labels[:, :, :s]
                else:
                    s = int(split * self.data.shape[2])
                    self.data = self.data[:, :, s:]
                    self.labels = self.labels[:, :, s:]
        else:
            # assuming the images are in a directory now
            self.data_files = []
            files = os.listdir(data_path)
            files.sort()
            for file in files:  # filter out data files
                if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                    self.data_files.append(os.path.join(data_path, file))

            self.label_files = []
            files = os.listdir(label_path)
            files.sort()
            for file in files:  # filter out data files
                if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                    self.label_files.append(os.path.join(label_path, file))

    def __getitem__(self, i):

        # get random sample
        if self.preload:
            # data is already in memory
            input, target = sample_labeled_input(self.data, self.labels, self.input_shape)
        else:
            # load the sample in memory
            z = np.random.randint(0, len(self.data_files))
            input = read_file(self.data_files[z], dtype=self.dtypes[0], key=self.keys[0])
            target = read_file(self.label_files[z], dtype=self.dtypes[1], key=self.keys[1])

            # add z-axis if the data is 2D
            if len(input.shape) == 2:
                input, target = input[np.newaxis, ...], target[np.newaxis, ...]

            # rescale the sample if necessary
            if self.scaling is not None:
                target_size = np.asarray(np.multiply(input.shape, self.scaling), dtype=int)
                input = F.interpolate(torch.Tensor(input[np.newaxis, np.newaxis, ...]).cuda(), size=tuple(target_size), mode='area')[0,0,...].cpu().numpy()
                target = F.interpolate(torch.Tensor(target[np.newaxis, np.newaxis, ...]).cuda(), size=tuple(target_size), mode='area')[0,0,...].cpu().numpy()

            # if the input shape is specified, sample this from the data
            if self.input_shape is not None:
                input, target = sample_labeled_input(input, target, self.input_shape)

        # make sure the targets are binary
        target = np.asarray(target>0.5, dtype='uint8')

        # perform augmentation if necessary
        if self.augmenters is not None:
            input = self.augmenters[0](input)
            target = self.augmenters[1](target)

        # normalization
        if self.preprocess == 'z':
            if self.preload:
                mu = self.mu
                std = self.std
            else:
                mu = np.mean(input)
                std = np.std(input)
            input = normalize(input, mu, std)
        elif self.preprocess == 'unit':
            input = normalize(input, 0, 255)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class UnlabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape=None, len_epoch = 1000,
                 preprocess = 'z', augmenter=None, dtype = ('uint8'),
                 preload=True, key=None, scaling=None):

        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.augmenter = augmenter
        self.preload = preload
        self.dtype = dtype
        self.key = key
        self.preprocess = preprocess
        self.scaling = scaling

        if preload:
            # data should be preloaded
            self.data = read_data(data_path, dtype=dtype, key=key)

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
                self.data = F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[0,0,...].numpy()

            self.mu, self.std = self.get_stats()
        else:
            # assuming the images are in a directory now
            self.data_files = []
            files = os.listdir(data_path)
            files.sort()
            for file in files:  # filter out data files
                if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                    self.data_files.append(os.path.join(data_path, file))

    def __getitem__(self, i):

        # get random sample
        if self.preload:
            # data is already in memory
            input = sample_unlabeled_input(self.data, self.input_shape)
        else:
            # load the sample in memory
            z = np.random.randint(0, len(self.data_files))
            input = read_file(self.data_files[z], dtype=self.dtype, key=self.key)

            # add z-axis if the data is 2D
            if len(input.shape) == 2:
                input = input[np.newaxis, ...]

            # rescale the sample if necessary
            if self.scaling is not None:
                target_size = np.asarray(np.multiply(input.shape, self.scaling), dtype=int)
                input = F.interpolate(torch.Tensor(input[np.newaxis, np.newaxis, ...]).cuda(), size=tuple(target_size), mode='area')[0,0,...].cpu().numpy()

            # if the input shape is specified, sample this from the data
            if self.input_shape is not None:
                input = sample_unlabeled_input(input, self.input_shape)

        # perform augmentation if necessary
        if self.augmenter is not None:
            input = self.augmenter(input)

        # normalization
        if self.preprocess == 'z':
            if self.preload:
                mu = self.mu
                std = self.std
            else:
                mu = np.mean(input)
                std = np.std(input)
            input = normalize(input, mu, std)
        elif self.preprocess == 'unit':
            input = normalize(input, 0, 255)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std