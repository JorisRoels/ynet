
import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import imgaug as ia
from util.preprocessing import normalize
from util.ioio import read_data, read_file, write_tif
from util.tools import sample_unlabeled_input, sample_labeled_input, sample_labeled_input_embryo, sample_unlabeled_input_embryo

class StronglyLabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, label_path, input_shape=None, split = None, train = None, len_epoch = 1000,
                 preprocess = 'z', augmenter=None, dtypes = ('uint8', 'uint8'),
                 preload=True, keys=(None, None), scaling=None, fold=None, n_folds=5):

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.augmenter = augmenter
        self.preload = preload
        self.dtypes = dtypes
        self.keys = keys
        self.preprocess = preprocess
        self.scaling = scaling
        self.train = train
        self.fold = fold
        self.n_folds = n_folds

        if preload:
            # data should be preloaded
            self.data = read_data(data_path, dtype=dtypes[0], key=keys[0])
            self.labels = read_data(label_path, dtype=dtypes[1], key=keys[1])
            if fold is not None:
                if train:
                    self.data, self.labels, _, _ = self.select_folds(self.data, self.labels, fold, n_folds=self.n_folds)
                else:
                    _, _, self.data, self.labels = self.select_folds(self.data, self.labels, fold, n_folds=self.n_folds)

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
            fold_size = len(self.data_files) // self.n_folds
            if self.train is not None: # cross validation
                if self.train:
                    z = np.random.randint(self.fold*fold_size, (self.fold+1)*fold_size)
                else:
                    z = np.random.randint(0, (self.n_folds - 1)*fold_size)
                    if self.fold*fold_size < z and z < (self.fold+1)*fold_size:
                        z += fold_size
            else:
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
        if self.augmenter is not None:
            z, y, x = input.shape
            images = []
            labels = []
            for k in range(z):
                images.append(input[k, ...])
                labels.append(ia.SegmentationMapOnImage(target[k, ...], shape=target[k, ...].shape, nb_classes=2))
            augmenter_det = self.augmenter.to_deterministic()
            images_aug = augmenter_det.augment_images(images)
            labels_aug = augmenter_det.augment_segmentation_maps(labels)
            for k in range(z):
                input[k, ...] = images_aug[k]
                target[k, ...] = labels_aug[k].arr[..., 1]

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

    def select_folds(self, data, labels, fold, n_folds=5):

        fold_size = data.shape[0] // n_folds

        if fold == 0:
            return data[:fold_size,...], labels[:fold_size,...], data[fold_size:,...], labels[fold_size:,...]
        elif fold == n_folds-1:
            return data[fold*fold_size:,...], labels[fold*fold_size:,...], data[:fold*fold_size,...], labels[:fold*fold_size,...]
        else:
            data_train = data[fold*fold_size:(fold+1)*fold_size, ...]
            labels_train = labels[fold*fold_size:(fold+1)*fold_size, ...]
            data_test = np.concatenate((data[:fold*fold_size,...], data[(fold+1)*fold_size:,...]), axis=0)
            labels_test = np.concatenate((labels[:fold*fold_size,...], labels[(fold+1)*fold_size:,...]), axis=0)
            return data_train, labels_train, data_test, labels_test

class UnlabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape, len_epoch=1000, preprocess='unit', transform=None, dtype = 'uint8',
                 preload=True, key=None, scaling=None):

        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
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
                self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[0,0,...].numpy()

            mu, std = self.get_stats()
            self.mu = mu
            self.std = std
            if preprocess == 'z':
                self.data = normalize(self.data, mu, std)
            elif preprocess == 'unit':
                self.data = normalize(self.data, 0, 255)
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

            # normalization
            if self.preprocess == 'z':
                mu = np.mean(input)
                std = np.std(input)
                input = normalize(input, mu, std)
            elif self.preprocess == 'unit':
                input = normalize(input, 0, 255)

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
        if self.transform is not None:
            input = self.transform(input)

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

class StronglyLabeledEmbryoDataset(data.Dataset):

    def __init__(self, data_path_membranes, data_path_nuclei, label_path, align_file=None, input_shape=None, len_epoch = 1000,
                 preprocess = 'z', transform = None, target_transform = None, dtypes = ('uint8', 'uint8', 'uint8'),
                 preload=True, keys=('image', 'image', 'labels')):

        self.data_path_membranes = data_path_membranes
        self.data_path_nuclei = data_path_nuclei
        self.label_path = label_path
        self.align_file = align_file
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload
        self.dtypes = dtypes
        self.keys = keys
        self.preprocess = preprocess

        # assuming the images are in a directory now
        self.data_files = []
        self.data = []
        files = os.listdir(data_path_membranes)
        files.sort()
        for file in files:  # filter out data files
            if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                self.data_files.append((os.path.join(data_path_membranes, file), os.path.join(data_path_nuclei, file)))

                # read data if necessary
                if self.preload:
                    membrane_input = read_file(os.path.join(data_path_membranes, file), dtype=self.dtypes[0], key=self.keys[0])
                    nuclei_input = read_file(os.path.join(data_path_nuclei, file), dtype=self.dtypes[1], key=self.keys[1])
                    self.data.append((membrane_input, nuclei_input))

        self.label_files = []
        self.labels = []
        files = os.listdir(label_path)
        files.sort()
        for file in files:  # filter out data files
            if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                self.label_files.append(os.path.join(label_path, file))

                # read data if necessary
                if self.preload:
                    target = read_file(os.path.join(label_path, file), dtype=self.dtypes[2], key=self.keys[2])
                    self.labels.append(target)

        if align_file is not None:
            with open(align_file, "r") as f:
                self.align = json.load(f)

    def __getitem__(self, i):

        # get random sample
        z = np.random.randint(0, len(self.data_files))
        if self.preload:
            membrane_input, nuclei_input = self.data[z]
            target = self.labels[z]
        else:
            membrane_input = read_file(self.data_files[z][0], dtype=self.dtypes[0], key=self.keys[0])
            nuclei_input = read_file(self.data_files[z][1], dtype=self.dtypes[1], key=self.keys[1])
            target = read_file(self.label_files[z], dtype=self.dtypes[2], key=self.keys[2])

        # normalization
        if self.preprocess == 'z':
            n = np.prod(membrane_input.shape) * 2
            mu = (np.sum(membrane_input) + np.sum(nuclei_input)) / n
            std = (np.sum(membrane_input + nuclei_input - 2*mu)) / n
            membrane_input = normalize(membrane_input, mu, std)
            nuclei_input = normalize(nuclei_input, mu, std)
        elif self.preprocess == 'unit':
            membrane_input = normalize(membrane_input, 0, 255)
            nuclei_input = normalize(nuclei_input, 0, 255)
        target = normalize(target, 0, 255)

        # align the data
        if self.align_file is not None:
            align = self.align[os.path.basename(self.data_files[z][0])]

            abs_align = np.abs(align)
            nuclei_input = np.roll(np.pad(nuclei_input, ((abs_align, abs_align),(0, 0), (0, 0)), mode='constant'),
                                   -align, axis=0)[abs_align:(-abs_align), :, :]

        # stack inputs
        input = np.stack((membrane_input, nuclei_input))

        # if the input shape is specified, sample this from the data
        if self.input_shape is not None:
            nnz = False
            while not nnz: # sample until we've found a non-trivial (i.e. non-zero) target
                sample = sample_labeled_input_embryo(input, target, self.input_shape)
                if np.sum(sample[1]) > 0: # found a non-zero target
                    input, target = sample
                    nnz = True

        # perform augmentation if necessary
        if self.transform is not None:
            if len(input.shape) == 4: # 3D
                c, z, y, x = input.shape
                input = np.reshape(input, (c*z, y, x))
                input = self.transform(input)
                input = np.reshape(input, (c, z, y, x))
            else:
                input = self.transform(input)
        if self.target_transform is not None and len(target)>0:
            target = np.asarray(target, dtype='float64')
            if len(target.shape) == 4: # 3D
                c, z, y, x = target.shape
                target = np.reshape(target, (c*z, y, x))
                target = self.target_transform(target)
                target = np.reshape(target, (c, z, y, x))
            else:
                target = self.target_transform(target)

        return input, target

    def __len__(self):

        return self.len_epoch

class UnlabeledEmbryoDataset(data.Dataset):

    def __init__(self, data_path_membranes, data_path_nuclei, align_file=None, input_shape=None, len_epoch = 1000,
                 preprocess = 'z', transform = None, preload=True, dtypes = ('uint8', 'uint8'), keys=('image', 'image')):

        self.data_path_membranes = data_path_membranes
        self.data_path_nuclei = data_path_nuclei
        self.align_file = align_file
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.preload = preload
        self.dtypes = dtypes
        self.keys = keys
        self.preprocess = preprocess

        # assuming the images are in a directory now
        self.data_files = []
        self.data = []
        files = os.listdir(data_path_membranes)
        files.sort()
        for file in files:  # filter out data files
            if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                self.data_files.append((os.path.join(data_path_membranes, file), os.path.join(data_path_nuclei, file)))

                # read data if necessary
                if self.preload:
                    membrane_input = read_file(os.path.join(data_path_membranes, file), dtype=self.dtypes[0], key=self.keys[0])
                    nuclei_input = read_file(os.path.join(data_path_nuclei, file), dtype=self.dtypes[1], key=self.keys[1])
                    self.data.append((membrane_input, nuclei_input))

        if align_file is not None:
            with open(align_file, "r") as f:
                self.align = json.load(f)

    def __getitem__(self, i):

        # get random sample
        z = np.random.randint(0, len(self.data_files))
        if self.preload:
            membrane_input, nuclei_input = self.data[z]
        else:
            membrane_input = read_file(self.data_files[z][0], dtype=self.dtypes[0], key=self.keys[0])
            nuclei_input = read_file(self.data_files[z][1], dtype=self.dtypes[1], key=self.keys[1])

        # normalization
        if self.preprocess == 'z':
            membrane_input = normalize(membrane_input, np.mean(membrane_input), np.std(membrane_input))
            nuclei_input = normalize(nuclei_input, np.mean(nuclei_input), np.std(nuclei_input))
        elif self.preprocess == 'unit':
            membrane_input = normalize(membrane_input, 0, 255)
            nuclei_input = normalize(nuclei_input, 0, 255)

        # align the data
        if self.align_file is not None:
            align = self.align[os.path.basename(self.data_files[z][0])]

            abs_align = np.abs(align)
            nuclei_input = np.roll(np.pad(nuclei_input, ((abs_align, abs_align),(0, 0), (0, 0)), mode='constant'),
                                   -align, axis=0)[abs_align:(-abs_align), :, :]

        # stack inputs
        input = np.stack((membrane_input, nuclei_input))

        # if the input shape is specified, sample this from the data
        if self.input_shape is not None:
            input = sample_unlabeled_input_embryo(input, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            if len(input.shape) == 4: # 3D
                c, z, y, x = input.shape
                input = np.reshape(input, (c*z, y, x))
                input = self.transform(input)
                input = np.reshape(input, (c, z, y, x))
            else:
                input = self.transform(input)

        return input

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class WeaklyLabeledSlideScannerDataset(data.Dataset):

    def __init__(self, data_path, label_path, input_shape=None, len_epoch = 1000,
                 preprocess = 'z', augmenter=None):

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.preprocess = preprocess
        self.augmenter = augmenter

        self.data = read_file(data_path)
        self.labels = read_file(label_path)

        self.mu, self.std = self.get_stats()

    def __getitem__(self, i):

        # get random sample
        np.random.seed()

        # generate random position
        x = np.random.randint(0, self.data.shape[0]-self.input_shape[0]+1)
        y = np.random.randint(0, self.data.shape[1]-self.input_shape[1]+1)

        # extract input and target patch
        input = np.asarray(copy.copy(self.data[x:x+self.input_shape[0], y:y+self.input_shape[1], :]), dtype='float')
        target = copy.copy(self.labels[x:x+self.input_shape[0], y:y+self.input_shape[1]])

        # make sure the targets are binary
        target = int(np.sum(target)>0.0)

        # perform augmentation if necessary
        if self.augmenter is not None:
            input = self.augmenter.augment_images(input)

        # normalization
        if self.preprocess == 'z':
            input = normalize(input, self.mu, self.std)
        elif self.preprocess == 'unit':
            input = normalize(input, 0, 255)

        return np.transpose(input, (2,0,1)), target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class UnlabeledSlideScannerDataset(data.Dataset):

    def __init__(self, data_path, input_shape=None, len_epoch = 1000,
                 preprocess = 'z', augmenter=None):

        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.preprocess = preprocess
        self.augmenter = augmenter

        self.data = read_file(data_path)

        self.mu, self.std = self.get_stats()

    def __getitem__(self, i):

        # get random sample
        np.random.seed()

        # generate random position
        x = np.random.randint(0, self.data.shape[0]-self.input_shape[0]+1)
        y = np.random.randint(0, self.data.shape[1]-self.input_shape[1]+1)

        # extract input and target patch
        input = np.asarray(copy.copy(self.data[x:x+self.input_shape[0], y:y+self.input_shape[1], :]), dtype='float')

        # perform augmentation if necessary
        if self.augmenter is not None:
            input = self.augmenter.augment_images(input)

        # normalization
        if self.preprocess == 'z':
            input = normalize(input, self.mu, self.std)
        elif self.preprocess == 'unit':
            input = normalize(input, 0, 255)

        return np.transpose(input, (2,0,1))

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std
