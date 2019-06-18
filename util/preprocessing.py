
import numpy as np
import numpy.random as rnd
import torch
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

class ToFloatTensor(object):

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array
        :return: a tensor containing the same data as a float
        """
        return torch.FloatTensor(x.copy())

class ToLongTensor(object):

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array or float
        :return: a tensor or scalar containing the same data as a long
        """
        if isinstance(x, float):
            return torch.LongTensor([x.copy()])[0]
        else:
            return torch.LongTensor(x.copy())

class AddChannelAxis(object):

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array
        :return: the same array with an additional axis as first dimension
        """
        return x[np.newaxis, ...]

class AddNoise(object):

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=20.0):
        """

        :param prob: probability of adding noise
        :param sigma_min: minimum standard deviation of noise
        :param sigma_max: maximum standard deviation of noise
        """
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array
        :return: the numpy array with additional gaussian noise
        """
        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            return x + rnd.normal(0, sigma, x.shape)
        else:
            return x

class Normalize(object):

    def __init__(self, mu=0, std=1):
        """

        :param mu: mean for the normalization
        :param std: standard deviation for the normalization
        """
        self.mu = mu
        self.std = std

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array
        :return: the normalized array
        """
        return (x-self.mu)/self.std

class RandomDeformations2D(object):

    def __init__(self, prob=0.5, scale=(0.01, 0.05), seed=None, thr=False):
        """

        :param prob: probability of deforming
        :param scale: scale of the deformation
        :param seed: seed for the deformation
        :param thr: threshold to obtain binary outputs if necessary
        """
        self.prob = prob
        self.scale = scale
        self.augmenter = iaa.PiecewiseAffine(scale=self.scale)
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0,2**32)
        self.thr = thr

    def __call__(self, x):
        """

        :param x: a 2-dimensional numpy array
        :return: a deformed image
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand() < self.prob:
            s = rnd.randint(0,2**32)
            self.augmenter.reseed(s)
            x_aug = self.augmenter.augment_images(x[np.newaxis, ...].copy())[0, ...]
            if self.thr:
                x_aug = np.asarray(x_aug>0.5, float) # avoid weird noise artifacts
            return x_aug
        else:
            return x

class RandomDeformations3D(object):

    def __init__(self, prob=0.5, scale=(0.01, 0.05), seed=None, thr=False):
        """

        :param prob: probability of deforming
        :param scale: scale of the deformation
        :param seed: seed for the deformation
        :param thr: threshold to obtain binary outputs if necessary
        """
        self.prob = prob
        self.scale = scale
        self.augmenter = iaa.PiecewiseAffine(scale=self.scale)
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0,2**32)
        self.thr = thr

    def __call__(self, x):
        """

        :param x: a 3-dimensional numpy array (the first dimension is assumed to be the z dimension)
        :return: a deformed image
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand() < self.prob:
            s = rnd.randint(0, 2**32)  # fixed seed so that the same deformation is applied along z
            x_aug = np.zeros_like(x)
            for i in range(x.shape[0]):
                self.augmenter.reseed(s)
                x_aug[i:i+1,...] = self.augmenter.augment_images(x[i:i+1,...].copy())
            if self.thr:
                x_aug = np.asarray(x_aug>0.5, float) # avoid weird noise artifacts
            return x_aug
        else:
            return x

class Rotate2D(object):

    def __init__(self, prob=0.5, seed=None):
        """

        :param prob: probability of rotating
        :param seed: seed for the rotation
        """
        self.prob = prob
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0, 2**32)

    def __call__(self, x):
        """

        :param x: a 2 or 3-dimensional numpy array
        :return: a randomly rotated image (only 90 degree angles in the xy plane)
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            if x.ndim == 3:
                return np.rot90(x.copy(), k=rnd.randint(1,4), axes=(1, 2))
            else: # assuming 2-dimensional input now
                return np.rot90(x.copy(), k=rnd.randint(1, 4), axes=(0, 1))
        else:
            return x

class Rotate3D(object):

    def __init__(self, prob=0.5, axes=(1,2), seed=None):
        """

        :param prob: probability of rotating
        :param seed: seed for the rotation
        """
        self.prob = prob
        self.axes = axes
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0,2**32)

    def __call__(self, x):
        """

        :param x: a 3-dimensional numpy array (the first dimension is assumed to be the z dimension)
        :return: a randomly rotated image (only 90 degree angles)
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            # rotate in the xy plane
            x_ = np.rot90(x.copy(), k=rnd.randint(1,4), axes=(1,2))
            # rotate in the yz plane
            return np.rot90(x_.copy(), k=rnd.randint(1, 4), axes=(0, 1))
        else:
            return x

class FlipX(object):

    def __init__(self, prob=0.5, seed=None):
        """

        :param prob: probability of flipping
        :param seed: seed for the flip
        """
        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):
        """

        :param x: a 2 or 3-dimensional numpy array
        :return: a flipped version along the x axis
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            if x.ndim == 3:
                return x[:,:,::-1]
            else: # assuming 2-dimensional input now
                return x[:,::-1]
        else:
            return x

class FlipY(object):

    def __init__(self, prob=0.5, seed=None):
        """

        :param prob: probability of flipping
        :param seed: seed for the flip
        """
        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):
        """

        :param x: a 2 or 3-dimensional numpy array
        :return: a flipped version along the y axis
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            if x.ndim == 3:
                return x[:,::-1,:]
            else: # assuming 2-dimensional input now
                return x[::-1,:]
        else:
            return x

class FlipZ(object):

    def __init__(self, prob=0.5, seed=None):
        """

        :param prob: probability of flipping
        :param seed: seed for the flip
        """
        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):
        """

        :param x: a 3-dimensional numpy array
        :return: a flipped version along the z axis
        """
        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            return x[::-1,:,:]
        else:
            return x

class Flatten(object):

    def __call__(self, x):
        """

        :param x: a d-dimensional numpy array
        :return: a vectorized 1-dimensional array
        """
        return x.reshape(-1)

def get_augmenter(deformation_scale=(0.01, 0.05), sigma_max=0.1):

    augmenter = iaa.Sequential([
        iaa.Sequential([
            iaa.OneOf(
                [iaa.Affine(rotate=90),
                 iaa.Affine(rotate=180),
                 iaa.Affine(rotate=270),
                 iaa.Affine(rotate=0)]),
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Sometimes(1, iaa.PiecewiseAffine(scale=deformation_scale))
        ], random_order=True),
            iaa.Sometimes(0, iaa.AdditiveGaussianNoise(scale=sigma_max * 255))
    ])

    return augmenter

def get_augmenters_2d(augment_noise=True):
    # standard augmenters: rotation, flips, deformations

    # generate seeds for synchronized augmentation
    s1 = np.random.randint(0, 2 ** 32)
    s2 = np.random.randint(0, 2 ** 32)
    s3 = np.random.randint(0, 2 ** 32)
    s5 = np.random.randint(0, 2 ** 32)

    # define transforms
    if augment_noise:
        train_xtransform = transforms.Compose([Rotate2D(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               RandomDeformations2D(seed=s5),
                                               AddNoise(sigma_max=0.2),
                                               ToFloatTensor()])
    else:
        train_xtransform = transforms.Compose([Rotate2D(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               RandomDeformations2D(seed=s5),
                                               ToFloatTensor()])
    train_ytransform = transforms.Compose([Rotate2D(seed=s1),
                                           FlipX(seed=s2),
                                           FlipY(seed=s3),
                                           RandomDeformations2D(seed=s5, thr=True),
                                           ToLongTensor()])
    test_xtransform = transforms.Compose([ToFloatTensor()])
    test_ytransform = transforms.Compose([ToLongTensor()])

    return train_xtransform, train_ytransform, test_xtransform, test_ytransform

def get_augmenters_3d(augment_noise=True):
    # standard augmenters: rotation, flips, deformations

    # generate seeds for synchronized augmentation
    s1 = np.random.randint(0, 2 ** 32)
    s2 = np.random.randint(0, 2 ** 32)
    s3 = np.random.randint(0, 2 ** 32)
    s4 = np.random.randint(0, 2 ** 32)
    s5 = np.random.randint(0, 2 ** 32)

    # define transforms
    if augment_noise:
        train_xtransform = transforms.Compose([Rotate3D(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               FlipZ(seed=s4),
                                               RandomDeformations3D(seed=s5),
                                               AddNoise(sigma_max=0.2),
                                               ToFloatTensor()])
    else:
        train_xtransform = transforms.Compose([Rotate3D(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               FlipZ(seed=s4),
                                               RandomDeformations3D(seed=s5),
                                               ToFloatTensor()])
    train_ytransform = transforms.Compose([Rotate3D(seed=s1),
                                           FlipX(seed=s2),
                                           FlipY(seed=s3),
                                           FlipZ(seed=s4),
                                           RandomDeformations3D(seed=s5, thr=True),
                                           ToLongTensor()])
    test_xtransform = transforms.Compose([ToFloatTensor()])
    test_ytransform = transforms.Compose([ToLongTensor()])

    return train_xtransform, train_ytransform, test_xtransform, test_ytransform

def get_train_augmenters_3d(rotate2D=True, rotate3D=False, flipx=True, flipy=True, flipz=True, deform=True, noise=True):
    # standard augmenters: rotation, flips, deformations, noise

    # generate seeds for synchronized augmentation
    s1 = np.random.randint(0, 2 ** 32)
    s2 = np.random.randint(0, 2 ** 32)
    s3 = np.random.randint(0, 2 ** 32)
    s4 = np.random.randint(0, 2 ** 32)
    s5 = np.random.randint(0, 2 ** 32)
    s6 = np.random.randint(0, 2 ** 32)

    # define transforms
    tx = transforms.Compose([])
    ty = transforms.Compose([])
    if rotate2D and not rotate3D:
        tx = transforms.Compose([tx, Rotate2D(seed=s1)])
        ty = transforms.Compose([ty, Rotate2D(seed=s1)])
    if rotate3D:
        tx = transforms.Compose([tx, Rotate3D(seed=s2)])
        ty = transforms.Compose([ty, Rotate3D(seed=s2)])
    if flipx:
        tx = transforms.Compose([tx, FlipX(seed=s3)])
        ty = transforms.Compose([ty, FlipX(seed=s3)])
    if flipy:
        tx = transforms.Compose([tx, FlipY(seed=s4)])
        ty = transforms.Compose([ty, FlipY(seed=s4)])
    if flipz:
        tx = transforms.Compose([tx, FlipZ(seed=s5)])
        ty = transforms.Compose([ty, FlipZ(seed=s5)])
    if deform:
        tx = transforms.Compose([tx, RandomDeformations3D(seed=s6)])
        ty = transforms.Compose([ty, RandomDeformations3D(seed=s6, thr=True)])
    if noise:
        tx = transforms.Compose([tx, AddNoise()])
    tx = transforms.Compose([tx, ToFloatTensor()])
    ty = transforms.Compose([ty, ToLongTensor()])

    return tx, ty

def get_test_augmenters_3d():
    # define transforms
    test_xtransform = transforms.Compose([ToFloatTensor()])
    test_ytransform = transforms.Compose([ToLongTensor()])

    return test_xtransform, test_ytransform

def normalize(x, mu, std):
    return (x-mu)/std
