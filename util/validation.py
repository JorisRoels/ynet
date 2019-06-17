
import os
import json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.tools import gaussian_window
from util.metrics import jaccard, dice, accuracy_metrics
from util.ioio import imwrite3D, read_file
from util.preprocessing import normalize

# sliding window iterator
def sliding_window(image, step_size, window_size):

    # define range
    zrange = [0]
    while zrange[-1] < image.shape[0] - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = image.shape[0] - window_size[0]
    yrange = [0]
    while yrange[-1] < image.shape[1] - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = image.shape[1] - window_size[1]
    xrange = [0]
    while xrange[-1] < image.shape[2] - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = image.shape[2] - window_size[2]

    # loop over the range
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # yield the current window
                yield (z, y, x, image[z:z+window_size[0], y:y+window_size[1], x:x+window_size[2]])

# sliding window iterator for multichannel images
# image is assumed to have dimensions (num_channels, z_dim, y_dim, x_dim)
def sliding_window_multichannel(image, step_size, window_size):

    # define range
    zrange = [0]
    while zrange[-1] < image.shape[1] - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = image.shape[1] - window_size[0]
    yrange = [0]
    while yrange[-1] < image.shape[2] - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = image.shape[2] - window_size[1]
    xrange = [0]
    while xrange[-1] < image.shape[3] - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = image.shape[3] - window_size[2]

    # loop over the range
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # yield the current window
                if window_size[0] == 1: # 2D
                    yield (z, y, x, image[:, z, y:y+window_size[1], x:x+window_size[2]])
                else: # 3D
                    yield (z, y, x, image[:, z:z+window_size[0], y:y+window_size[1], x:x+window_size[2]])

# sliding window iterator with loading
def loadsliding_window(image_dir, step_size, max_pixels = 512**3,
                       dtype = 'uint8', key=None):

    # get image dimensions
    image_files = []
    files = os.listdir(image_dir)
    files.sort()
    for file in files:  # filter out data files
        if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
            image_files.append(os.path.join(image_dir, file))

    image = read_file(image_files[0], dtype=dtype, key=key)
    zdim, ydim, xdim = len(image_files), image.shape[0], image.shape[1]
    zstep = max_pixels // (xdim*ydim)
    window_size = (zstep, ydim, xdim)

    # define range
    zrange = [0]
    while zrange[-1] < zdim - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = zdim - window_size[0]
    yrange = [0]
    while yrange[-1] < ydim - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = ydim - window_size[1]
    xrange = [0]
    while xrange[-1] < xdim - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = xdim - window_size[2]

    # loop over the range
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # load the current window
                image_window = np.zeros(window_size)
                for z_ind in range(z, z+window_size[0]):
                    image_window[z_ind] = read_file(image_files[z_ind], dtype=dtype, key=key)[y:y+window_size[1], x:x+window_size[2]]

                # yield the current window
                yield (z, y, x, image_window)

# sliding window iterator with loading
def loadsliding_labeled_window(image_dir, label_dir, max_pixels = 512**3,
                               dtypes = ('uint8', 'uint8'), keys=(None, None)):

    # get image dimensions
    image_files = []
    files = os.listdir(image_dir)
    files.sort()
    for file in files:  # filter out data files
        if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
            image_files.append(os.path.join(image_dir, file))

    label_files = []
    files = os.listdir(label_dir)
    files.sort()
    for file in files:  # filter out data files
        if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
            label_files.append(os.path.join(label_dir, file))

    image = read_file(image_files[0], dtype=dtypes[0], key=keys[0])
    zdim, ydim, xdim = len(image_files), image.shape[0], image.shape[1]
    zstep = max_pixels // (xdim*ydim)
    window_size = (zstep, ydim, xdim)

    # define range
    zrange = [0]
    while zrange[-1] < zdim - window_size[0]:
        zrange.append(zrange[-1] + window_size[0])
    zrange[-1] = zdim - window_size[0]
    yrange = [0]
    while yrange[-1] < ydim - window_size[1]:
        yrange.append(yrange[-1] + window_size[1])
    yrange[-1] = ydim - window_size[1]
    xrange = [0]
    while xrange[-1] < xdim - window_size[2]:
        xrange.append(xrange[-1] + window_size[2])
    xrange[-1] = xdim - window_size[2]

    # loop over the range
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # load the current window
                image_window = np.zeros(window_size)
                label_window = np.zeros(window_size)
                for z_ind in range(z, z+window_size[0]):
                    image_window[z_ind-z, ...] = read_file(image_files[z_ind], dtype=dtypes[0], key=keys[0])[y:y+window_size[1], x:x+window_size[2]]
                    label_window[z_ind-z, ...] = read_file(label_files[z_ind], dtype=dtypes[0], key=keys[0])[y:y+window_size[1], x:x+window_size[2]]

                # yield the current window
                yield (z, y, x, image_window, label_window)

# # segment a data set with a given network with a sliding window
# # data is assumed a 3D volume
# # input shape should come as
# #   - 2D (y_dim, x_dim)
# #   - 3D (z_dim, y_dim, x_dim)
# def segment(data, net, input_shape, batch_size=1, in_channels=1, step_size=None):
#
#     # make sure we compute everything on the gpu and in evaluation mode
#     net.cuda()
#     net.eval()
#
#     # 2D or 3D
#     is2d = len(input_shape) == 2
#
#     # set step size to half of the window if necessary
#     if step_size == None:
#         if is2d:
#             step_size = (1, input_shape[0]//2, input_shape[1]//2)
#         else:
#             step_size = (input_shape[0]//2, input_shape[1]//2, input_shape[2]//2)
#
#     # gaussian window for smooth block merging
#     if is2d:
#         g_window = gaussian_window((1,input_shape[0],input_shape[1]), sigma=input_shape[-1]/4)
#     else:
#         g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)
#
#     # symmetric extension only necessary along z-axis if multichannel 2D inputs
#     if is2d and in_channels>1:
#         z_pad = in_channels // 2
#         padding = ((z_pad, z_pad), (0, 0), (0, 0))
#         data = np.pad(data, padding, mode='symmetric')
#     else:
#         z_pad = 0
#
#     # allocate space
#     seg_cum = np.zeros(data.shape)
#     counts_cum = np.zeros(data.shape)
#
#     # define sliding window
#     if is2d:
#         sw = sliding_window(data, step_size=step_size, window_size=(in_channels, input_shape[0],input_shape[1]))
#     else:
#         sw = sliding_window(data, step_size=step_size, window_size=input_shape)
#
#     # start prediction
#     batch_counter = 0
#     if is2d:
#         batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1]))
#     else:
#         batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1], input_shape[2]))
#     positions = np.zeros((batch_size, 3), dtype=int)
#     for (z, y, x, inputs) in sw:
#
#         # fill batch
#         if not is2d: # add channel in case of 3D processing, in 2D case, it's already there
#             inputs = inputs[np.newaxis, ...]
#         batch[batch_counter, ...] = inputs
#         positions[batch_counter, :] = [z, y, x]
#
#         # increment batch counter
#         batch_counter += 1
#
#         # perform segmentation when a full batch is filled
#         if batch_counter == batch_size:
#
#             # convert to tensors
#             inputs = torch.FloatTensor(batch).cuda()
#
#             # forward prop
#             outputs = net(inputs)[-1]
#             outputs = F.softmax(outputs, dim=1)
#
#             # cumulate segmentation volume
#             for b in range(batch_size):
#                 (z_b, y_b, x_b) = positions[b, :]
#                 # take into account the gaussian filtering
#                 if is2d:
#                     seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
#                         np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
#                     counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
#                 else:
#                     seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
#                         np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
#                     counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window
#
#             # reset batch counter
#             batch_counter = 0
#
#     # don't forget last batch
#     # convert to tensors
#     inputs = torch.FloatTensor(batch).cuda()
#
#     # forward prop
#     outputs = net(inputs)[-1]
#     outputs = F.softmax(outputs, dim=1)
#
#     # cumulate segmentation volume
#     for b in range(batch_counter):
#         (z_b, y_b, x_b) = positions[b, :]
#         # take into account the gaussian filtering
#         if is2d:
#             seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
#                 np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
#             counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
#         else:
#             seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
#                 np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
#             counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window
#
#     # crop out the symmetric extension and compute segmentation
#     segmentation = np.divide(seg_cum[0:counts_cum.shape[0]-2*z_pad, :, :],
#                              counts_cum[0:counts_cum.shape[0] - 2*z_pad, :, :])
#
#     return segmentation

# segment a data set with a given network with a sliding window
# data is assumed a 3D volume
# input shape should come as
#   - 2D (y_dim, x_dim)
#   - 3D (z_dim, y_dim, x_dim)
def segment(data, net, input_shape, batch_size=1, step_size=None):

    return segment_multichannel(data[np.newaxis, ...], net, input_shape,
                                batch_size=batch_size, step_size=step_size)

# segment a data set with a given network with a sliding window
# data is assumed a multichannel 3D volume
# input shape should come as
#   - 2D (y_dim, x_dim)
#   - 3D (z_dim, y_dim, x_dim)
def segment_multichannel(data, net, input_shape, batch_size=1, step_size=None):

    # make sure we compute everything on the gpu and in evaluation mode
    net.cuda()
    net.eval()

    channels = data.shape[0]

    # 2D or 3D
    is2d = len(input_shape) == 2

    # set step size to half of the window if necessary
    if step_size == None:
        if is2d:
            step_size = (1, input_shape[0]//2, input_shape[1]//2)
        else:
            step_size = (input_shape[0]//2, input_shape[1]//2, input_shape[2]//2)

    # gaussian window for smooth block merging
    if is2d:
        g_window = gaussian_window((1,input_shape[0],input_shape[1]), sigma=input_shape[-1]/4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)

    # allocate space
    seg_cum = np.zeros(data.shape[1:])
    counts_cum = np.zeros(data.shape[1:])

    # define sliding window
    if is2d:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=(1, input_shape[0],input_shape[1]))
    else:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=input_shape)

    # start prediction
    batch_counter = 0
    if is2d:
        batch = np.zeros((batch_size, channels, input_shape[0], input_shape[1]))
    else:
        batch = np.zeros((batch_size, channels, input_shape[0], input_shape[1], input_shape[2]))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:

            # convert to tensors
            inputs = torch.FloatTensor(batch).cuda()

            # forward prop
            outputs = net(inputs)
            if type(outputs) is tuple:
                outputs = outputs[-1]
            outputs = F.softmax(outputs, dim=1)

            # cumulate segmentation volume
            for b in range(batch_size):
                (z_b, y_b, x_b) = positions[b, :]
                # take into account the gaussian filtering
                if is2d:
                    seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
                    counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
                else:
                    seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
                    counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

            # reset batch counter
            batch_counter = 0

    # don't forget last batch
    # convert to tensors
    inputs = torch.FloatTensor(batch).cuda()

    # forward prop
    outputs = net(inputs)
    if type(outputs) is tuple:
        outputs = outputs[-1]
    outputs = F.softmax(outputs, dim=1)

    # cumulate segmentation volume
    for b in range(batch_counter):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

    # crop out the symmetric extension and compute segmentation
    segmentation = np.divide(seg_cum, counts_cum)

    return segmentation

# segment a data set directory with a given network with a sliding window
# data is assumed a 3D volume and processed in blocks, according to block_size (a 3-tuple)
# segmentation result is written to a directory
# input shape should come as
#   - 2D (y_dim, x_dim)
#   - 3D (z_dim, y_dim, x_dim)
def segment_block(data_dir, seg_dir, net, input_shape, batch_size=1, in_channels=1, step_size=None):

    # make sure we compute everything on the gpu and in evaluation mode
    net.cuda()
    net.eval()

    # define sliding window
    sw = loadsliding_window(data_dir)

    # start prediction
    for (z, y, x, block) in sw:

        # segment block
        segmentation = segment(block, net, input_shape, batch_size=batch_size, in_channels=in_channels, step_size=step_size)

        # write out the block
        imwrite3D(segmentation, seg_dir, rescale=True)

# segment a data set with a given network with a sliding window
# data is assumed a 3D volume
# input shape should come as
#   - 2D (y_dim, x_dim)
#   - 3D (z_dim, y_dim, x_dim)
def transform(data, net, input_shape, batch_size=1, in_channels=1, step_size=None):

    # make sure we compute everything on the gpu and in evaluation mode
    net.cuda()
    net.eval()

    # 2D or 3D
    is2d = len(input_shape) == 2

    # upsampling might be necessary depending on the network
    interp = nn.Upsample(size=input_shape, mode='bilinear', align_corners=True)

    # set step size to half of the window if necessary
    if step_size == None:
        if is2d:
            step_size = (1, input_shape[0]//2, input_shape[1]//2)
        else:
            step_size = (input_shape[0]//2, input_shape[1]//2, input_shape[2]//2)

    # gaussian window for smooth block merging
    if is2d:
        g_window = gaussian_window((1,input_shape[0],input_shape[1]), sigma=input_shape[-1]/4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)

    # symmetric extension only necessary along z-axis if multichannel 2D inputs
    if is2d and in_channels>1:
        z_pad = in_channels // 2
        padding = ((z_pad, z_pad), (0, 0), (0, 0))
        data = np.pad(data, padding, mode='symmetric')
    else:
        z_pad = 0

    # allocate space
    transf_cum = np.zeros(data.shape)
    counts_cum = np.zeros(data.shape)

    # define sliding window
    if is2d:
        sw = sliding_window(data, step_size=step_size, window_size=(in_channels, input_shape[0],input_shape[1]))
    else:
        sw = sliding_window(data, step_size=step_size, window_size=input_shape)

    # start prediction
    batch_counter = 0
    if is2d:
        batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1]))
    else:
        batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1], input_shape[2]))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        if not is2d: # add channel in case of 3D processing, in 2D case, it's already there
            inputs = inputs[np.newaxis, ...]
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:

            # convert to tensors
            inputs = torch.FloatTensor(batch).cuda()

            # forward prop
            outputs = net(inputs)
            if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
                outputs = interp(outputs)
            outputs = torch.sigmoid(outputs)

            # cumulate segmentation volume
            for b in range(batch_size):
                (z_b, y_b, x_b) = positions[b, :]
                # take into account the gaussian filtering
                if is2d:
                    transf_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 0:1, :, :])
                    counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
                else:
                    transf_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 0, ...])
                    counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

            # reset batch counter
            batch_counter = 0

    # don't forget last batch
    # convert to tensors
    inputs = torch.FloatTensor(batch).cuda()

    # forward prop
    outputs = net(inputs)
    if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
        outputs = interp(outputs)
    outputs = torch.sigmoid(outputs)

    # cumulate segmentation volume
    for b in range(batch_counter):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            transf_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 0:1, :, :])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            transf_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 0, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

    # crop out the symmetric extension and compute segmentation
    transform = np.divide(transf_cum[0:counts_cum.shape[0] - 2*z_pad, :, :],
                          counts_cum[0:counts_cum.shape[0] - 2*z_pad, :, :])

    return transform

def validate(net, data, labels, input_size, batch_size=1, write_dir=None, val_file=None,
             dtypes = ('uint8', 'uint8'), keys=(None, None), spacing=[1], writer=None, epoch=0):
    print('[%s] Validating the trained network' % (datetime.datetime.now()))

    if write_dir is not None and not os.path.exists(write_dir):
        os.mkdir(write_dir)

    if isinstance(data, str): # data is a directory and should be processed in blocks
        # define sliding window
        sw = loadsliding_labeled_window(data, labels, dtypes=dtypes, keys=keys)
        # start prediction
        eps = 1e-10
        tp = tn = fp = fn = x_cum = y_cum = 0
        for (z, y, x, block, labels) in sw:
            # segment block
            segmentation = segment(block/255, net, input_size, batch_size=batch_size)
            # cumulate validation statistics
            x = segmentation > 0.5
            y = labels > 0.5
            tp += np.sum(np.multiply(x, y))
            tn += np.sum(np.multiply(1-x, 1-y))
            fp += np.sum(np.multiply(x, 1-y))
            fn += np.sum(np.multiply(1-x, y))
            x_cum += np.sum(x)
            y_cum += np.sum(y)
            # write out the block
            if write_dir is not None:
                print('[%s] Writing the output %d' % (datetime.datetime.now(), z))
                imwrite3D(segmentation, write_dir, rescale=True, z_start=z)
        j = tp / (x_cum + y_cum - tp)
        d = (2*tp) / (x_cum + y_cum)
        a = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        p = (tp + eps) / (tp + fp + eps)
        r = (tp + eps) / (tp + fn + eps)
        f = (2 * (p * r) + eps) / (p + r + eps)
    elif isinstance(data, tuple): # data is a tuple of membrane/nuclei directories
        membrane_dir, nuclei_dir = data
        label_dir = labels

        # read every file separately
        j_cum = d_cum = a_cum = p_cum = r_cum = f_cum = 0
        cnt = 0
        membrane_files = os.listdir(membrane_dir)
        membrane_files.sort()
        nuclei_files = os.listdir(nuclei_dir)
        nuclei_files.sort()
        label_files = os.listdir(label_dir)
        label_files.sort()
        for i, file in enumerate(membrane_files):  # filter out data files
            if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                cnt += 1
                # print('processing file ' + file + '...')

                # read data
                membrane_input = read_file(os.path.join(membrane_dir, file), dtype=dtypes[0], key=keys[0])
                nuclei_input = read_file(os.path.join(nuclei_dir, file), dtype=dtypes[1], key=keys[1])
                labels = read_file(os.path.join(label_dir, file), dtype=dtypes[2], key=keys[2])

                # normalize data
                membrane_input = normalize(membrane_input, 0, 255)
                nuclei_input = normalize(nuclei_input, 0, 255)

                # stack inputs
                input = np.stack((membrane_input, nuclei_input))

                # segmentation
                segmentation = segment_multichannel(input, net, input_size, batch_size=batch_size)

                # validation
                j_cum += jaccard(segmentation, labels)
                d_cum += dice(segmentation, labels)
                a, p, r, f = accuracy_metrics(segmentation, labels)
                a_cum += a; p_cum += p; r_cum += r; f_cum += f

                # write output
                if write_dir is not None:
                    print('[%s] Writing the output' % (datetime.datetime.now()))
                    imwrite3D(segmentation, os.path.join(write_dir, file[:-3]), rescale=True)

        j = j_cum / cnt; d = d_cum / cnt; a = a_cum / cnt; p = p_cum / cnt; r = r_cum / cnt; f = f_cum / cnt;

    else: # data is a numpy array, so can be segmented straightforwardly
        segmentation = segment(data, net, input_size, batch_size=batch_size)
        j = jaccard(segmentation, labels)
        d = dice(segmentation, labels)
        a, p, r, f = accuracy_metrics(segmentation, labels)
        if write_dir is not None:
            print('[%s] Writing the output' % (datetime.datetime.now()))
            imwrite3D(segmentation, write_dir, rescale=True)
        if writer is not None:
            z = data.shape[0]//2
            N = 1024
            if data.shape[1] > N:
                writer.add_image('target/input', data[z:z+1, :N, :N], epoch)
                writer.add_image('target/segmentation', segmentation[z:z+1, :N, :N], epoch)
            else:
                writer.add_image('target/input', data[z:z+1, ...], epoch)
                writer.add_image('target/segmentation', segmentation[z:z+1, ...], epoch)
    print('[%s] Network performance: Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
    if val_file is not None:
        np.save(val_file, np.asarray([a, p, r, f, j, d]))
    return a, p, r, f, j, d

def validate_classifier(net, data, labels, input_size=128, batch_size=1, n_samples=1000, write_dir=None, val_file=None, writer=None, epoch=0):

    print('[%s] Validating the trained network' % (datetime.datetime.now()))

    if write_dir is not None and not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # randomize seed
    np.random.seed()

    n_batches = n_samples // batch_size
    y = np.zeros((n_batches*batch_size))
    y_pred = np.zeros((n_batches*batch_size))
    for b in range(n_batches):

        # build a batch
        inputs = np.zeros((batch_size, data.shape[2], input_size, input_size))
        for i in range(batch_size):

            # generate random position
            x = np.random.randint(0, data.shape[0]-input_size+1)
            y_ = np.random.randint(0, data.shape[1]-input_size+1)

            # extract input and target patch
            input = data[x:x+input_size, y_:y_+input_size, :]
            target = labels[x:x+input_size, y_:y_+input_size]

            # fill the batch
            inputs[i, ...] = np.transpose(input, (2, 0, 1))
            y[b*batch_size+i] = np.sum(target)>0

        # run the network on the batch
        _, pred = net(torch.Tensor(inputs).cuda())
        pred = F.softmax(pred, dim=1)

        # save the results
        y_pred[b*batch_size:(b+1)*batch_size] = np.argmax(pred.cpu().data.numpy(), axis=1)

    j = jaccard(y_pred, y)
    d = dice(y_pred, y)
    a, p, r, f = accuracy_metrics(y_pred, y)

    print('[%s] Network performance: Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
    if val_file is not None:
        np.save(val_file, np.asarray([a, p, r, f, j, d]))

    return a, p, r, f, j, d