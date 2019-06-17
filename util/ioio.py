
import os
import tifffile as tiff
import h5py
import numpy as np
import torch

from util.tools import load_net
from util.preprocessing import normalize

# reads tif formatted file and returns the data in it as a numpy array
def read_tif(file, dtype='uint8'):

    data = tiff.imread(file).astype(dtype)

    return data

def write_tif(file, data):

    tiff.imwrite(file, data)

def read_hdf5(file, dtype='uint8', key=None):

    f = h5py.File(file, 'r')
    data = np.array(f.get(key), dtype=dtype)
    f.close()

    return data

def read_file(file, dtype='uint8', key=None):

    if file[-3:]=='tif': # tif file
        data = read_tif(file, dtype=dtype)
    elif file[-2:]=='h5' or file[-3:] == 'hdf': # hdf5 file
        data = read_hdf5(file, dtype=dtype, key=key)
    else: # unsupported file type
        return None

    return data

# utilization function that reads data
# data_path is either a single file, or a directory of a number of files (in which case the files are stacked)
# format of the data can be tif, or hdf5 (in which case a key has to be provided)
def read_data(data_path, dtype='uint8', key=None):

    if os.path.isdir(data_path): # data path is a directory, read in a sequence of data
        images = []
        files = os.listdir(data_path)
        files.sort()
        for file in files: # filter out tif files
            if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':
                images.append(file)

        # read the first image
        image = read_file(os.path.join(data_path, images[0]), dtype=dtype, key=key)
        sz = (len(images), ) + image.shape
        data = np.zeros(sz, dtype=dtype)
        for i, image in enumerate(images):
            data[i, ...] = read_file(os.path.join(data_path, image), dtype=dtype, key=key)
    else: # data path is a file
        data = read_file(data_path, dtype=dtype, key=key)

    return data

# write a 3D data set to a directory (slice by slice)
def imwrite3D(x, dir, prefix='', rescale=False, z_start=0):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(z_start, z_start+x.shape[0]):
        if rescale:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i-z_start,:,:] * 255).astype('uint8'))
        else:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i-z_start, :, :]).astype('uint8'))

# write out the activations of a segmentation network for a specific input
def write_activations(model_file, x, write_dir):

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    xn = normalize(x, np.max(x), np.max(x) - np.min(x))
    tiff.imsave(os.path.join(write_dir, 'input.tif'), (xn * 255).astype('uint8'))

    # transform data to cuda tensor
    x = torch.FloatTensor(x[np.newaxis, np.newaxis, ...]).cuda()

    # load network
    net = load_net(model_file=model_file)
    net.eval()
    net.cuda()

    # apply forward prop and extract network activations
    encoder_outputs, encoded_outputs = net.encoder(x)
    decoder_outputs, final_outputs = net.decoder(encoded_outputs, encoder_outputs)

    # write random activations
    for i, encoder_output in enumerate(encoder_outputs):
        c = np.random.randint(encoder_output.size(1))
        act = encoder_output[0, c, :, :].data.cpu().numpy()
        act = normalize(act, np.max(act), np.max(act) - np.min(act))
        tiff.imsave(os.path.join(write_dir, 'enc_act_'+str(len(encoder_outputs)-i-1)+'.tif'), (act * 255).astype('uint8'))

    c = np.random.randint(encoded_outputs.size(1))
    act = encoded_outputs[0, c, :, :].data.cpu().numpy()
    act = normalize(act, np.max(act), np.max(act) - np.min(act))
    tiff.imsave(os.path.join(write_dir, 'enc_act_'+str(len(encoder_outputs))+'.tif'), (act * 255).astype('uint8'))

    for i, decoder_output in enumerate(decoder_outputs):
        c = np.random.randint(decoder_output.size(1))
        act = decoder_output[0, c, :, :].data.cpu().numpy()
        act = normalize(act, np.max(act), np.max(act) - np.min(act))
        tiff.imsave(os.path.join(write_dir, 'dec_act_'+str(len(decoder_outputs)-i-1)+'.tif'), (act * 255).astype('uint8'))

def write_affinities(labels_dir, write_dir, spacing=1, dtype_in='float64', dtype_out='uint8', key='labels'):

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    files = os.listdir(labels_dir)
    files.sort()
    for i, file in enumerate(files):  # filter out data files
        if file[-3:] == 'tif' or file[-2:] == 'h5' or file[-3:] == 'hdf':

            print('Processing file %d/%d' % (i, len(files)))

            # load data file
            target = read_file(os.path.join(labels_dir, file), dtype=dtype_in, key=key)

            # compute affinities
            a = np.concatenate((target, target[:, :, -spacing:]), axis=2)
            b = np.concatenate((target[:, :, :spacing], target), axis=2)
            xdiff = (a == b)[:, :, spacing:]
            a = np.concatenate((target, target[:, -spacing:, :]), axis=1)
            b = np.concatenate((target[:, :spacing, :], target), axis=1)
            ydiff = (a == b)[:, spacing:, :]
            a = np.concatenate((target, target[-spacing:, :, :]), axis=0)
            b = np.concatenate((target[:spacing, :, :], target), axis=0)
            zdiff = (a == b)[spacing:, :, :]
            affinities = np.maximum(np.maximum(1 - xdiff, 1 - ydiff), 1 - zdiff)*255

            # write data file
            f = h5py.File(os.path.join(write_dir, file), 'w')
            f.create_dataset(name=key, data=affinities, dtype=dtype_out)
            f.close()