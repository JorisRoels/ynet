import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralnets.networks.unet import UNetEncoder2D, UNetDecoder2D, UNet2D, UNetEncoder3D, UNetDecoder3D, UNet3D
from neuralnets.networks.cnn import CNN2D
from neuralnets.util.metrics import jaccard, accuracy_metrics
from neuralnets.util.losses import DiceLoss
from neuralnets.util.tools import module_to_device, tensor_to_device, log_scalars, log_images_2d, log_images_3d, \
    augment_samples, get_labels
from neuralnets.util.io import print_frm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function

SEGMENTATION = 0
RECONSTRUCTION = 1
JOINT = 2


class ReverseLayerF(Function):
    """
    Gradient reversal layer (https://arxiv.org/abs/1505.07818)
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None


# 2D W-Net model for domain adaptive segmentation
class WNet2D(nn.Module):

    def __init__(self, in_channels=1, coi=(0, 1), feature_maps=64, levels=4, norm='instance', lambda_rec=1e-3,
                 lambda_dc=1e-1, dropout_enc=0.0, dropout_dec=0.0, activation='relu', sigma_noise=0.25,
                 input_size=(1, 256, 256), conv_channels=(16, 16, 16, 16, 16), fc_channels=(128, 32)):
        super(WNet2D, self).__init__()

        self.in_channels = in_channels
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.lambda_rec = lambda_rec
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.dropout_dec = dropout_dec
        self.train_mode = JOINT
        self.sigma_noise = sigma_noise
        # self.seg_loss = nn.CrossEntropyLoss()
        self.seg_loss = DiceLoss()
        self.rec_loss = nn.MSELoss()
        self.dc_loss = nn.CrossEntropyLoss()
        self.lambda_rec = lambda_rec
        self.lambda_dc = lambda_dc
        self.input_size = input_size
        self.conv_channels = conv_channels
        fc_channels = tuple(np.asarray([*fc_channels, 2]).astype('int'))
        self.fc_channels = fc_channels
        self.p = 0.5

        # reconstruction encoder
        self.rec_encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                         dropout=dropout_enc, activation=activation)

        # reconstruction decoder
        self.rec_decoder = UNetDecoder2D(out_channels=in_channels, feature_maps=feature_maps, levels=levels,
                                         norm=norm, dropout=dropout_dec, activation=activation)

        # segmentation encoder
        self.seg_encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                         dropout=dropout_enc, activation=activation)

        # segmentation decoder
        self.seg_decoder = UNetDecoder2D(out_channels=self.out_channels, feature_maps=feature_maps, levels=levels,
                                         norm=norm, dropout=dropout_dec, activation=activation)

        # domain classifiers
        self.rec_dc = CNN2D(conv_channels, fc_channels, (1, *input_size), norm="batch")
        self.seg_dc = CNN2D(conv_channels, fc_channels, (2, *input_size), norm="batch")

    def forward_rec(self, x, add_noise=True):

        # add noise to the inputs if necessary
        if add_noise:
            sigma = self.sigma_noise * torch.randn(1, device=x.device)
            x_ = x + sigma * torch.rand(x.size(), device=x.device)
        else:
            x_ = x

        # contractive path
        encoder_outputs, encoded = self.rec_encoder(x_)

        # reconstruction decoder
        _, x_rec = self.rec_decoder(encoded, encoder_outputs)
        x_rec = torch.sigmoid(x_rec)

        # gradient reversal on the predicted image
        x_rec_rev = ReverseLayerF.apply(x_rec)
        rec_dom_pred = self.rec_dc(x_rec_rev)

        return x_rec, rec_dom_pred

    def forward_seg(self, x):

        # contractive path
        encoder_outputs, encoded = self.seg_encoder(x)

        # segmentation decoder
        _, y_pred = self.seg_decoder(encoded, encoder_outputs)

        # gradient reversal on the predicted segmentation
        y_pred_rev = ReverseLayerF.apply(torch.softmax(y_pred, dim=1))
        seg_dom_pred = self.seg_dc(y_pred_rev)

        return y_pred, seg_dom_pred

    def forward(self, x):

        if self.train_mode == SEGMENTATION:
            return self.forward_seg(x)
        elif self.train_mode == RECONSTRUCTION:
            return self.forward_rec(x, add_noise=True)
        else:
            x_rec, x_rec_dom = self.forward_rec(x, add_noise=False)
            y_pred, y_pred_dom = self.forward_seg(x_rec)
            return x_rec, y_pred, x_rec_dom, y_pred_dom

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.seg_encoder.in_channels, coi=self.coi, feature_maps=self.seg_encoder.feature_maps,
                     levels=self.seg_encoder.levels, skip_connections=self.seg_decoder.skip_connections,
                     norm=self.seg_encoder.norm, activation=self.seg_encoder.activation,
                     dropout_enc=self.seg_encoder.dropout, dropout_dec=self.seg_decoder.dropout)

        net.encoder.load_state_dict(self.seg_encoder.state_dict())
        net.decoder.load_state_dict(self.seg_decoder.state_dict())

        return net

    def train_epoch(self, loader_src, loader_tar_ul, loader_tar_l, optimizer, epoch, augmenter=None, print_stats=1,
                    writer=None, write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader_src: source dataloader (labeled)
        :param loader_tar_ul: target dataloader (unlabeled)
        :param loader_tar_l: target dataloader (labeled)
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter: data augmenter
        :param print_stats: frequency of printing statistics
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
        loss_dc_x_cum = 0.0
        loss_dc_y_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # zip dataloaders
        if loader_tar_l is None:
            dl = zip(loader_src, loader_tar_ul)
        else:
            dl = zip(loader_src, loader_tar_ul, loader_tar_l)

        # start epoch
        time_start = datetime.datetime.now()
        for i, data in enumerate(dl):

            # transfer to suitable device
            data_src = tensor_to_device(data[0], device)
            x_tar_ul = tensor_to_device(data[1], device)
            if loader_tar_l is not None:
                data_tar_l = tensor_to_device(data[2], device)

            # augment if necessary
            if loader_tar_l is None:
                data_aug = (data_src[0], x_tar_ul, data_src[1], x_tar_ul)
                x_src, x_tar_ul, y_src, _ = augment_samples(data_aug, augmenter=augmenter)
            else:
                data_aug = (data_src[0], x_tar_ul, data_tar_l[0], data_src[1], x_tar_ul, data_tar_l[1])
                x_src, x_tar_ul, x_tar_l, y_src, _, y_tar_l = augment_samples(data_aug, augmenter=augmenter)
                y_tar_l = get_labels(y_tar_l, coi=self.coi, dtype=int)
            y_src = get_labels(y_src, coi=self.coi, dtype=int)
            x_tar_ul = x_tar_ul.float()

            # zero the gradient buffers
            self.zero_grad()

            # get domain labels for domain confusion
            dom_labels_x = tensor_to_device(torch.zeros((x_src.size(0) + x_tar_ul.size(0))), device).long()
            dom_labels_x[x_src.size(0):] = 1
            dom_labels_y = tensor_to_device(torch.zeros((x_src.size(0) + x_tar_ul.size(0))), device).long()

            # check train mode and compute loss
            loss_seg_src = torch.Tensor([0])
            loss_seg_tar = torch.Tensor([0])
            loss_rec_src = torch.Tensor([0])
            loss_rec_tar = torch.Tensor([0])
            loss_dc_x = torch.Tensor([0])
            loss_dc_y = torch.Tensor([0])
            if self.train_mode == RECONSTRUCTION:
                x_src_rec, x_src_rec_dom = self.forward_rec(x_src)
                x_tar_ul_rec, x_tar_ul_rec_dom = self.forward_rec(x_tar_ul)
                loss_rec_src = self.rec_loss(x_src_rec, x_src)
                loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
                loss_dc_x = self.dc_loss(torch.cat((x_src_rec_dom, x_tar_ul_rec_dom), dim=0), dom_labels_x)
                total_loss = loss_rec_src + loss_rec_tar + self.lambda_dc * loss_dc_x
            elif self.train_mode == SEGMENTATION:
                # switch between reconstructed and original inputs
                if np.random.rand() < self.p:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src)
                else:
                    x_src_rec, _ = self.forward_rec(x_src)
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src_rec)
                    dom_labels_y[:x_src.size(0)] = 1
                if np.random.rand() < self.p:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul)
                else:
                    x_tar_ul_rec, _ = self.forward_rec(x_tar_ul)
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul_rec)
                    dom_labels_y[x_src.size(0):] = 1
                loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
                loss_dc_y = self.dc_loss(torch.cat((y_src_pred_dom, y_tar_ul_pred_dom), dim=0), dom_labels_y)
                total_loss = loss_seg_src + self.lambda_dc * loss_dc_y
                if loader_tar_l is not None:
                    y_tar_l_pred, _ = self.forward_seg(x_tar_l)
                    loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                    total_loss = total_loss + loss_seg_tar
            else:
                x_src_rec, x_src_rec_dom = self.forward_rec(x_src)
                if np.random.rand() < self.p:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src)
                else:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src_rec)
                    dom_labels_y[:x_src.size(0)] = 1
                x_tar_ul_rec, x_tar_ul_rec_dom = self.forward_rec(x_tar_ul)
                if np.random.rand() < self.p:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul)
                else:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul_rec)
                    dom_labels_y[x_src.size(0):] = 1
                loss_rec_src = self.rec_loss(x_src_rec, x_src)
                loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
                loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
                loss_dc_x = self.dc_loss(torch.cat((x_src_rec_dom, x_tar_ul_rec_dom), dim=0), dom_labels_x)
                loss_dc_y = self.dc_loss(torch.cat((y_src_pred_dom, y_tar_ul_pred_dom), dim=0), dom_labels_y)
                total_loss = loss_seg_src + self.lambda_rec * (loss_rec_src + loss_rec_tar) + \
                             self.lambda_dc * (loss_dc_x + loss_dc_y)
                if loader_tar_l is not None:
                    _, y_tar_l_pred, _, y_tar_l_pred_dom = self(x_tar_l)
                    loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                    total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
            loss_dc_x_cum += loss_dc_x.data.cpu().numpy()
            loss_dc_y_cum += loss_dc_y.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print(
                    '[%s] Epoch %5d - Iteration %5d/%5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss DCX: %.6f - Loss DCY: %.6f - Loss: %.6f'
                    % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size,
                       loss_seg_src_cum / cnt, loss_seg_tar_cum / cnt, loss_rec_src_cum / cnt, loss_rec_tar_cum / cnt,
                       loss_dc_x_cum / cnt, loss_dc_y_cum / cnt, total_loss_cum / cnt))

        # keep track of time
        runtime = datetime.datetime.now() - time_start
        seconds = runtime.total_seconds()
        hours = seconds // 3600
        minutes = (seconds - hours * 3600) // 60
        seconds = seconds - hours * 3600 - minutes * 60
        print_frm(
            'Epoch %5d - Runtime for training: %d hours, %d minutes, %f seconds' % (epoch, hours, minutes, seconds))

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / cnt
        loss_seg_tar_avg = loss_seg_tar_cum / cnt
        loss_rec_src_avg = loss_rec_src_cum / cnt
        loss_rec_tar_avg = loss_rec_tar_cum / cnt
        loss_dc_x_avg = loss_dc_x_cum / cnt
        loss_dc_y_avg = loss_dc_y_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Training Epoch %4d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss DCX: %.6f - Loss DCY: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg,
               loss_dc_x_avg, loss_dc_y_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            if self.train_mode == RECONSTRUCTION:
                log_scalars([loss_rec_src_avg, loss_rec_tar_avg, loss_dc_x_avg],
                            ['train/' + s for s in ['loss-rec-src', 'loss-rec-tar', 'loss-dc-x']], writer, epoch=epoch)
            elif self.train_mode == SEGMENTATION:
                log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_dc_y_avg],
                            ['train/' + s for s in ['loss-seg-src', 'loss-seg-tar', 'loss-dc-y']], writer, epoch=epoch)
            else:
                log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg, loss_dc_x_avg,
                             loss_dc_y_avg], ['train/' + s for s in
                                              ['loss-seg-src', 'loss-seg-tar', 'loss-rec-src', 'loss-rec-tar',
                                               'loss-dc-x', 'loss-dc-y']], writer, epoch=epoch)
            log_scalars([total_loss_avg], ['train/' + s for s in ['total-loss']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                log_images_2d([x_src.data], ['train/' + s for s in ['src/x']], writer, epoch=epoch)
                if self.train_mode == RECONSTRUCTION:
                    log_images_2d([x_src_rec.data, x_tar_ul.data, x_tar_ul_rec.data],
                                  ['train/' + s for s in ['src/x-rec', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                                  epoch=epoch)
                elif self.train_mode == SEGMENTATION:
                    y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([y_src.data, y_src_pred], ['train/' + s for s in ['src/y', 'src/y-pred']], writer,
                                  epoch=epoch)
                    if loader_tar_l is not None:
                        y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                        log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                      ['train/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer,
                                      epoch=epoch)
                else:
                    y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([x_src_rec.data, y_src.data, y_src_pred, x_tar_ul.data, x_tar_ul_rec.data],
                                  ['train/' + s for s in
                                   ['src/x-rec', 'src/y', 'src/y-pred', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                                  epoch=epoch)
                    if loader_tar_l is not None:
                        y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                        log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                      ['train/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer,
                                      epoch=epoch)

        return total_loss_avg

    def test_epoch(self, loader_src, loader_tar_ul, loader_tar_l, epoch, writer=None, write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader_src: source dataloader (labeled)
        :param loader_tar_ul: target dataloader (unlabeled)
        :param loader_tar_l: target dataloader (labeled)
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.eval()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
        loss_dc_x_cum = 0.0
        loss_dc_y_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # zip dataloaders
        dl = zip(loader_src, loader_tar_ul, loader_tar_l)

        # start epoch
        y_preds = []
        ys = []
        time_start = datetime.datetime.now()
        for i, data in enumerate(dl):

            # transfer to suitable device
            x_src, y_src = tensor_to_device(data[0], device)
            x_tar_ul = tensor_to_device(data[1], device)
            x_tar_l, y_tar_l = tensor_to_device(data[2], device)
            x_src = x_src.float()
            x_tar_ul = x_tar_ul.float()
            x_tar_l = x_tar_l.float()
            y_src = y_src.long()
            y_tar_l = y_tar_l.long()

            # get domain labels for domain confusion
            dom_labels_x = tensor_to_device(torch.zeros((x_src.size(0) + x_tar_ul.size(0))), device).long()
            dom_labels_x[x_src.size(0):] = 1
            dom_labels_y = tensor_to_device(torch.zeros((x_src.size(0) + x_tar_ul.size(0))), device).long()

            # check train mode and compute loss
            loss_seg_src = torch.Tensor([0])
            loss_seg_tar = torch.Tensor([0])
            loss_rec_src = torch.Tensor([0])
            loss_rec_tar = torch.Tensor([0])
            loss_dc_x = torch.Tensor([0])
            loss_dc_y = torch.Tensor([0])
            if self.train_mode == RECONSTRUCTION:
                x_src_rec, x_src_rec_dom = self.forward_rec(x_src)
                x_tar_ul_rec, x_tar_ul_rec_dom = self.forward_rec(x_tar_ul)
                loss_rec_src = self.rec_loss(x_src_rec, x_src)
                loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
                loss_dc_x = self.dc_loss(torch.cat((x_src_rec_dom, x_tar_ul_rec_dom), dim=0), dom_labels_x)
                total_loss = loss_rec_src + loss_rec_tar + self.lambda_dc * loss_dc_x
            elif self.train_mode == SEGMENTATION:
                # switch between reconstructed and original inputs
                if np.random.rand() < self.p:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src)
                else:
                    x_src_rec, _ = self.forward_rec(x_src)
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src_rec)
                    dom_labels_y[:x_src.size(0)] = 1
                if np.random.rand() < self.p:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul)
                else:
                    x_tar_ul_rec, _ = self.forward_rec(x_tar_ul)
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul_rec)
                    dom_labels_y[x_src.size(0):] = 1
                loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
                loss_dc_y = self.dc_loss(torch.cat((y_src_pred_dom, y_tar_ul_pred_dom), dim=0), dom_labels_y)
                total_loss = loss_seg_src + self.lambda_dc * loss_dc_y
                y_tar_l_pred, _ = self.forward_seg(x_tar_l)
                loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                total_loss = total_loss + loss_seg_tar
            else:
                x_src_rec, x_src_rec_dom = self.forward_rec(x_src)
                if np.random.rand() < self.p:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src)
                else:
                    y_src_pred, y_src_pred_dom = self.forward_seg(x_src_rec)
                    dom_labels_y[:x_src.size(0)] = 1
                x_tar_ul_rec, x_tar_ul_rec_dom = self.forward_rec(x_tar_ul)
                if np.random.rand() < self.p:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul)
                else:
                    y_tar_ul_pred, y_tar_ul_pred_dom = self.forward_seg(x_tar_ul_rec)
                    dom_labels_y[x_src.size(0):] = 1
                loss_rec_src = self.rec_loss(x_src_rec, x_src)
                loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
                loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
                loss_dc_x = self.dc_loss(torch.cat((x_src_rec_dom, x_tar_ul_rec_dom), dim=0), dom_labels_x)
                loss_dc_y = self.dc_loss(torch.cat((y_src_pred_dom, y_tar_ul_pred_dom), dim=0), dom_labels_y)
                total_loss = loss_seg_src + self.lambda_rec * (loss_rec_src + loss_rec_tar) + \
                             self.lambda_dc * (loss_dc_x + loss_dc_y)
                _, y_tar_l_pred, _, y_tar_l_pred_dom = self(x_tar_l)
                loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
            loss_dc_x_cum += loss_dc_x.data.cpu().numpy()
            loss_dc_y_cum += loss_dc_y.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            if self.train_mode == SEGMENTATION or self.train_mode == JOINT:
                for b in range(y_tar_l_pred.size(0)):
                    y_preds.append(
                        F.softmax(y_tar_l_pred, dim=1)[b, ...].view(y_tar_l_pred.size(1), -1).data.cpu().numpy())
                    ys.append(y_tar_l[b, 0, ...].flatten().cpu().numpy())

        # keep track of time
        runtime = datetime.datetime.now() - time_start
        seconds = runtime.total_seconds()
        hours = seconds // 3600
        minutes = (seconds - hours * 3600) // 60
        seconds = seconds - hours * 3600 - minutes * 60
        print_frm(
            'Epoch %5d - Runtime for testing: %d hours, %d minutes, %f seconds' % (epoch, hours, minutes, seconds))

        # prep for metric computation
        if self.train_mode == SEGMENTATION or self.train_mode == JOINT:
            y_preds = np.concatenate(y_preds, axis=1)
            ys = np.concatenate(ys)
            js = np.asarray([jaccard((ys == i).astype(int), y_preds[i, :]) for i in range(len(self.coi))])
            ams = np.asarray([accuracy_metrics((ys == i).astype(int), y_preds[i, :]) for i in range(len(self.coi))])

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / cnt
        loss_seg_tar_avg = loss_seg_tar_cum / cnt
        loss_rec_src_avg = loss_rec_src_cum / cnt
        loss_rec_tar_avg = loss_rec_tar_cum / cnt
        loss_dc_x_avg = loss_dc_x_cum / cnt
        loss_dc_y_avg = loss_dc_y_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Testing Epoch %5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss DCX: %.6f - Loss DCY: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg,
               loss_dc_x_avg, loss_dc_y_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            if self.train_mode == RECONSTRUCTION:
                log_scalars([loss_rec_src_avg, loss_rec_tar_avg, loss_dc_x_avg],
                            ['test/' + s for s in ['loss-rec-src', 'loss-rec-tar', 'loss-dc-x']], writer, epoch=epoch)
            elif self.train_mode == SEGMENTATION:
                log_scalars(
                    [loss_seg_src_avg, loss_seg_tar_avg, loss_dc_y_avg, np.mean(js, axis=0), *(np.mean(ams, axis=0))],
                    ['test/' + s for s in
                     ['loss-seg-src', 'loss-seg-tar', 'loss-dc-y', 'jaccard', 'accuracy', 'balanced-accuracy',
                      'precision', 'recall', 'f-score']], writer, epoch=epoch)
            else:
                log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg, loss_dc_x_avg,
                             loss_dc_y_avg, np.mean(js, axis=0), *(np.mean(ams, axis=0))], ['test/' + s for s in
                                                                                            ['loss-seg-src',
                                                                                             'loss-seg-tar',
                                                                                             'loss-rec-src',
                                                                                             'loss-rec-tar',
                                                                                             'loss-dc-x', 'loss-dc-y',
                                                                                             'jaccard', 'accuracy',
                                                                                             'balanced-accuracy',
                                                                                             'precision', 'recall',
                                                                                             'f-score']], writer,
                            epoch=epoch)
            log_scalars([total_loss_avg], ['test/' + s for s in
                                           ['total-loss']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                log_images_2d([x_src.data], ['test/' + s for s in ['src/x']], writer, epoch=epoch)
                if self.train_mode == RECONSTRUCTION:
                    log_images_2d([x_src_rec.data, x_tar_ul.data, x_tar_ul_rec.data],
                                  ['test/' + s for s in ['src/x-rec', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                                  epoch=epoch)
                elif self.train_mode == SEGMENTATION:
                    y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([y_src.data, y_src_pred], ['test/' + s for s in ['src/y', 'src/y-pred']], writer,
                                  epoch=epoch)
                    if loader_tar_l is not None:
                        y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                        log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                      ['test/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer,
                                      epoch=epoch)
                else:
                    y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([x_src_rec.data, y_src.data, y_src_pred, x_tar_ul.data, x_tar_ul_rec.data],
                                  ['test/' + s for s in
                                   ['src/x-rec', 'src/y', 'src/y-pred', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                                  epoch=epoch)
                    if loader_tar_l is not None:
                        y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                        log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                      ['test/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer,
                                      epoch=epoch)

        return total_loss_avg

    def train_net(self, train_loader_src, train_loader_tar_ul, train_loader_tar_l, test_loader_src, test_loader_tar_ul,
                  test_loader_tar_l, optimizer, epochs, scheduler=None, test_freq=1, augmenter=None, print_stats=1,
                  log_dir=None, write_images_freq=1, device=0):
        """
        Trains the network in a semi-supervised fashion
        :param train_loader_src: source dataloader for training (labeled)
        :param train_loader_tar_ul: target dataloader for training (unlabeled)
        :param train_loader_tar_l: target dataloader for training (labeled)
        :param test_loader_src: source dataloader for testing (labeled)
        :param test_loader_tar_ul: target dataloader for testing (unlabeled)
        :param test_loader_tar_l: target dataloader for testing (labeled)
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter: data augmenter
        :param print_stats: frequency of printing statistics
        :param log_dir: logging directory
        :param write_images_freq:
        :param device: GPU device where the computations should occur
        """
        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(train_loader_src, train_loader_tar_ul, train_loader_tar_l, optimizer, epoch,
                             augmenter=augmenter, print_stats=print_stats, writer=writer,
                             write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step()

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_last_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar_ul, test_loader_tar_l, epoch,
                                            writer=writer, write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
