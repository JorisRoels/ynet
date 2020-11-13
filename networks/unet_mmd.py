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
from util.losses import feature_regularization_loss


# 2D U-Net model with maximum mean discrepancy regularization in the final feature layer for domain adaptive
# segmentation
class UNetMMD2D(nn.Module):

    def __init__(self, in_channels=1, coi=(0, 1), feature_maps=64, levels=4, norm='instance', lambda_mmd=1e0,
                 dropout=0.0, activation='relu'):
        super(UNetMMD2D, self).__init__()

        self.in_channels = in_channels
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.lambda_mmd = lambda_mmd
        self.dropout = dropout
        # self.seg_loss = nn.CrossEntropyLoss()
        self.seg_loss = DiceLoss()
        self.dc_loss = nn.CrossEntropyLoss()

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout, activation=activation)

        # segmentation decoder
        self.decoder = UNetDecoder2D(out_channels=self.out_channels, feature_maps=feature_maps, levels=levels,
                                     norm=norm, activation=activation)

    def forward(self, x):

        # contractive path
        encoder_outputs, encoded = self.encoder(x)

        # segmentation decoder
        decoder_outputs, y_pred = self.decoder(encoded, encoder_outputs)

        return y_pred, decoder_outputs[-1]

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.encoder.in_channels, coi=self.coi, feature_maps=self.encoder.feature_maps,
                     levels=self.encoder.levels, skip_connections=self.decoder.skip_connections,
                     norm=self.encoder.norm, activation=self.encoder.activation, dropout_enc=self.encoder.dropout)

        net.encoder.load_state_dict(self.encoder.state_dict())
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
        loss_mmd_cum = 0.0
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
                data_aug = (data_src[0], data_src[1])
                x_src, y_src = augment_samples(data_aug, augmenter=augmenter)
                data_aug = (x_tar_ul, x_tar_ul)
                x_tar_ul, _ = augment_samples(data_aug, augmenter=augmenter)
            else:
                data_aug = (data_src[0], data_src[1])
                x_src, y_src = augment_samples(data_aug, augmenter=augmenter)
                data_aug = (x_tar_ul, x_tar_ul)
                x_tar_ul, _ = augment_samples(data_aug, augmenter=augmenter)
                data_aug = (data_tar_l[0], data_tar_l[1])
                x_tar_l, y_tar_l = augment_samples(data_aug, augmenter=augmenter)
                y_tar_l = get_labels(y_tar_l, coi=self.coi, dtype=int)
            y_src = get_labels(y_src, coi=self.coi, dtype=int)
            x_tar_ul = x_tar_ul.float()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop and compute loss
            loss_seg_tar = torch.Tensor([0])
            y_src_pred, f_src = self(x_src)
            y_tar_ul_pred, f_tar_ul = self(x_tar_ul)
            loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
            loss_mmd = feature_regularization_loss(f_src, f_tar_ul, method='mmd')
            total_loss = loss_seg_src + self.lambda_mmd * loss_mmd
            if loader_tar_l is not None:
                y_tar_l_pred, _ = self(x_tar_l)
                loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_mmd_cum += loss_mmd.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print(
                    '[%s] Epoch %5d - Iteration %5d/%5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss MMD: %.6f - Loss: %.6f'
                    % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size,
                       loss_seg_src_cum / cnt, loss_seg_tar_cum / cnt, loss_mmd_cum / cnt, total_loss_cum / cnt))

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
        loss_mmd_avg = loss_mmd_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Training Epoch %4d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss MMD: %.6f - Loss: %.6f' % (
            datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_mmd_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_mmd_avg, total_loss_avg],
                        ['train/' + s for s in ['loss-seg-src', 'loss-seg-tar', 'loss-mmd', 'total-loss']], writer,
                        epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x_src.data, y_src.data, y_src_pred, x_tar_ul.data],
                              ['train/' + s for s in ['src/x', 'src/y', 'src/y-pred', 'tar/x-ul']], writer, epoch=epoch)
                if loader_tar_l is not None:
                    y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                  ['train/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer, epoch=epoch)

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
        loss_mmd_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # zip dataloaders
        if loader_tar_l is None:
            dl = zip(loader_src, loader_tar_ul)
        else:
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

            # forward prop and compute loss
            y_src_pred, f_src = self(x_src)
            y_tar_ul_pred, f_tar_ul = self(x_tar_ul)
            loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
            loss_mmd = feature_regularization_loss(f_src, f_tar_ul, method='mmd')
            total_loss = loss_seg_src + self.lambda_mmd * loss_mmd
            y_tar_l_pred, _ = self(x_tar_l)
            loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
            total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_mmd_cum += loss_mmd.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            for b in range(y_tar_l_pred.size(0)):
                y_preds.append(F.softmax(y_tar_l_pred, dim=1)[b, ...].view(y_tar_l_pred.size(1), -1).data.cpu().numpy())
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
        y_preds = np.concatenate(y_preds, axis=1)
        ys = np.concatenate(ys)
        js = np.asarray([jaccard((ys == i).astype(int), y_preds[i, :]) for i in range(len(self.coi))])
        ams = np.asarray([accuracy_metrics((ys == i).astype(int), y_preds[i, :]) for i in range(len(self.coi))])

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / cnt
        loss_seg_tar_avg = loss_seg_tar_cum / cnt
        loss_mmd_avg = loss_mmd_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Testing Epoch %4d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss MMD: %.6f - Loss: %.6f' % (
            datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_mmd_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars(
                [loss_seg_src_avg, loss_seg_tar_avg, loss_mmd_avg, total_loss_avg, np.mean(js, axis=0),
                 *(np.mean(ams, axis=0))], ['test/' + s for s in
                                            ['loss-seg-src', 'loss-seg-tar', 'loss-mmd', 'total-loss', 'jaccard',
                                             'accuracy', 'balanced-accuracy', 'precision', 'recall', 'f-score']],
                writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x_src.data, y_src.data, y_src_pred, x_tar_ul.data, x_tar_l.data, y_tar_l, y_tar_l_pred],
                              ['test/' + s for s in
                               ['src/x', 'src/y', 'src/y-pred', 'tar/x-ul', 'tar/x-l', 'tar/y-l', 'tar/y-l-pred']],
                              writer, epoch=epoch)

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
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self.state_dict(), os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
