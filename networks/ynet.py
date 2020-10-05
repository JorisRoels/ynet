import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralnets.networks.unet import UNetEncoder2D, UNetDecoder2D, UNet2D, UNetEncoder3D, UNetDecoder3D, UNet3D
from neuralnets.util.metrics import jaccard, accuracy_metrics
from neuralnets.util.losses import DiceLoss
from neuralnets.util.tools import module_to_device, tensor_to_device, log_scalars, log_images_2d, log_images_3d, \
    augment_samples, get_labels
from neuralnets.util.io import print_frm
from torch.utils.tensorboard import SummaryWriter


# 2D Y-Net model for domain adaptive segmentation
class YNet2D(nn.Module):

    def __init__(self, in_channels=1, coi=(0, 1), feature_maps=64, levels=4, norm='instance', lambda_rec=1e-3,
                 dropout=0.0, activation='relu'):
        super(YNet2D, self).__init__()

        self.in_channels = in_channels
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.lambda_rec = lambda_rec
        self.dropout = dropout
        # self.seg_loss = nn.CrossEntropyLoss()
        self.seg_loss = DiceLoss()
        self.rec_loss = nn.MSELoss()
        self.dc_loss = nn.CrossEntropyLoss()

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout, activation=activation)

        # reconstruction decoder
        self.rec_decoder = UNetDecoder2D(out_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                         activation=activation, skip_connections=False)

        # segmentation decoder
        self.seg_decoder = UNetDecoder2D(out_channels=self.out_channels, feature_maps=feature_maps, levels=levels,
                                         norm=norm, activation=activation)

    def forward_rec(self, x):

        # contractive path
        encoder_outputs, encoded = self.encoder(x)

        # reconstruction decoder
        _, x_rec = self.rec_decoder(encoded, encoder_outputs)
        x_rec = torch.sigmoid(x_rec)

        return x_rec

    def forward_seg(self, x):

        # contractive path
        encoder_outputs, encoded = self.encoder(x)

        # segmentation decoder
        _, y_pred = self.seg_decoder(encoded, encoder_outputs)

        return y_pred

    def forward(self, x):

        # contractive path
        encoder_outputs, encoded = self.encoder(x)

        # reconstruction decoder
        _, x_rec = self.rec_decoder(encoded, encoder_outputs)
        x_rec = torch.sigmoid(x_rec)

        # segmentation decoder
        _, y_pred = self.seg_decoder(encoded, encoder_outputs)

        return x_rec, y_pred

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.encoder.in_channels, coi=self.coi, feature_maps=self.encoder.feature_maps,
                     levels=self.encoder.levels, skip_connections=self.seg_decoder.skip_connections,
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
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
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

            # forward prop and compute loss
            loss_seg_tar = torch.Tensor([0])
            x_src_rec, y_src_pred = self(x_src)
            x_tar_ul_rec, y_tar_ul_pred = self(x_tar_ul)
            loss_rec_src = self.rec_loss(x_src_rec, x_src)
            loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
            loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
            total_loss = loss_seg_src + self.lambda_rec * (loss_rec_src + loss_rec_tar)
            if loader_tar_l is not None:
                _, y_tar_l_pred = self(x_tar_l)
                loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print(
                    '[%s] Epoch %5d - Iteration %5d/%5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss: %.6f'
                    % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size,
                       loss_seg_src_cum / cnt, loss_seg_tar_cum / cnt, loss_rec_src_cum / cnt, loss_rec_tar_cum / cnt,
                       total_loss_cum / cnt))

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
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Training Epoch %4d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg,
               total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg, total_loss_avg],
                        ['train/' + s for s in
                         ['loss-seg-src', 'loss-seg-tar', 'loss-rec-src', 'loss-rec-tar', 'total-loss']], writer,
                        epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x_src.data, x_src_rec.data, y_src.data, y_src_pred, x_tar_ul.data, x_tar_ul_rec.data],
                              ['train/' + s for s in
                               ['src/x', 'src/x-rec', 'src/y', 'src/y-pred', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                              epoch=epoch)
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
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
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

            # check train mode and compute loss
            x_src_rec, y_src_pred = self(x_src)
            x_tar_ul_rec, y_tar_ul_pred = self(x_tar_ul)
            loss_rec_src = self.rec_loss(x_src_rec, x_src)
            loss_rec_tar = self.rec_loss(x_tar_ul, x_tar_ul_rec)
            loss_seg_src = self.seg_loss(y_src_pred, y_src[:, 0, ...])
            total_loss = loss_seg_src + self.lambda_rec * (loss_rec_src + loss_rec_tar)
            _, y_tar_l_pred = self(x_tar_l)
            loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
            total_loss = total_loss + loss_seg_tar

            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
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
        loss_rec_src_avg = loss_rec_src_cum / cnt
        loss_rec_tar_avg = loss_rec_tar_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Testing Epoch %4d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss rec src: %.6f - Loss rec tar: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg,
               total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars(
                [loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg, total_loss_avg,
                 np.mean(js, axis=0), *(np.mean(ams, axis=0))], ['test/' + s for s in
                                                                 ['loss-seg-src', 'loss-seg-tar', 'loss-rec-src',
                                                                  'loss-rec-tar', 'total-loss', 'jaccard', 'accuracy',
                                                                  'balanced-accuracy', 'precision', 'recall',
                                                                  'f-score']],
                writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x_src.data, x_src_rec.data, y_src.data, y_src_pred, x_tar_ul.data, x_tar_ul_rec.data],
                              ['test/' + s for s in
                               ['src/x', 'src/x-rec', 'src/y', 'src/y-pred', 'tar/x-ul', 'tar/x-ul-rec']], writer,
                              epoch=epoch)
                if loader_tar_l is not None:
                    y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, :, :].data
                    log_images_2d([x_tar_l.data, y_tar_l, y_tar_l_pred],
                                  ['test/' + s for s in ['tar/x-l', 'tar/y-l', 'tar/y-l-pred']], writer, epoch=epoch)

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


# 3D Y-Net model for domain adaptive segmentation
class YNet3D(nn.Module):

    def __init__(self, in_channels=1, coi=(0, 1), feature_maps=64, levels=4, norm='instance', lambda_rec=1e-3,
                 dropout_enc=0.0, dropout_dec=0.0, activation='relu'):
        super(YNet3D, self).__init__()

        self.in_channels = in_channels
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.lambda_rec = lambda_rec
        self.encoder_outputs = None
        self.segmentation_decoder_outputs = None
        self.reconstruction_decoder_outputs = None
        self.reconstruction_outputs = None

        # encoder
        self.encoder = UNetEncoder3D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder3D(out_channels=self.out_channels, feature_maps=feature_maps,
                                                  levels=levels,
                                                  norm=norm, dropout=dropout_dec, activation=activation)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder3D(out_channels=in_channels, feature_maps=feature_maps, levels=levels,
                                                    norm=norm, dropout=dropout_dec, activation=activation,
                                                    skip_connections=False)

    def forward(self, inputs):

        # contractive path
        self.encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        self.segmentation_decoder_outputs, segmentation_outputs = self.segmentation_decoder(encoded,
                                                                                            self.encoder_outputs)

        # reconstruction decoder
        self.reconstruction_decoder_outputs, self.reconstruction_outputs = self.reconstruction_decoder(encoded,
                                                                                                       self.encoder_outputs)
        self.reconstruction_outputs = torch.sigmoid(self.reconstruction_outputs)

        return segmentation_outputs

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet3D(in_channels=self.encoder.in_channels, coi=self.coi, feature_maps=self.encoder.feature_maps,
                     levels=self.encoder.levels, skip_connections=self.segmentation_decoder.skip_connections,
                     norm=self.encoder.norm, activation=self.encoder.activation,
                     dropout_enc=self.encoder.dropout, dropout_dec=self.segmentation_decoder.dropout)

        params = list(net.encoder.parameters())
        for i, param in enumerate(self.encoder.parameters()):
            params[i] = param

        params = list(net.decoder.parameters())
        for i, param in enumerate(self.segmentation_decoder.parameters()):
            params[i] = param

        return net

    def train_epoch_unsupervised(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, optimizer, epoch,
                                 augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                 write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader_src: source dataloader (labeled)
        :param loader_tar: target dataloader (unlabeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter_src: source data augmenter
        :param augmenter_tar: target data augmenter
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
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(zip(loader_src, loader_tar)):

            # transfer to suitable device
            data_src = tensor_to_device(data[0], device)
            data_tar = tensor_to_device(data[1], device)

            # augment if necessary
            x_src, y_src = augment_samples(data_src, augmenter=augmenter_src)
            x_tar, y_tar = augment_samples(data_tar, augmenter=augmenter_tar)
            y_src = get_labels(y_src, coi=self.coi, dtype=int)

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_src_pred = self(x_src)
            x_src_pred = self.reconstruction_outputs
            y_tar_pred = self(x_tar)
            x_tar_pred = self.reconstruction_outputs

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size, loss_seg,
                         loss_rec, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        loss_rec_avg = loss_rec_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_avg, loss_rec_avg, total_loss_avg],
                        ['train/' + s for s in ['loss-rec', 'loss-seg', 'total-loss']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, ...].data
                y_tar_pred = F.softmax(y_tar_pred, dim=1)[:, 1:2, ...].data
                log_images_3d([x_src, x_src_pred.data, y_src, y_src_pred, x_tar, x_tar_pred.data, y_tar, y_tar_pred],
                              ['train/' + s for s in
                               ['src/x', 'src/x-pred', 'src/y', 'src/y-pred', 'tar/x', 'tar/x-pred', 'tar/y',
                                'tar/y-pred']], writer, epoch=epoch)

        return total_loss_avg

    def train_epoch_semi_supervised(self, loader_src, loader_tar_ul, loader_tar_l, loss_seg_fn, loss_rec_fn, optimizer,
                                    epoch, augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                    write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader_src: source dataloader (labeled)
        :param loader_tar_ul: target dataloader (unlabeled)
        :param loader_tar_l: target dataloader (labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter_src: source data augmenter
        :param augmenter_tar: target data augmenter
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
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(zip(loader_src, loader_tar_ul, loader_tar_l)):

            # transfer to suitable device
            data_src = tensor_to_device(data[0], device)
            x_tar_ul = tensor_to_device(data[1], device)
            data_tar_l = tensor_to_device(data[2], device)

            # augment if necessary
            x_src, y_src = augment_samples(data_src, augmenter=augmenter_src)
            x_tar_l, y_tar_l = augment_samples(data_tar_l, augmenter=augmenter_tar)
            y_src = get_labels(y_src, coi=self.coi, dtype=int)
            y_tar_l = get_labels(y_tar_l, coi=self.coi, dtype=int)
            x_tar_ul = x_tar_ul.float()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_src_pred = self(x_src)
            x_src_pred = self.reconstruction_outputs
            y_tar_ul_pred = self(x_tar_ul)
            x_tar_ul_pred = self.reconstruction_outputs
            y_tar_l_pred = self(x_tar_l)
            x_tar_l_pred = self.reconstruction_outputs

            # compute loss
            loss_seg = 0.5 * (loss_seg_fn(y_src_pred, y_src) + loss_seg_fn(y_tar_l_pred, y_tar_l))
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_ul_pred, x_tar_ul))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size, loss_seg,
                         loss_rec, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        loss_rec_avg = loss_rec_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_avg, loss_rec_avg, total_loss_avg],
                        ['train/' + s for s in ['loss-rec', 'loss-seg', 'total-loss']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, ...].data
                y_tar_l_pred = F.softmax(y_tar_l_pred, dim=1)[:, 1:2, ...].data
                log_images_3d(
                    [x_src, x_src_pred.data, y_src, y_src_pred, x_tar_l, x_tar_l_pred.data, y_tar_l, y_tar_l_pred],
                    ['train/' + s for s in
                     ['src/x', 'src/x-pred', 'src/y', 'src/y-pred', 'tar/x', 'tar/x-pred', 'tar/y',
                      'tar/y-pred']], writer, epoch=epoch)

        return total_loss_avg

    def test_epoch(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, epoch, writer=None, write_images=False,
                   device=0):
        """
        Tests the network for one epoch
        :param loader_src: source dataloader (should be labeled)
        :param loader_tar: target dataloader (should be labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        y_src_preds = []
        ys_src = []
        y_tar_preds = []
        ys_tar = []
        for i, data in enumerate(zip(loader_src, loader_tar)):
            # get inputs and transfer to suitable device
            x_src, y_src = tensor_to_device(data[0], device)
            x_tar, y_tar = tensor_to_device(data[1], device)
            y_src = get_labels(y_src, coi=self.coi, dtype=int)
            y_tar = get_labels(y_tar, coi=self.coi, dtype=int)
            x_src = x_src.float()
            x_tar = x_tar.float()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_src_pred = self(x_src)
            x_src_pred = self.reconstruction_outputs
            y_tar_pred = self(x_tar)
            x_tar_pred = self.reconstruction_outputs

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            for b in range(y_src_pred.size(0)):
                y_src_preds.append(F.softmax(y_src_pred, dim=1).data.cpu().numpy()[b, 1, ...])
                y_tar_preds.append(F.softmax(y_tar_pred, dim=1).data.cpu().numpy()[b, 1, ...])
                ys_src.append(y_src[b, 0, ...].cpu().numpy())
                ys_tar.append(y_tar[b, 0, ...].cpu().numpy())

        # compute interesting metrics
        y_src_preds = np.asarray(y_src_preds)
        y_tar_preds = np.asarray(y_tar_preds)
        ys_src = np.asarray(ys_src)
        ys_tar = np.asarray(ys_tar)
        j_src = jaccard(ys_src, y_src_preds)
        j_tar = jaccard(ys_src, y_tar_preds)
        a_src, ba_src, p_src, r_src, f_src = accuracy_metrics(ys_src, y_src_preds)
        a_tar, ba_tar, p_tar, r_tar, f_tar = accuracy_metrics(ys_tar, y_tar_preds)

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        loss_rec_avg = loss_rec_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars(
                [loss_seg_avg, loss_rec_avg, total_loss_avg, j_src, a_src, ba_src, p_src, r_src, f_src, j_tar, a_tar,
                 ba_tar, p_tar, r_tar, f_tar], ['test/' + s for s in
                                                ['loss-rec', 'loss-seg', 'total-loss', 'src/jaccard', 'src/accuracy',
                                                 'src/balanced-accuracy', 'src/precision', 'src/recall', 'src/f-score',
                                                 'tar/jaccard', 'tar/accuracy', 'tar/balanced-accuracy',
                                                 'tar/precision', 'tar/recall', 'tar/f-score']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, ...].data
                y_tar_pred = F.softmax(y_tar_pred, dim=1)[:, 1:2, ...].data
                log_images_3d([x_src, x_src_pred.data, y_src, y_src_pred, x_tar, x_tar_pred.data, y_tar, y_tar_pred],
                              ['test/' + s for s in
                               ['src/x', 'src/x-pred', 'src/y', 'src/y-pred', 'tar/x', 'tar/x-pred', 'tar/y',
                                'tar/y-pred']], writer, epoch=epoch)

        return total_loss_avg

    def train_net_unsupervised(self, train_loader_src, train_loader_tar, test_loader_src, test_loader_tar,
                               loss_seg_fn, loss_rec_fn, optimizer, epochs, scheduler=None, test_freq=1,
                               augmenter_src=None, augmenter_tar=None, print_stats=1, log_dir=None,
                               write_images_freq=1, device=0):
        """
        Trains the network in an unsupervised fashion
        :param train_loader_src: source dataloader for training (labeled)
        :param train_loader_tar: target dataloader for training (unlabeled)
        :param test_loader_src: source dataloader for testing (labeled)
        :param test_loader_tar: target dataloader for testing (labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter_src: source data augmenter
        :param augmenter_tar: target data augmenter
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
            self.train_epoch_unsupervised(train_loader_src, train_loader_tar, loss_seg_fn, loss_rec_fn, optimizer,
                                          epoch, augmenter_src=augmenter_src, augmenter_tar=augmenter_tar,
                                          print_stats=print_stats, writer=writer,
                                          write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

    def train_net_semi_supervised(self, train_loader_src, train_loader_tar_ul, train_loader_tar_l, test_loader_src,
                                  test_loader_tar, loss_seg_fn, loss_rec_fn, optimizer, epochs, scheduler=None,
                                  test_freq=1, augmenter_src=None, augmenter_tar=None, print_stats=1, log_dir=None,
                                  write_images_freq=1, device=0):
        """
        Trains the network in a semi-supervised fashion
        :param train_loader_src: source dataloader for training (labeled)
        :param train_loader_tar_ul: target dataloader for training (unlabeled)
        :param train_loader_tar_l: target dataloader for training (labeled)
        :param test_loader_src: source dataloader for testing (labeled)
        :param test_loader_tar: target dataloader for testing (labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter_src: source data augmenter
        :param augmenter_tar: target data augmenter
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
            self.train_epoch_semi_supervised(train_loader_src, train_loader_tar_ul, train_loader_tar_l, loss_seg_fn,
                                             loss_rec_fn, optimizer, epoch, augmenter_src=augmenter_src,
                                             augmenter_tar=augmenter_tar, print_stats=print_stats, writer=writer,
                                             write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
