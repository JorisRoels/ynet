import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from neuralnets.networks.unet import UNetEncoder2D, UNetDecoder2D, UNet2D, UNetEncoder3D, UNetDecoder3D, UNet3D
from neuralnets.util.metrics import jaccard, accuracy_metrics
from torch.utils.tensorboard import SummaryWriter


# 2D Y-Net model for domain adaptive segmentation
class YNet2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, norm='instance', lambda_rec=1e-3,
                 dropout_enc=0.0, dropout_dec=0.0, activation='relu'):
        super(YNet2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder2D(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                                                  norm=norm, dropout=dropout_dec, activation=activation)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder2D(out_channels=in_channels, feature_maps=feature_maps, levels=levels,
                                                    norm=norm, dropout=dropout_dec, activation=activation,
                                                    skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.encoder.in_channels, out_channels=self.decoder.out_channels,
                     feature_maps=self.encoder.feature_maps, levels=self.encoder.levels,
                     skip_connections=self.segmentation_decoder.skip_connections,
                     norm=self.encoder.norm, activation=self.encoder.activation,
                     dropout_enc=self.encoder.dropout, dropout_dec=self.segmentation_decoder.dropout)

        params = list(net.encoder.parameters())
        for i, param in enumerate(self.encoder.parameters()):
            params[i] = param

        params = list(net.decoder.parameters())
        for i, param in enumerate(self.decoder.parameters()):
            params[i] = param

        return net

    def train_epoch_unsupervised(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, optimizer, epoch,
                                 augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                 write_images=False):
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
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs and augment if necessary
            if augmenter_src is not None:
                bs = data[0].size(0)
                xy = augmenter_src(torch.cat((data[0].cuda().float(), data[1].cuda().float()), dim=0))
                x_src = xy[:bs, ...].float()
                y_src = xy[bs:, ...].long()

                x_tar = torch.Tensor(x_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar[b, ...] = torch.Tensor(loader_tar.dataset[0])
                x_tar = augmenter_tar(x_tar.cuda().float())
            else:
                x_src = data[0].cuda().float()
                y_src = data[1].cuda().long()
                x_tar = torch.Tensor(x_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar[b, ...] = torch.Tensor(loader_tar.dataset[0])

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

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
            writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('train/loss', total_loss_avg, epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3))
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3))
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
                x_pred = vutils.make_grid(x_pred.data, normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', ys, epoch)
                writer.add_image('train/x-pred', x_pred, epoch)
                writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    def train_epoch_semi_supervised(self, loader_src, loader_tar_ul, loader_tar_l, loss_seg_fn, loss_rec_fn, optimizer,
                                    epoch,
                                    augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                    write_images=False):
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
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs and augment if necessary
            if augmenter_src is not None:
                bs = data[0].size(0)
                xy = augmenter_src(torch.cat((data[0].cuda().float(), data[1].cuda().float()), dim=0))
                x_src = xy[:bs, ...].float()
                y_src = xy[bs:, ...].long()

                x_tar_ul = torch.Tensor(x_src.shape)
                x_tar_l = torch.Tensor(x_src.shape)
                y_tar_l = torch.Tensor(y_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar_ul[b, ...] = torch.Tensor(loader_tar_ul.dataset[0])
                    data = loader_tar_l.dataset[0]
                    x_tar_l[b, ...] = torch.Tensor(data[0])
                    y_tar_l[b, ...] = torch.Tensor(data[1])
                x_tar_ul = augmenter_tar(x_tar_ul.cuda().float())
                xy = augmenter_src(torch.cat((x_tar_l.cuda().float(), y_tar_l.cuda().float()), dim=0))
                x_tar_l = xy[:bs, ...].float()
                y_tar_l = xy[bs:, ...].long()
            else:
                x_src = data[0].cuda().float()
                y_src = data[1].cuda().long()
                x_tar_ul = torch.Tensor(x_src.shape)
                x_tar_l = torch.Tensor(x_src.shape)
                y_tar_l = torch.Tensor(y_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar_ul[b, ...] = torch.Tensor(loader_tar_ul.dataset[0])
                    data = loader_tar_l.dataset[0]
                    x_tar_l[b, ...] = torch.Tensor(data[0])
                    y_tar_l[b, ...] = torch.Tensor(data[1])

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_ul_pred, y_tar_ul_pred = self(x_tar_ul)
            x_tar_l_pred, y_tar_l_pred = self(x_tar_l)

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
            writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('train/loss', total_loss_avg, epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar_ul, x_tar_l), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3))
                x_pred = torch.cat((x_src_pred, x_tar_ul_pred, x_tar_l_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3))
                y_pred = torch.cat((y_src_pred, y_tar_ul_pred, y_tar_l_pred), dim=0)
                y = torch.cat((y_src, y_tar_l), dim=0)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                ys = vutils.make_grid(y, normalize=y.max() - y.min() > 0, scale_each=True)
                x_pred = vutils.make_grid(x_pred.data, normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', ys, epoch)
                writer.add_image('train/x-pred', x_pred, epoch)
                writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    def test_epoch(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, epoch, writer=None, write_images=False):
        """
        Tests the network for one epoch
        :param loader_src: source dataloader (should be labeled)
        :param loader_tar: target dataloader (should be labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        j_cum = np.asarray([0.0, 0.0])
        a_cum = np.asarray([0.0, 0.0])
        p_cum = np.asarray([0.0, 0.0])
        r_cum = np.asarray([0.0, 0.0])
        f_cum = np.asarray([0.0, 0.0])
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src = data[0].cuda().float()
            y_src = data[1].cuda().long()
            x_tar = torch.Tensor(x_src.shape)
            y_tar = torch.Tensor(y_src.shape)
            for b in range(x_src.shape[0]):
                data = loader_tar.dataset[0]
                x_tar[b, ...] = torch.Tensor(data[0])
                y_tar[b, ...] = torch.Tensor(data[1])
            x_tar, y_tar = x_tar.cuda().float(), y_tar.cuda().long()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # compute other interesting metrics
            y_src_ = F.softmax(y_src_pred, dim=1).data.cpu().numpy()[:, 1, ...]
            y_tar_ = F.softmax(y_tar_pred, dim=1).data.cpu().numpy()[:, 1, ...]
            j_cum[0] += jaccard(y_src_, y_src.cpu().numpy())
            j_cum[1] += jaccard(y_tar_, y_tar.cpu().numpy())
            a_src, p_src, r_src, f_src = accuracy_metrics(y_src_, y_src.cpu().numpy())
            a_tar, p_tar, r_tar, f_tar = accuracy_metrics(y_tar_, y_tar.cpu().numpy())
            a_cum[0] += a_src
            a_cum[1] += a_tar
            p_cum[0] += p_src
            p_cum[1] += p_tar
            r_cum[0] += r_src
            r_cum[1] += r_tar
            f_cum[0] += f_src
            f_cum[1] += f_tar

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        loss_rec_avg = loss_rec_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('test/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('test/loss', total_loss_avg, epoch)

            writer.add_scalar('test/src-jaccard', j_avg[0], epoch)
            writer.add_scalar('test/src-accuracy', a_avg[0], epoch)
            writer.add_scalar('test/src-precision', p_avg[0], epoch)
            writer.add_scalar('test/src-recall', r_avg[0], epoch)
            writer.add_scalar('test/src-f-score', f_avg[0], epoch)

            writer.add_scalar('test/tar-jaccard', j_avg[1], epoch)
            writer.add_scalar('test/tar-accuracy', a_avg[1], epoch)
            writer.add_scalar('test/tar-precision', p_avg[1], epoch)
            writer.add_scalar('test/tar-recall', r_avg[1], epoch)
            writer.add_scalar('test/tar-f-score', f_avg[1], epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3))
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3))
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
                x_pred = vutils.make_grid(x_pred.data, normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/y', ys, epoch)
                writer.add_image('test/x-pred', x_pred, epoch)
                writer.add_image('test/y-pred', y_pred, epoch)

        return total_loss_avg

    def train_net_unsupervised(self, train_loader_src, train_loader_tar, test_loader_src, test_loader_tar,
                               loss_seg_fn, loss_rec_fn, optimizer, epochs, scheduler=None, test_freq=1,
                               augmenter_src=None, augmenter_tar=None, print_stats=1, log_dir=None,
                               write_images_freq=1):
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
                                          write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True)

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
                                  write_images_freq=1):
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
                                             write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()


# 3D Y-Net model for domain adaptive segmentation
class YNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, norm='instance', lambda_rec=1e-3,
                 dropout_enc=0.0, dropout_dec=0.0, activation='relu'):
        super(YNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder3D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder3D(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                                                  norm=norm, dropout=dropout_dec, activation=activation)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder3D(out_channels=in_channels, feature_maps=feature_maps, levels=levels,
                                                    norm=norm, dropout=dropout_dec, activation=activation,
                                                    skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet3D(in_channels=self.encoder.in_channels, out_channels=self.decoder.out_channels,
                     feature_maps=self.encoder.feature_maps, levels=self.encoder.levels,
                     skip_connections=self.segmentation_decoder.skip_connections,
                     norm=self.encoder.norm, activation=self.encoder.activation,
                     dropout_enc=self.encoder.dropout, dropout_dec=self.segmentation_decoder.dropout)

        params = list(net.encoder.parameters())
        for i, param in enumerate(self.encoder.parameters()):
            params[i] = param

        params = list(net.decoder.parameters())
        for i, param in enumerate(self.decoder.parameters()):
            params[i] = param

        return net

    def train_epoch_unsupervised(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, optimizer, epoch,
                                 augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                 write_images=False):
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
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs and augment if necessary
            if augmenter_src is not None:
                bs = data[0].size(0)
                xy = augmenter_src(torch.cat((data[0].cuda().float(), data[1].cuda().float()), dim=0))
                x_src = xy[:bs, ...].float()
                y_src = xy[bs:, ...].long()

                x_tar = torch.Tensor(x_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar[b, ...] = torch.Tensor(loader_tar.dataset[0])
                x_tar = augmenter_tar(x_tar.cuda().float())
            else:
                x_src = data[0].cuda().float()
                y_src = data[1].cuda().long()
                x_tar = torch.Tensor(x_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar[b, ...] = torch.Tensor(loader_tar.dataset[0])

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

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
            writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('train/loss', total_loss_avg, epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3), x.size(4))
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3), x_pred.size(4))
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                x = vutils.make_grid(x[:, :, x.size(2) // 2, :, :], normalize=True, scale_each=True)
                ys = vutils.make_grid(y_src[:, :, y_src.size(2) // 2, :, :], normalize=y_src.max() - y_src.min() > 0,
                                      scale_each=True)
                x_pred = vutils.make_grid(x_pred.data[:, :, x_pred.size(2) // 2, :, :], normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, y_pred.size(2) // 2, :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', ys, epoch)
                writer.add_image('train/x-pred', x_pred, epoch)
                writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    def train_epoch_semi_supervised(self, loader_src, loader_tar_ul, loader_tar_l, loss_seg_fn, loss_rec_fn, optimizer,
                                    epoch,
                                    augmenter_src=None, augmenter_tar=None, print_stats=1, writer=None,
                                    write_images=False):
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
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs and augment if necessary
            if augmenter_src is not None:
                bs = data[0].size(0)
                xy = augmenter_src(torch.cat((data[0].cuda().float(), data[1].cuda().float()), dim=0))
                x_src = xy[:bs, ...].float()
                y_src = xy[bs:, ...].long()

                x_tar_ul = torch.Tensor(x_src.shape)
                x_tar_l = torch.Tensor(x_src.shape)
                y_tar_l = torch.Tensor(y_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar_ul[b, ...] = torch.Tensor(loader_tar_ul.dataset[0])
                    data = loader_tar_l.dataset[0]
                    x_tar_l[b, ...] = torch.Tensor(data[0])
                    y_tar_l[b, ...] = torch.Tensor(data[1])
                x_tar_ul = augmenter_tar(x_tar_ul.cuda().float())
                xy = augmenter_src(torch.cat((x_tar_l.cuda().float(), y_tar_l.cuda().float()), dim=0))
                x_tar_l = xy[:bs, ...].float()
                y_tar_l = xy[bs:, ...].long()
            else:
                x_src = data[0].cuda().float()
                y_src = data[1].cuda().long()
                x_tar_ul = torch.Tensor(x_src.shape)
                x_tar_l = torch.Tensor(x_src.shape)
                y_tar_l = torch.Tensor(y_src.shape)
                for b in range(x_src.shape[0]):
                    x_tar_ul[b, ...] = torch.Tensor(loader_tar_ul.dataset[0])
                    data = loader_tar_l.dataset[0]
                    x_tar_l[b, ...] = torch.Tensor(data[0])
                    y_tar_l[b, ...] = torch.Tensor(data[1])

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_ul_pred, y_tar_ul_pred = self(x_tar_ul)
            x_tar_l_pred, y_tar_l_pred = self(x_tar_l)

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
            writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('train/loss', total_loss_avg, epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar_ul, x_tar_l), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3), x.size(4))
                x_pred = torch.cat((x_src_pred, x_tar_ul_pred, x_tar_l_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3), x_pred.size(4))
                y_pred = torch.cat((y_src_pred, y_tar_ul_pred, y_tar_l_pred), dim=0)
                y = torch.cat((y_src, y_tar_l), dim=0)
                x = vutils.make_grid(x[:,:,x.size(2)//2,:,:], normalize=True, scale_each=True)
                ys = vutils.make_grid(y[:,:,y.size(2)//2,:,:], normalize=y.max() - y.min() > 0, scale_each=True)
                x_pred = vutils.make_grid(x_pred.data[:,:,x_pred.size(2)//2,:,:], normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, y_pred.size(2), :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', ys, epoch)
                writer.add_image('train/x-pred', x_pred, epoch)
                writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    def test_epoch(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, epoch, writer=None, write_images=False):
        """
        Tests the network for one epoch
        :param loader_src: source dataloader (should be labeled)
        :param loader_tar: target dataloader (should be labeled)
        :param loss_seg_fn: segmentation loss function
        :param loss_rec_fn: reconstruction loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0
        j_cum = np.asarray([0.0, 0.0])
        a_cum = np.asarray([0.0, 0.0])
        p_cum = np.asarray([0.0, 0.0])
        r_cum = np.asarray([0.0, 0.0])
        f_cum = np.asarray([0.0, 0.0])
        cnt = 0

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src = data[0].cuda().float()
            y_src = data[1].cuda().long()
            x_tar = torch.Tensor(x_src.shape)
            y_tar = torch.Tensor(y_src.shape)
            for b in range(x_src.shape[0]):
                data = loader_tar.dataset[0]
                x_tar[b, ...] = torch.Tensor(data[0])
                y_tar[b, ...] = torch.Tensor(data[1])
            x_tar, y_tar = x_tar.cuda().float(), y_tar.cuda().long()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # compute other interesting metrics
            y_src_ = F.softmax(y_src_pred, dim=1).data.cpu().numpy()[:, 1, ...]
            y_tar_ = F.softmax(y_tar_pred, dim=1).data.cpu().numpy()[:, 1, ...]
            j_cum[0] += jaccard(y_src_, y_src.cpu().numpy())
            j_cum[1] += jaccard(y_tar_, y_tar.cpu().numpy())
            a_src, p_src, r_src, f_src = accuracy_metrics(y_src_, y_src.cpu().numpy())
            a_tar, p_tar, r_tar, f_tar = accuracy_metrics(y_tar_, y_tar.cpu().numpy())
            a_cum[0] += a_src
            a_cum[1] += a_tar
            p_cum[0] += p_src
            p_cum[1] += p_tar
            r_cum[0] += r_src
            r_cum[1] += r_tar
            f_cum[0] += f_src
            f_cum[1] += f_tar

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        loss_rec_avg = loss_rec_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss-seg', loss_seg_avg, epoch)
            writer.add_scalar('test/loss-rec', loss_rec_avg, epoch)
            writer.add_scalar('test/loss', total_loss_avg, epoch)

            writer.add_scalar('test/src-jaccard', j_avg[0], epoch)
            writer.add_scalar('test/src-accuracy', a_avg[0], epoch)
            writer.add_scalar('test/src-precision', p_avg[0], epoch)
            writer.add_scalar('test/src-recall', r_avg[0], epoch)
            writer.add_scalar('test/src-f-score', f_avg[0], epoch)

            writer.add_scalar('test/tar-jaccard', j_avg[1], epoch)
            writer.add_scalar('test/tar-accuracy', a_avg[1], epoch)
            writer.add_scalar('test/tar-precision', p_avg[1], epoch)
            writer.add_scalar('test/tar-recall', r_avg[1], epoch)
            writer.add_scalar('test/tar-f-score', f_avg[1], epoch)

            # log images if necessary
            if write_images:
                x = torch.cat((x_src, x_tar), dim=0)
                x = x.view(-1, 1, x.size(2), x.size(3), x.size(4))
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3), x_pred.size(4))
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                x = vutils.make_grid(x[:,:,x.size(2)//2,:,:], normalize=True, scale_each=True)
                ys = vutils.make_grid(y_src[:,:,y_src.size(2)//2,:,:], normalize=y_src.max() - y_src.min() > 0, scale_each=True)
                x_pred = vutils.make_grid(x_pred.data[:,:,x_pred.size(2)//2,:,:], normalize=True, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, y_pred.size(2), :, :].data,
                                          normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/y', ys, epoch)
                writer.add_image('test/x-pred', x_pred, epoch)
                writer.add_image('test/y-pred', y_pred, epoch)

        return total_loss_avg

    def train_net_unsupervised(self, train_loader_src, train_loader_tar, test_loader_src, test_loader_tar,
                               loss_seg_fn, loss_rec_fn, optimizer, epochs, scheduler=None, test_freq=1,
                               augmenter_src=None, augmenter_tar=None, print_stats=1, log_dir=None,
                               write_images_freq=1):
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
                                          write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True)

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
                                  write_images_freq=1):
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
                                             write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(test_loader_src, test_loader_tar, loss_seg_fn, loss_rec_fn, epoch,
                                            writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
