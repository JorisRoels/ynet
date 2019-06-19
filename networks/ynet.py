
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.unet import UNetEncoder2D, UNetDecoder2D, UNetEncoder3D, UNetDecoder3D, unet_from_encoder_decoder
from util.validation import validate

# Y-Net model
class YNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, batch_norm=True, lambda_rec=1e-3, dropout=0.0):
        super(YNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout = dropout
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, dropout=dropout)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder2D(out_channels=out_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder2D(out_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder, vol=False)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar,
                    optimizer, loss_seg_fn, loss_rec_fn, epoch,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        # list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda().float(), data[1].long().cuda()
            # x_tar = list_tar[i % len(list_tar)][1].float().cuda()
            x_tar = torch.Tensor(x_src.shape)
            for b in range(x_src.shape[0]):
                x_tar[b, ...] = torch.Tensor(loader_tar.dataset[0])
            x_tar = x_tar.float().cuda()

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

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset)/loader_src.batch_size, loss_seg, loss_rec, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
        loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # scalars
        writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
        writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
        writer.add_scalar('train/loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            x = torch.cat((x_src, x_tar), dim=0); x = x.view(-1, 1, x.size(2), x.size(3))
            x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0); x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3))
            y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0);
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

    # trains the network
    def train_net(self, train_loader_source, train_loader_target, test_data, test_labels,
                  optimizer, loss_seg_fn, loss_rec_fn, scheduler=None, epochs=100, test_freq=1, print_stats=1,
                  log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(logdir=log_dir)
        else:
            writer = None

        j_max = 0
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source, loader_tar=train_loader_target,
                             optimizer=optimizer, loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0 and test_freq > 0:
                a, p, r, f, j, d = validate(self, test_data, test_labels, train_loader_source.dataset.input_shape[-2:],
                                            val_file=os.path.join(log_dir, 'validation_'+str(epoch)+'.npy'),
                                            dtypes=('uint8', 'uint8', 'float64'), keys=('image', 'image', 'labels'),
                                            writer=writer, epoch=epoch)
                writer.add_scalar('target/accuracy', a, epoch)
                writer.add_scalar('target/precision', p, epoch)
                writer.add_scalar('target/recall', r, epoch)
                writer.add_scalar('target/f-score', f, epoch)
                writer.add_scalar('target/jaccard', j, epoch)
                writer.add_scalar('target/dice', d, epoch)

                # and save model if lower test loss is found
                if j > j_max:
                    j_max = j
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

# Y-Net model
class YNetFT(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, batch_norm=True, lambda_rec=1e-3, dropout=0.0):
        super(YNetFT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout = dropout
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, dropout=dropout)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder2D(out_channels=out_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder2D(out_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder, vol=False)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar_unlabeled, loader_tar_labeled,
                    optimizer, loss_seg_fn, loss_rec_fn, epoch,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        # list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            # labeled source labels
            x_src, y_src = data[0].cuda().float(), data[1].long().cuda()
            # unlabeled target samples
            x_tar_ul = torch.Tensor(x_src.shape)
            for b in range(x_src.shape[0]):
                x_tar_ul[b, ...] = torch.Tensor(loader_tar_unlabeled.dataset[0])
            x_tar_ul = x_tar_ul.float().cuda()
            # labeled target samples
            x_tar_l = torch.Tensor(x_src.shape)
            y_tar_l = torch.Tensor(y_src.shape)
            for b in range(x_src.shape[0]):
                sample = loader_tar_labeled.dataset[0]
                x_tar_l[b, ...] = torch.Tensor(sample[0])
                y_tar_l[b, ...] = torch.Tensor(sample[1])
            x_tar_l = x_tar_l.float().cuda()
            y_tar_l = y_tar_l.long().cuda()

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

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset)/loader_src.batch_size, loss_seg, loss_rec, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
        loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # scalars
        writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
        writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
        writer.add_scalar('train/loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            x = torch.cat((x_src, x_tar_ul, x_tar_l), dim=0); x = x.view(-1, 1, x.size(2), x.size(3))
            x_pred = torch.cat((x_src_pred, x_tar_ul_pred, x_tar_l_pred), dim=0); x_pred = x_pred.view(-1, 1, x_pred.size(2), x_pred.size(3))
            y_pred = torch.cat((y_src_pred, y_tar_ul_pred, y_tar_l_pred), dim=0);
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

    # trains the network
    def train_net(self, train_loader_source_labeled, train_loader_target_unlabeled, train_loader_target_labeled,
                  test_data, test_labels, optimizer, loss_seg_fn, loss_rec_fn, scheduler=None, epochs=100, test_freq=1,
                  print_stats=1, log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(logdir=log_dir)
        else:
            writer = None

        j_max = 0
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source_labeled, loader_tar_unlabeled=train_loader_target_unlabeled,
                             loader_tar_labeled=train_loader_target_labeled, optimizer=optimizer,
                             loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch, print_stats=print_stats,
                             writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0 and test_freq > 0:
                a, p, r, f, j, d = validate(self, test_data, test_labels, train_loader_source_labeled.dataset.input_shape[-2:],
                                            val_file=os.path.join(log_dir, 'validation_'+str(epoch)+'.npy'),
                                            dtypes=('uint8', 'uint8', 'float64'), keys=('image', 'image', 'labels'),
                                            writer=writer, epoch=epoch)
                writer.add_scalar('target/accuracy', a, epoch)
                writer.add_scalar('target/precision', p, epoch)
                writer.add_scalar('target/recall', r, epoch)
                writer.add_scalar('target/f-score', f, epoch)
                writer.add_scalar('target/jaccard', j, epoch)
                writer.add_scalar('target/dice', d, epoch)

                # and save model if lower test loss is found
                if j > j_max:
                    j_max = j
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

# 3D Y-Net model
class YNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, batch_norm=True, lambda_rec=1e-3, dropout=0.0):
        super(YNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.dropout = dropout
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder3D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, dropout=dropout)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder3D(out_channels=out_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder3D(out_channels=in_channels, feature_maps=feature_maps, levels=levels, batch_norm=batch_norm, skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder, vol=True)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar,
                    optimizer, loss_seg_fn, loss_rec_fn, epoch,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda().long()
            x_tar = list_tar[i % len(list_tar)][1].cuda()

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

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset)/loader_src.batch_size, loss_seg, loss_rec, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
        loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, total_loss_avg))

        # scalars
        writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
        writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
        writer.add_scalar('train/loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            x = torch.cat((x_src, x_tar), dim=0).reshape(-1,x_src.size(2),x_src.size(3),x_src.size(4))
            x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0).reshape(-1,x_src_pred.size(2),x_src_pred.size(3),x_src_pred.size(4))
            y_pred = F.softmax(torch.cat((y_src_pred, y_tar_pred), dim=0), dim=1)[:, 1:2, :, :].data.reshape(-1,y_src_pred.size(2),y_src_pred.size(3),y_src_pred.size(4))
            y_src = y_src.reshape(-1,y_src.size(2),y_src.size(3),y_src.size(4))
            ind = x.size(1)//2
            x = vutils.make_grid(x[:,ind:ind+1,:,:], normalize=True, scale_each=True)
            ys = vutils.make_grid(y_src[:,ind:ind+1,:,:], normalize=y_src.max() - y_src.min() > 0, scale_each=True)
            x_pred = vutils.make_grid(x_pred.data[:,ind:ind+1,:,:], normalize=True, scale_each=True)
            y_pred = vutils.make_grid(y_pred[:,ind:ind+1,:,:],
                                      normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
            writer.add_image('train/x', x, epoch)
            writer.add_image('train/y', ys, epoch)
            writer.add_image('train/x-pred', x_pred, epoch)
            writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    # trains the network
    def train_net(self, train_loader_source, train_loader_target, optimizer, loss_seg_fn, loss_rec_fn, test_data=None,
                  test_labels=None, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None,
                  write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(logdir=log_dir)
        else:
            writer = None

        j_max = 0
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source, loader_tar=train_loader_target,
                             optimizer=optimizer, loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0 and test_freq > 0 and test_data is not None and test_labels is not None:
                a, p, r, f, j, d = validate(self, test_data, test_labels, train_loader_source.dataset.input_shape,
                                            val_file=os.path.join(log_dir, 'validation_'+str(epoch)+'.npy'),
                                            dtypes=('uint8', 'uint8', 'float64'), keys=('image', 'image', 'labels'),
                                            writer=writer, epoch=epoch)
                writer.add_scalar('target/accuracy', a, epoch)
                writer.add_scalar('target/precision', p, epoch)
                writer.add_scalar('target/recall', r, epoch)
                writer.add_scalar('target/f-score', f, epoch)
                writer.add_scalar('target/jaccard', j, epoch)
                writer.add_scalar('target/dice', d, epoch)

                # and save model if lower test loss is found
                if j > j_max:
                    j_max = j
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            if epoch % test_freq == 0 and test_freq > 0:
                torch.save(self, os.path.join(log_dir, 'checkpoint_'+str(epoch)+'.pytorch'))

        writer.close()