import datetime
import os

import torch.nn as nn
import torch.nn.functional as F
from neuralnets.networks.unet import UNetEncoder2D, UNetDecoder2D, UNet2D
from neuralnets.util.tools import *
from neuralnets.util.losses import DiceLoss
from neuralnets.util.io import print_frm
from neuralnets.util.metrics import jaccard, accuracy_metrics
from torch.utils.tensorboard import SummaryWriter
from util.losses import feature_regularization_loss


class UNetTS2D(nn.Module):
    """
    2D two-stream U-Net (https://ieeexplore.ieee.org/document/8363602)
    :param optional in_channels: number of input channels
    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional lambda_w: regularization parameter for the weights
    :param optional lambda_o: regularization parameter for the feature representations
    """

    def __init__(self, in_channels=1, coi=(0, 1), feature_maps=64, levels=4, norm="batch", skip_connections=True,
                 lambda_w=0, lambda_o=0):
        super(UNetTS2D, self).__init__()

        self.in_channels = in_channels
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.skip_connections = skip_connections
        self.lambda_w = lambda_w
        self.lambda_o = lambda_o
        self.seg_loss = DiceLoss()
        self.dc_loss = nn.CrossEntropyLoss()

        self.src_encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm)
        self.tar_encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm)
        self.decoder = UNetDecoder2D(self.out_channels, feature_maps=feature_maps, levels=levels,
                                     skip_connections=skip_connections, norm=norm)

        index = 0
        for weight in self.src_encoder.parameters():
            self.register_parameter('a' + str(index), nn.Parameter(torch.ones(weight.shape)))
            self.register_parameter('b' + str(index), nn.Parameter(torch.zeros(weight.shape)))
            index += 1

        self.n_params = 2 * index

    def forward_src(self, inputs):

        # contractive path
        encoder_outputs, encoded = self.src_encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(encoded, encoder_outputs)

        return outputs, ([*encoder_outputs, encoded], decoder_outputs)

    def forward_tar(self, inputs):

        # contractive path
        encoder_outputs, encoded = self.tar_encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(encoded, encoder_outputs)

        return outputs, ([*encoder_outputs, encoded], decoder_outputs)

    def forward(self, inputs):

        # contractive path
        encoder_outputs_src, final_output_src = self.src_encoder(inputs)
        encoder_outputs_tar, final_output_tar = self.tar_encoder(inputs)

        # expansive path
        decoder_outputs_src, outputs_src = self.decoder(final_output_src, encoder_outputs_src)
        decoder_outputs_tar, outputs_tar = self.decoder(final_output_tar, encoder_outputs_tar)

        return outputs_src, outputs_tar

    def get_unet(self, tar=True):
        """
        Get the segmentation network branch
        :param tar: return the target or source branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.encoder.in_channels, coi=self.coi, feature_maps=self.encoder.feature_maps,
                     levels=self.encoder.levels, skip_connections=self.decoder.skip_connections, norm=self.encoder.norm)

        if tar:
            net.encoder.load_state_dict(self.tar_encoder.state_dict())
        else:
            net.encoder.load_state_dict(self.src_encoder.state_dict())
        net.decoder.load_state_dict(self.decoder.state_dict())

        return net

    def train_epoch(self, loader_src, loader_tar_ul, loader_tar_l, optimizer, epoch, augmenter=None, print_stats=1,
                    writer=None, write_images=False, device=0, n_samples_coral=64):
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
        :param n_samples_coral: number of samples selected for CORAL computation
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_weights_cum = 0.0
        loss_feature_cum = 0.0
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
            y_src_pred, f_src = self.forward_src(x_src)
            y_tar_pred, f_tar = self.forward_tar(x_tar_ul)
            loss_seg_src = self.seg_loss(y_src_pred, y_src)
            loss_weights = self.param_regularization_loss(self.src_encoder.parameters(), self.tar_encoder.parameters())
            loss_feature = feature_regularization_loss(f_src[1][-1], f_tar[1][-1], method='coral',
                                                       n_samples=n_samples_coral)
            total_loss = loss_seg_src + self.lambda_w * loss_weights + self.lambda_o * loss_feature
            if loader_tar_l is not None:
                y_tar_l_pred, _ = self.forward_tar(x_tar_l)
                loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
                total_loss = total_loss + loss_seg_tar

            # compute loss
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_weights_cum += loss_weights.data.cpu().numpy()
            loss_feature_cum += loss_feature.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print(
                    '[%s] Epoch %5d - Iteration %5d/%5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss weights: %.6f - Loss feature: %.6f - Loss: %.6f'
                    % (datetime.datetime.now(), epoch, i, len(loader_src.dataset) / loader_src.batch_size, loss_seg_src,
                       loss_seg_tar, self.lambda_w * loss_weights, self.lambda_o * loss_feature, total_loss))

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
        loss_weights_avg = loss_weights_cum / cnt
        loss_feature_avg = loss_feature_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Training Epoch %5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss weights: %.6f  - Loss feature: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_weights_avg, loss_feature_avg,
               total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_weights_avg, loss_feature_avg, total_loss_avg],
                        ['train/' + s for s in
                         ['loss-seg-src', 'loss-seg-tar', 'loss-weights', 'loss-features', 'total-loss']], writer,
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

    def test_epoch(self, loader_src, loader_tar_ul, loader_tar_l, epoch, writer=None, write_images=False, device=0,
                   n_samples_coral=64):
        """
        Trains the network for one epoch
        :param loader_src: source dataloader (labeled)
        :param loader_tar_ul: target dataloader (unlabeled)
        :param loader_tar_l: target dataloader (labeled)
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :param n_samples_coral: number of samples selected for CORAL computation
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.eval()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_weights_cum = 0.0
        loss_feature_cum = 0.0
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
            loss_seg_tar = torch.Tensor([0])
            y_src_pred, f_src = self.forward_src(x_src)
            y_tar_pred, f_tar = self.forward_tar(x_tar_ul)
            loss_seg_src = self.seg_loss(y_src_pred, y_src)
            loss_weights = self.param_regularization_loss(self.src_encoder.parameters(), self.tar_encoder.parameters())
            loss_feature = feature_regularization_loss(f_src[1][-1], f_tar[1][-1], method='coral',
                                                       n_samples=n_samples_coral)
            total_loss = loss_seg_src + self.lambda_w * loss_weights + self.lambda_o * loss_feature
            y_tar_l_pred, _ = self.forward_tar(x_tar_l)
            loss_seg_tar = self.seg_loss(y_tar_l_pred, y_tar_l[:, 0, ...])
            total_loss = total_loss + loss_seg_tar

            # compute loss
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_weights_cum += loss_weights.data.cpu().numpy()
            loss_feature_cum += loss_feature.data.cpu().numpy()
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
        loss_weights_avg = loss_weights_cum / cnt
        loss_feature_avg = loss_feature_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print(
            '[%s] Testing Epoch %5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss weights: %.6f  - Loss feature: %.6f - Loss: %.6f'
            % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, loss_weights_avg, loss_feature_avg,
               total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_seg_src_avg, loss_seg_tar_avg, loss_weights_avg, loss_feature_avg, total_loss_avg,
                         np.mean(js, axis=0), *(np.mean(ams, axis=0))],
                        ['test/' + s for s in
                         ['loss-seg-src', 'loss-seg-tar', 'loss-weights', 'loss-features', 'total-loss', 'jaccard',
                          'accuracy', 'balanced-accuracy', 'precision', 'recall', 'f-score']], writer,
                        epoch=epoch)

            # log images if necessary
            if write_images:
                y_src_pred = F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x_src.data, y_src.data, y_src_pred, x_tar_ul.data],
                              ['test/' + s for s in ['src/x', 'src/y', 'src/y-pred', 'tar/x-ul']], writer, epoch=epoch)
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

    def param_regularization_loss(self, src_params, tar_params):
        """
        Computes the regularization loss on the parameters of the two streams
        :param src_params: parameters in the source encoder
        :param tar_params: parameters in the target encoder
        :return: parameter regularization loss
        """
        params = list(self.named_parameters())[:self.n_params]
        index = 0
        cum_sum = 0
        w_loss = 0
        for src_weight, tar_weight in zip(src_params, tar_params):
            a = params[2 * index][1]
            b = params[2 * index + 1][1]
            d = a.mul(src_weight) + b - tar_weight
            w_loss = w_loss + torch.norm(d, 2)
            cum_sum += np.prod(np.array(d.shape))
            index += 1
        w_loss = w_loss / cum_sum
        return w_loss
