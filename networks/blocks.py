
import torch
import torch.nn as nn

# 2D convolution layer with relu activation
class ConvRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1, dropout=0.0):
        super(ConvRelu2D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0

        if dropout > 0.0:
            self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.Dropout2d(p=dropout),
                                      nn.ReLU(),)
        else:
            self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 3D convolution layer with relu activation
class ConvRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1, dropout=0.0):
        super(ConvRelu3D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0
        if dropout > 0.0:
            self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.Dropout3d(p=dropout),
                                      nn.ReLU(),)
        else:
            self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolution layer with batch normalization and relu activation
class ConvBatchNormRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1, dropout=0.0):
        super(ConvBatchNormRelu2D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0

        if dropout > 0.0:
            self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(out_channels)),
                                      nn.Dropout2d(p=dropout),
                                      nn.ReLU(),)
        else:
            self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(out_channels)),
                                      nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 3D convolution layer with batch normalization and relu activation
class ConvBatchNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1, dropout=0.0):
        super(ConvBatchNormRelu3D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0
        if dropout > 0.0:
            self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm3d(int(out_channels)),
                                      nn.Dropout3d(p=dropout),
                                      nn.ReLU(),)
        else:
            self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=p, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm3d(int(out_channels)),
                                      nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolution block of the classical unet
class UNetConvBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', batch_norm=True, dropout=0.0):
        super(UNetConvBlock2D, self).__init__()

        if dropout>0: # no batch norm with dropout
            batch_norm = False

        if batch_norm:
            self.conv1 = ConvBatchNormRelu2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
            self.conv2 = ConvBatchNormRelu2D(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
        else:
            self.conv1 = ConvRelu2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
            self.conv2 = ConvRelu2D(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

# 3D convolution block of the classical unet
class UNetConvBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', batch_norm=True, dropout=0.0):
        super(UNetConvBlock3D, self).__init__()

        if dropout>0: # no batch norm with dropout
            batch_norm = False

        if batch_norm:
            self.conv1 = ConvBatchNormRelu3D(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
            self.conv2 = ConvBatchNormRelu3D(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
        else:
            self.conv1 = ConvRelu3D(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)
            self.conv2 = ConvRelu3D(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

# 2D upsampling block of the classical unet:
# upsamples the input and concatenates with another input
class UNetUpSamplingBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock2D, self).__init__()

        if deconv: # use transposed convolution
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else: # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)

# 3D upsampling block of the classical unet:
# upsamples the input and concatenates with another input
class UNetUpSamplingBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock3D, self).__init__()

        if deconv: # use transposed convolution
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else: # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)