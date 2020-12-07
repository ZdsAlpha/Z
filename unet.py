from z import *


class UNet2d(UBase):
    def __init__(self, in_channels, out_channels, filters=[64, 128], layers=2, kernel_size=3,
                 down_factor=2, activation=F.relu):
        self.kernel_size = kernel_size
        self.down_factor = down_factor
        super(UNet2d, self).__init__(self.encoder_initializer, self.decoder_initializer,
                                     self.downsampler_initializer, self.upsampler_initializer, self.shape_adjustment,
                                     in_channels, out_channels, filters, layers, activation)

    def encoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return StackedConv2d(in_channels, mid_channels, out_channels, self.layers[id - 1], self.kernel_size,
                             padding=self.kernel_size // 2, activation=self.activation)

    def decoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return StackedConvTranspose2d(in_channels, mid_channels, out_channels, self.layers[id - 1], self.kernel_size,
                                      padding=self.kernel_size // 2, activation=self.activation)

    def downsampler_initializer(self, filter, id):
        return nn.Conv2d(filter, filter, kernel_size=self.down_factor, stride=self.down_factor)

    def upsampler_initializer(self, filter, id):
        return nn.ConvTranspose2d(filter, filter, kernel_size=self.down_factor, stride=self.down_factor)

    def shape_adjustment(self, x, shape):
        return Pad.pad(x, shape[-1], shape[-2])


class ZUNet2d(UBase):
    def __init__(self, in_channels, out_channels, routes, filters=[64, 128], layers=2, kernel_size=3,
                 down_factor=2, activation=None):
        self.kernel_size = kernel_size
        self.down_factor = down_factor
        self.routes = routes
        super(ZUNet2d, self).__init__(self.encoder_initializer, self.decoder_initializer,
                                     self.downsampler_initializer, self.upsampler_initializer, self.shape_adjustment,
                                     in_channels, out_channels, filters, layers, activation)

    def encoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return ZStackedConv2d(in_channels, mid_channels, out_channels, self.routes, self.layers[id - 1], self.kernel_size,
                             padding=self.kernel_size // 2, activation=self.activation)

    def decoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return ZStackedConvTranspose2d(in_channels, mid_channels, out_channels, self.routes, self.layers[id - 1], self.kernel_size,
                                      padding=self.kernel_size // 2, activation=self.activation)

    def downsampler_initializer(self, filter, id):
        return ZConv2d(filter, filter, self.routes, kernel_size=self.down_factor, stride=self.down_factor)

    def upsampler_initializer(self, filter, id):
        return ZConvTranspose2d(filter, filter, self.routes, kernel_size=self.down_factor, stride=self.down_factor)

    def shape_adjustment(self, x, shape):
        return Pad.pad(x, shape[-1], shape[-2])


class DenseUNet2d(UBase):
    def __init__(self, in_channels, out_channels, filters=[64, 128], layers=2, kernel_size=3,
                 down_factor=2, activation=F.relu):
        self.kernel_size = kernel_size
        self.down_factor = down_factor
        super(DenseUNet2d, self).__init__(self.encoder_initializer, self.decoder_initializer,
                                     self.downsampler_initializer, self.upsampler_initializer, self.shape_adjustment,
                                     in_channels, out_channels, filters, layers, activation)

    def encoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return DenseConv2d(in_channels, mid_channels, out_channels, self.layers[id - 1], self.kernel_size,
                             padding=self.kernel_size // 2, activation=self.activation)

    def decoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return DenseConvTranspose2d(in_channels, mid_channels, out_channels, self.layers[id - 1], self.kernel_size,
                                      padding=self.kernel_size // 2, activation=self.activation)

    def downsampler_initializer(self, filter, id):
        return nn.Conv2d(filter, filter, kernel_size=self.down_factor, stride=self.down_factor)

    def upsampler_initializer(self, filter, id):
        return nn.ConvTranspose2d(filter, filter, kernel_size=self.down_factor, stride=self.down_factor)

    def shape_adjustment(self, x, shape):
        return Pad.pad(x, shape[-1], shape[-2])


class ZDenseUNet2d(UBase):
    def __init__(self, in_channels, out_channels, routes, filters=[64, 128], layers=2, kernel_size=3,
                 down_factor=2, activation=None):
        self.kernel_size = kernel_size
        self.down_factor = down_factor
        self.routes = routes
        super(ZDenseUNet2d, self).__init__(self.encoder_initializer, self.decoder_initializer,
                                     self.downsampler_initializer, self.upsampler_initializer, self.shape_adjustment,
                                     in_channels, out_channels, filters, layers, activation)

    def encoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return ZDenseConv2d(in_channels, mid_channels, out_channels, self.routes, self.layers[id - 1], self.kernel_size,
                             padding=self.kernel_size // 2, activation=self.activation)

    def decoder_initializer(self, in_channels, mid_channels, out_channels, id):
        return ZDenseConvTranspose2d(in_channels, mid_channels, out_channels, self.routes, self.layers[id - 1], self.kernel_size,
                                      padding=self.kernel_size // 2, activation=self.activation)

    def downsampler_initializer(self, filter, id):
        return ZConv2d(filter, filter, self.routes, kernel_size=self.down_factor, stride=self.down_factor)

    def upsampler_initializer(self, filter, id):
        return ZConvTranspose2d(filter, filter, self.routes, kernel_size=self.down_factor, stride=self.down_factor)

    def shape_adjustment(self, x, shape):
        return Pad.pad(x, shape[-1], shape[-2])