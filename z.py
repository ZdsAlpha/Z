import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x, hardness=1, dim=0):
    return torch.sum(x * torch.softmax(hardness * x, dim=dim), dim=dim)


class ZBase(nn.Module):
    def __init__(self, layer_initializer, in_features, out_features, routes, dims=0, non_convex=True):
        super(ZBase, self).__init__()
        self.layer_initializer = layer_initializer
        self.in_features = in_features
        self.out_features = out_features
        self.routes = routes
        self.dims = dims
        self.a = layer_initializer(in_features, out_features * routes)
        self.l = layer_initializer(in_features, out_features * routes)
        self.z = nn.Parameter(torch.zeros(*tuple([1, out_features, routes if non_convex else 1] + [1] * dims)).normal_(0, 1),
                              requires_grad=True)

    def forward(self, *x):
        a = self.a(*x)
        l = self.l(*x)
        shape = (a.shape[0], self.out_features, self.routes, *a.shape[2:])
        return torch.sum(a.view(*shape) * torch.softmax(self.z * l.view(*shape), dim=2), dim=2)


class StackedBase(nn.Module):
    def __init__(self, layer_initializer, in_features, mid_features, out_features, layers, activation=F.relu):
        super(StackedBase, self).__init__()
        assert layers >= 2
        self.layer_initializer = layer_initializer
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.total_layers = layers
        self.activation = activation
        self.in_layer = layer_initializer(in_features, mid_features, 1)
        self.mid_layers = []
        self.out_layer = layer_initializer(mid_features, out_features, layers)
        for i in range(1, layers - 1):
            self.mid_layers.append(layer_initializer(mid_features, mid_features, i + 1))
        self.mid_layers = nn.Sequential(*self.mid_layers)

    def forward(self, x):
        x = self.in_layer(x) if self.activation is None else self.activation(self.in_layer(x))
        for i, layer in enumerate(self.mid_layers):
            x = layer(x) if self.activation is None else self.activation(layer(x))
        return self.out_layer(x)


class ResBase(nn.Module):
    def __init__(self, layer_initializer, in_features, mid_features, out_features, layers, activation=F.relu):
        super(ResBase, self).__init__()
        assert layers >= 2
        self.layer_initializer = layer_initializer
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.total_layers = layers
        self.activation = activation
        self.in_layer = layer_initializer(in_features, mid_features, 1)
        self.mid_layers = []
        self.out_layer = layer_initializer(mid_features, out_features, layers)
        for i in range(1, layers - 1):
            self.mid_layers.append(layer_initializer(mid_features, mid_features, i + 1))
        self.mid_layers = nn.Sequential(*self.mid_layers)

    def forward(self, x):
        _x = self.in_layer(x) if self.activation is None else self.activation(self.in_layer(x))
        x = x + _x if x.shape == _x.shape else _x
        for i, layer in enumerate(self.mid_layers):
            _x = layer(x) if self.activation is None else self.activation(layer(x))
            x = x + _x if x.shape == _x.shape else _x
        _x = self.out_layer(x)
        x = x + _x if x.shape == _x.shape else _x
        return x


class ReBase(nn.Module):
    def __init__(self, layer_initializer, in_features, mid_features, out_features, layers, activation=F.relu, dims=0):
        super(ReBase, self).__init__()
        assert layers >= 2
        self.layer_initializer = layer_initializer
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.total_layers = layers
        self.activation = activation
        self.in_layer = layer_initializer(in_features, mid_features, 1)
        self.in_weight = nn.Parameter(torch.zeros(*tuple([1, mid_features] + [1] * dims))) if in_features == mid_features else None
        self.mid_layers = []
        self.mid_weights = []
        self.out_layer = layer_initializer(mid_features, out_features, layers)
        self.out_weight = nn.Parameter(torch.zeros(*tuple([1, mid_features] + [1] * dims))) if mid_features == out_features else None
        for i in range(1, layers - 1):
            self.mid_layers.append(layer_initializer(mid_features, mid_features, i + 1))
            self.mid_weights.append(nn.Parameter(torch.zeros(*tuple([1, mid_features] + [1] * dims))))
        self.mid_layers = nn.Sequential(*self.mid_layers)
        self.mid_weights = nn.ParameterList(self.mid_weights)

    def forward(self, x):
        _x = self.in_layer(x) if self.activation is None else self.activation(self.in_layer(x))
        x = x + self.in_weight * _x if self.in_weight is not None else _x
        for i, layer in enumerate(self.mid_layers):
            _x = layer(x) if self.activation is None else self.activation(layer(x))
            x = x + self.mid_weights[i] * _x
        _x = self.out_layer(x)
        x = x + self.out_weight * _x if self.out_weight is not None else _x
        return x


class DenseBase(nn.Module):
    def __init__(self, layer_initializer, in_features, mid_features, out_features, layers, activation=F.relu):
        super(DenseBase, self).__init__()
        assert layers >= 2
        self.layer_initializer = layer_initializer
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.total_layers = layers
        self.activation = activation
        self.in_layer = layer_initializer(in_features, mid_features, 1)
        self.mid_layers = []
        self.out_layer = layer_initializer(in_features + (layers - 1) * mid_features, out_features, layers)
        for i in range(1, layers - 1):
            self.mid_layers.append(layer_initializer(in_features + i * mid_features, mid_features, i + 1))
        self.mid_layers = nn.Sequential(*self.mid_layers)

    def forward(self, x):
        _x = self.in_layer(x) if self.activation is None else self.activation(self.in_layer(x))
        x = torch.cat([x, _x], dim=1)
        for i, layer in enumerate(self.mid_layers):
            _x = layer(x) if self.activation is None else self.activation(layer(x))
            x = torch.cat([x, _x], dim=1)
        return self.out_layer(x)


class UBase(nn.Module):
    def __init__(self, encoder_initializer, decoder_initializer, downsampler_initializer, upsampler_initializer, shape_adjustment,
                 in_channels, out_channels, filters, layers, activation=F.relu):
        super(UBase, self).__init__()
        assert len(filters) >= 1
        if type(layers) is int:
            layers = [layers] * len(filters)
        for layer in layers:
            assert layer >= 2
        self.encoder_initializer = encoder_initializer
        self.decoder_initializer = decoder_initializer
        self.downsampler_initializer = downsampler_initializer
        self.upsampler_initializer = upsampler_initializer
        self.shape_adjustment = shape_adjustment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.layers = layers
        self.activation = activation
        encoders = []
        decoders = []
        dsamplers = []
        usamplers = []
        for i, filter in enumerate(filters):
            encoders = encoders + [encoder_initializer(in_channels, filter, filter, i + 1)]
            decoders = [decoder_initializer(2 * filter, filter, out_channels, i + 1)] + decoders
            dsamplers = dsamplers + [downsampler_initializer(filter, i + 1) if downsampler_initializer is not None else None]
            usamplers = [upsampler_initializer(filter, i + 1) if upsampler_initializer is not None else None] + usamplers
            in_channels = filter
            out_channels = filter
        self.encoders = nn.Sequential(*encoders)
        self.decoders = nn.Sequential(*decoders)
        self.dsampler = nn.Sequential(*dsamplers)
        self.usampler = nn.Sequential(*usamplers)

    def forward(self, x):
        tensors = []
        shapes = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x) if self.activation is None else self.activation(encoder(x))
            tensors.append(x)
            shapes.append(x.shape)
            x = self.dsampler[i](x) if self.activation is None else self.activation(self.dsampler[i](x))
        for i, decoder in enumerate(self.decoders):
            shape = shapes.pop()
            x = self.usampler[i](x) if self.activation is None else self.activation(self.usampler[i](x))
            x = self.shape_adjustment(x, shape)
            x = torch.cat([tensors.pop(), x], dim=1)
            if self.activation is None or i == len(self.decoders) - 1:
                x = decoder(x)
            else:
                x = self.activation(decoder(x))
        return x


class Pad(nn.Module):
    def __init__(self, width, height, mode="constant", value=0):
        super(Pad, self).__init__()
        self.width = width
        self.height = height
        self.mode = mode
        self.value = value

    def forward(self, x):
        return Pad.pad(x, self.width, self.height, self.mode, self.value)

    @staticmethod
    def pad(x, w, h, mode="constant", value=0):
        _, _, height, width = x.shape
        assert height <= h
        assert width <= w
        d_h = h - height
        d_w = w - width
        if d_h == 0 and d_w == 0:
            return x
        else:
            lp = d_w // 2
            rp = d_w - lp
            tp = d_h // 2
            bp = d_h - tp
            x = F.pad(x, (lp, rp, tp, bp), mode=mode, value=value)
            return x


class SoftPool2d(nn.Module):
    def __init__(self, features, kernel_size=2, stride=2):
        super(SoftPool2d, self).__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(stride) is int:
            stride = (stride, stride)
        assert type(kernel_size) is tuple
        assert type(stride) is tuple
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.z = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x = x.view(x.shape[0], self.features, self.kernel_size[0] * self.kernel_size[1], -1)
        x = softmax(x, dim=2, hardness=self.z)
        return x.view(x.shape[0], self.features,
                       (h - self.kernel_size[1]) // self.stride[1] + 1, (w - self.kernel_size[0]) // self.stride[0] + 1)


class ZLinear(ZBase):
    def __init__(self, in_features, out_features, routes, bias=True, non_convex=True):
        super(ZLinear, self).__init__(lambda i, o: nn.Linear(i, o, bias=bias), in_features, out_features, routes, 0, non_convex)


class ZConv1d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True):
        super(ZConv1d, self).__init__(lambda i, o: nn.Conv1d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, out_channels, routes, 1, non_convex)


class ZConv2d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True):
        super(ZConv2d, self).__init__(lambda i, o: nn.Conv2d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, out_channels, routes, 2, non_convex)


class ZConv3d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True):
        super(ZConv3d, self).__init__(lambda i, o: nn.Conv3d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, out_channels, routes, 3, non_convex)


class ZConvTranspose1d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True):
        super(ZConvTranspose1d, self).__init__(lambda i, o: nn.ConvTranspose1d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, out_channels, routes, 1, non_convex)


class ZConvTranspose2d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True):
        super(ZConvTranspose2d, self).__init__(lambda i, o: nn.ConvTranspose2d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, out_channels, routes, 2, non_convex)


class ZConvTranspose3d(ZBase):
    def __init__(self, in_channels, out_channels, routes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True):
        super(ZConvTranspose3d, self).__init__(lambda i, o: nn.ConvTranspose3d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, out_channels, routes, 3, non_convex)


class StackedLinear(StackedBase):
    def __init__(self, in_features, mid_features, out_features, layers, bias=True, activation=F.relu):
        super(StackedLinear, self).__init__(lambda i, o, l: nn.Linear(i, o, bias=bias), in_features, mid_features, out_features, layers, activation)


class StackedConv1d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(StackedConv1d, self).__init__(lambda i, o, l: nn.Conv1d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class StackedConv2d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(StackedConv2d, self).__init__(lambda i, o, l: nn.Conv2d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class StackedConv3d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(StackedConv3d, self).__init__(lambda i, o, l: nn.Conv3d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class StackedConvTranspose1d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(StackedConvTranspose1d, self).__init__(lambda i, o, l: nn.ConvTranspose1d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class StackedConvTranspose2d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(StackedConvTranspose2d, self).__init__(lambda i, o, l: nn.ConvTranspose2d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class StackedConvTranspose3d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(StackedConvTranspose3d, self).__init__(lambda i, o, l: nn.ConvTranspose3d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedLinear(StackedBase):
    def __init__(self, in_features, mid_features, out_features, routes, layers, bias=True, non_convex=True, activation=None):
        super(ZStackedLinear, self).__init__(lambda i, o, l: ZLinear(i, o, routes, bias, non_convex), in_features, mid_features, out_features, layers, activation)


class ZStackedConv1d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConv1d, self).__init__(lambda i, o, l: ZConv1d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedConv2d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConv2d, self).__init__(lambda i, o, l: ZConv2d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedConv3d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConv3d, self).__init__(lambda i, o, l: ZConv3d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedConvTranspose1d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConvTranspose1d, self).__init__(lambda i, o, l: ZConvTranspose1d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedConvTranspose2d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConvTranspose2d, self).__init__(lambda i, o, l: ZConvTranspose2d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedConvTranspose3d(StackedBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZStackedConvTranspose3d, self).__init__(lambda i, o, l: ZConvTranspose3d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class DenseLinear(DenseBase):
    def __init__(self, in_features, mid_features, out_features, layers, bias=True, activation=F.relu):
        super(DenseLinear, self).__init__(lambda i, o, l: nn.Linear(i, o, bias=bias), in_features, mid_features, out_features, layers, activation)


class DenseConv1d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(DenseConv1d, self).__init__(lambda i, o, l: nn.Conv1d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class DenseConv2d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(DenseConv2d, self).__init__(lambda i, o, l: nn.Conv2d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class DenseConv3d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu):
        super(DenseConv3d, self).__init__(lambda i, o, l: nn.Conv3d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class DenseConvTranspose1d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(DenseConvTranspose1d, self).__init__(lambda i, o, l: nn.ConvTranspose1d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class DenseConvTranspose2d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(DenseConvTranspose2d, self).__init__(lambda i, o, l: nn.ConvTranspose2d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class DenseConvTranspose3d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu):
        super(DenseConvTranspose3d, self).__init__(lambda i, o, l: nn.ConvTranspose3d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation)


class ZStackedLinear(StackedBase):
    def __init__(self, in_features, mid_features, out_features, routes, layers, bias=True, non_convex=True, activation=None):
        super(ZStackedLinear, self).__init__(lambda i, o, l: ZLinear(i, o, routes, bias, non_convex), in_features, mid_features, out_features, layers, activation)


class ZDenseConv1d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConv1d, self).__init__(lambda i, o, l: ZConv1d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZDenseConv2d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConv2d, self).__init__(lambda i, o, l: ZConv2d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZDenseConv3d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConv3d, self).__init__(lambda i, o, l: ZConv3d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZDenseConvTranspose1d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConvTranspose1d, self).__init__(lambda i, o, l: ZConvTranspose1d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZDenseConvTranspose2d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConvTranspose2d, self).__init__(lambda i, o, l: ZConvTranspose2d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


class ZDenseConvTranspose3d(DenseBase):
    def __init__(self, in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=None):
        super(ZDenseConvTranspose3d, self).__init__(lambda i, o, l: ZConvTranspose3d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation)


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