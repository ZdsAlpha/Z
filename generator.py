CLASS_TEMPLATE = \
"""
class {name}({super}):
    def __init__(self, {args}):
        super({name}, self).__init__({super_args})

"""


def create_class(name, super, args, super_args):
    return CLASS_TEMPLATE.format(name=name, super=super, args=args, super_args=super_args)


if __name__ == "__main__":
    # Reading base code
    f = open("z_base.py", "r", encoding="utf-8")
    base_code = f.read()
    f.close()
    # Writing z.py
    f = open("z.py", "w", encoding="utf-8")
    f.write(base_code)
    # Extending ZBase
    f.write(create_class("ZLinear", "ZBase",
                         "in_features, out_features, routes, bias=True, non_convex=True",
                         "lambda i, o: nn.Linear(i, o, bias=bias), in_features, out_features, routes, 0, non_convex"))
    for dim in range(1, 4):
        f.write(create_class("ZConv{0}d".format(dim), "ZBase",
                             "in_channels, out_channels, routes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True",
                             "lambda i, o: nn.Conv{0}d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, out_channels, routes, {0}, non_convex".format(dim)))
    for dim in range(1, 4):
        f.write(create_class("ZConvTranspose{0}d".format(dim), "ZBase",
                             "in_channels, out_channels, routes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True",
                             "lambda i, o: nn.ConvTranspose{0}d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, out_channels, routes, {0}, non_convex".format(dim)))
    # Extending StackedBase
    f.write(create_class("StackedLinear", "StackedBase",
                         "in_features, mid_features, out_features, layers, bias=True, activation=F.relu",
                         "lambda i, o, l: nn.Linear(i, o, bias=bias), in_features, mid_features, out_features, layers, activation"))
    for dim in range(1, 4):
        f.write(create_class("StackedConv{0}d".format(dim), "StackedBase",
                             "in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu",
                             "lambda i, o, l: nn.Conv{0}d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    for dim in range(1, 4):
        f.write(create_class("StackedConvTranspose{0}d".format(dim), "StackedBase",
                             "in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu",
                             "lambda i, o, l: nn.ConvTranspose{0}d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    # StackedBase + ZBase
    f.write(create_class("ZStackedLinear", "StackedBase",
                         "in_features, mid_features, out_features, routes, layers, bias=True, non_convex=True, activation=F.relu",
                         "lambda i, o, l: ZLinear(i, o, routes, bias, non_convex), in_features, mid_features, out_features, layers, activation"))
    for dim in range(1, 4):
        f.write(create_class("ZStackedConv{0}d".format(dim), "StackedBase",
                             "in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=F.relu",
                             "lambda i, o, l: ZConv{0}d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    for dim in range(1, 4):
        f.write(create_class("ZStackedConvTranspose{0}d".format(dim), "StackedBase",
                             "in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=F.relu",
                             "lambda i, o, l: ZConvTranspose{0}d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    # Extending DenseBase
    f.write(create_class("DenseLinear", "DenseBase",
                         "in_features, mid_features, out_features, layers, bias=True, activation=F.relu",
                         "lambda i, o, l: nn.Linear(i, o, bias=bias), in_features, mid_features, out_features, layers, activation"))
    for dim in range(1, 4):
        f.write(create_class("DenseConv{0}d".format(dim), "DenseBase",
                             "in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activation=F.relu",
                             "lambda i, o, l: nn.Conv{0}d(i, o, kernel_size, stride, padding, dilation, groups, bias, padding_mode), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    for dim in range(1, 4):
        f.write(create_class("DenseConvTranspose{0}d".format(dim), "DenseBase",
                             "in_channels, mid_channels, out_channels, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', activation=F.relu",
                             "lambda i, o, l: nn.ConvTranspose{0}d(i, o, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    # DenseBase + ZBase
    f.write(create_class("ZStackedLinear", "StackedBase",
                         "in_features, mid_features, out_features, routes, layers, bias=True, non_convex=True, activation=F.relu",
                         "lambda i, o, l: ZLinear(i, o, routes, bias, non_convex), in_features, mid_features, out_features, layers, activation"))
    for dim in range(1, 4):
        f.write(create_class("ZDenseConv{0}d".format(dim), "DenseBase",
                             "in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', non_convex=True, activation=F.relu",
                             "lambda i, o, l: ZConv{0}d(i, o, routes, kernel_size, stride, padding, dilation, groups, bias, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    for dim in range(1, 4):
        f.write(create_class("ZDenseConvTranspose{0}d".format(dim), "DenseBase",
                             "in_channels, mid_channels, out_channels, routes, layers, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', non_convex=True, activation=F.relu",
                             "lambda i, o, l: ZConvTranspose{0}d(i, o, routes, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, non_convex), in_channels, mid_channels, out_channels, layers, activation".format(dim)))
    f.close()
