import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PSconvResidualBlock(nn.Module):
    def __init__(self, input_num, channel_num, dilation=1):
        super(PSconvResidualBlock, self).__init__()
        self.conv1 = PSConv2d(input_num, channel_num, 3, stride=1, padding=1, dilation=dilation, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = PSConv2d(channel_num, channel_num, 3, stride=1, padding=1, dilation=dilation, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = y + x
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)

class PSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, parts=4, bias=False):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // parts
        _out_channels = out_channels // parts
        for i in range(parts):
            self.mask[i * _out_channels: (i + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
            self.mask[(i + parts//2)%parts * _out_channels: ((i + parts//2)%parts + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        #torchvision.utils.save_image(squeeze_tensor, "./spatial/squeeze_tensor.jpg")
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class Group(nn.Module):
    def __init__(self, inc, outc, dilation=2):
        super(Group, self).__init__()
        self.res1 = PSconvResidualBlock(inc, outc, dilation=dilation)
        self.act1 = nn.ReLU(inplace=True)
        self.res2 = PSconvResidualBlock(inc, outc, dilation=dilation)
        #self.res3 = PSconvResidualBlock(inc, outc, dilation=dilation)
        #self.conv =  PSConv2d(outc*3, outc, 3, stride=1, padding=1, bias=False)
        #self.channel = ChannelSELayer(outc)
        self.spatialSElayer = SpatialSELayer(outc)
    def forward(self, x):
        res1 = self.act1(self.res1(x))
        res2 = self.res2(res1)
        #res3 = self.res3(res2)
        #out = self.conv(torch.cat((res1, res2, res3),dim=1))
        #out = self.spatialSElayer(out)
        #res = self.channel(res2)
        out = self.spatialSElayer(res2)
        return x+out
        

class PSConv(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super(PSConv, self).__init__()
        self.bottom_layer = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )
        self.g1 = Group(inc=64, outc=64, dilation=2)
        self.g2 = Group(inc=64, outc=64, dilation=2)
        self.g3 = Group(inc=64, outc=64, dilation=2)
        self.channel = ChannelSELayer(64)
        
        self.top_layer = nn.Sequential(
            #nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.Conv2d(64, out_c, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        bottom_layer = self.bottom_layer(x)
        g1 = self.g1(bottom_layer)
        #print("g1 is out")
        #torchvision.utils.save_image(g1, "./spatial/g1.jpg")
        g2 = self.g2(g1)
        #torchvision.utils.save_image(g2, "./spatial/g2.jpg")
        g3 = self.g3(g2)
        #torchvision.utils.save_image(g3, "./spatial/g3.jpg")
        channel_out = self.channel(g3)
        #torchvision.utils.save_image(channel_out, "./spatial/channel_out.jpg")
        out = self.top_layer(channel_out)
        #torchvision.utils.save_image(out, "./spatial/out.jpg")
        return out #x+out
        
        