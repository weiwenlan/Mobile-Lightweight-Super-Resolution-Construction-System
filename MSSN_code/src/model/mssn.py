from model import common
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def make_model(args, parent=False):
    return MODEL(args)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Efficient_HetConv2d(nn.Module):
    def __init__(self, n_feats, wn,p):
        super(Efficient_HetConv2d, self).__init__()
        if n_feats % p != 0:
            raise ValueError('in_channels must be divisible by p')
        self.conv3x3 = wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=p))
        self.conv1x1_ = wn(nn.Conv2d(n_feats, n_feats, kernel_size=1, groups=p))
        self.conv1x1 = wn(nn.Conv2d(n_feats, n_feats, kernel_size=1))

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MobileUnitV2(nn.Module):
    def __init__(self,inp, oup, expand_ratio, wn, res_scale=1,stride=1):
        super(MobileUnitV2, self).__init__()
        self.stride = stride
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.body = nn.Sequential(
                # dw
                wn(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                nn.ReLU6(inplace=True),
                # pw-linear
                wn(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                #nn.BatchNorm2d(oup),
            )
        else:
            self.body = nn.Sequential(
                # pw
                wn(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                nn.ReLU6(inplace=True),
                # dw
                wn(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                wn(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                #nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class CAB(nn.Module):
    def __init__(
        self, n_feats, kernel_size, expand_ratio, wn, res_scale=1, act=nn.ReLU(True)):
        super(CAB, self).__init__()
        body = []
        for i in range(4):
            body.append(
                MobileUnitV2(n_feats, n_feats, expand_ratio, wn=wn, res_scale=res_scale))
    
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)
    
class ASBG(nn.Module):
    def __init__(
        self, n_resblocks, n_feats, kernel_size, expand_ratio, wn, res_scale=1, act=nn.ReLU(True)):
        super(ASBG, self).__init__()
        self.res_scale0 = Scale(1)
        self.res_scale1 = Scale(1)
        self.res_scale2 = Scale(1)
        self.res_scale3 = Scale(1)
        self.res_scale4 = Scale(1)
        self.res_scale5 = Scale(1)

        self.x_scale0 = Scale(1)
        self.x_scale1 = Scale(1)
        self.x_scale2 = Scale(1)
        self.x_scale3 = Scale(1)
        self.x_scale4 = Scale(1)
        self.x_scale5 = Scale(1)

        self.b0 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)
        self.b1 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)
        self.b2 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)
        self.b3 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)
        self.b4 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)
        self.b5 = CAB(n_feats, kernel_size, expand_ratio, wn=wn, res_scale=res_scale, act=act)

    def forward(self, x):
        x0 = self.res_scale0(self.b0(x)) + self.x_scale0(x)
        x1 = self.res_scale1(self.b1(x0)) + self.x_scale1(x)
        x2 = self.res_scale2(self.b2(x1)) + self.x_scale2(x)
        x3 = self.res_scale3(self.b3(x2)) + self.x_scale3(x)
        x4 = self.res_scale2(self.b4(x3)) + self.x_scale4(x)
        x5 = self.res_scale3(self.b5(x4)) + self.x_scale5(x)
        return x5


class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        expand_ratio = args.expand_ratio
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        body.append(
            ASBG(n_resblocks, n_feats, kernel_size, expand_ratio, wn=wn, res_scale=args.res_scale, act=act))

        # define tail module
        tail = []
        out_feats = scale*scale*args.n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x*127.5 + self.rgb_mean.cuda()*255
        return x


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
