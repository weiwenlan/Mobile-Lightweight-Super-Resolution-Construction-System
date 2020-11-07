from model import common
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def make_model(args, parent=False):
    return EDSR(args)

class HetConv2d(nn.Module):
    def __init__(self, n_feats, groups=1, p=32):
        super(HetConv2d, self).__init__()
        if n_feats % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.n_feats = n_feats
        self.groups = groups
        self.blocks = nn.ModuleList()
        for i in range(n_feats):
            self.blocks.append(self.make_HetConv2d(i, p))

    def make_HetConv2d(self, n, p):
        layers = nn.ModuleList()
        for i in range(self.n_feats):
            if ((i - n) % (p)) == 0:
                layers.append(nn.Conv2d(1, 1, 3, 1, 1))

            else:
                layers.append(nn.Conv2d(1, 1, 1, 1, 0))
        return layers

    def forward(self, x):
        out = []
        for i in range(0, self.n_feats):
            out_ = self.blocks[i][0](x[:, 0: 1, :, :])
            for j in range(1, self.n_feats):
               out_ += self.blocks[i][j](x[:, j:j + 1, :, :])
            out.append(out_)
        return torch.cat(out, 1)

class Efficient_HetConv2d(nn.Module):
    def __init__(self, n_feats, p=32):
        super(Efficient_HetConv2d, self).__init__()
        if n_feats % p != 0:
            raise ValueError('in_channels must be divisible by p')
        self.conv3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=p)
        self.conv1x1_ = nn.Conv2d(n_feats, n_feats, kernel_size=1, groups=p)
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, res_scale=1, act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1, groups=n_feats, bias=False)))
        #body.append(wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))      
        body.append(act)
        #body.append(
        #   wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))
        body.append(
            Efficient_HetConv2d(n_feats, p=32))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
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
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, wn=wn, res_scale=args.res_scale, act=act))

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

