import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

_EPS = 1e-5


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def extra_repr(self) -> str:
        return 'groups={}'.format(self.groups)

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class ShieldMonomialConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, num_seeds, exp_range, fanout_factor, kernel_size,
                 stride=1, padding=0, dilation=1, num_terms=1, mono_groups=1, mono_bias=False,
                 onebyone=True):

        # if you are not familiar with the code, this commend is recommended to be left on
        assert out_channels == int(num_seeds * fanout_factor)

        super(ShieldMonomialConv2d, self).__init__(
            in_channels, num_seeds, kernel_size, stride, padding, dilation, groups=1, bias=False)

        self.num_terms = num_terms
        self.exp_range = exp_range
        self.fanout_factor = fanout_factor
        self.mono_groups = mono_groups

        self.register_buffer(
            'mono_exponent', torch.tensor(
                exp_range[0] + (exp_range[1] - exp_range[0]) * torch.rand(
                    num_terms, out_channels, in_channels // mono_groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))

        if mono_bias:
            self.register_buffer(
                'mono_bias', torch.tensor(
                    torch.rand(out_channels, in_channels // mono_groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))

        if onebyone:
            self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
            self.act = nn.ReLU(inplace=True)
            self.onebyone = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.onebyone = None

    # override the print func to print out rbf kernel as well
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        if hasattr(self, 'mono_bias'):
            extra_lines[0] += ', num_terms={}, range=({}, {}), fanout_factor={}, mono_bias=True'.format(
                self.num_terms, self.exp_range[0], self.exp_range[1], self.fanout_factor)
        else:
            extra_lines[0] += ', num_terms={}, range=({}, {}), fanout_factor={}, mono_bias=False'.format(
                self.num_terms, self.exp_range[0], self.exp_range[1], self.fanout_factor)

        if self.onebyone:
            extra_lines[0] += ', 1x1_conv=True'
        else:
            extra_lines[0] += ', 1x1_conv=False'

        lines = extra_lines + child_lines

        main_str = self._get_name() + '('

        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'

        return main_str

    def forward(self, x):
        mono_exponent = self.__getattr__('mono_exponent')

        tmp = self.weight.repeat(self.fanout_factor, 1, 1, 1)
        tmp2 = torch.sum(tmp.repeat(self.num_terms, 1, 1, 1, 1).abs().pow(mono_exponent), dim=0)

        if hasattr(self, 'mono_bias'):
            tmp2 += self.__getattr__('mono_bias')

        w = torch.mul(tmp2, tmp.sign())
        # weight standardization from https://github.com/joe-siyuan-qiao/WeightStandardization
        weight_mean = w.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = w - weight_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + _EPS
        w = w / std.expand_as(w)

        features = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.mono_groups)

        if self.onebyone:
            return self.onebyone(self.act(self.gn(features)))
        else:
            return features


class ShieldRBFFamilyConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, num_seeds, eps_range, fanout_factor, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, rbf='gaussian', onebyone=False
                 ):

        assert out_channels == int(num_seeds * fanout_factor)

        super(ShieldRBFFamilyConv2d, self).__init__(
            in_channels, num_seeds, kernel_size, stride, padding, dilation, groups, bias)

        self.eps_range = eps_range
        self.fanout_factor = fanout_factor
        # self.mid_chs = int(num_seeds * fanout_factor)

        self.register_buffer(
            'epsilon', torch.tensor(
                eps_range[0] + (eps_range[1] - eps_range[0]) * torch.rand(
                    out_channels, in_channels // groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))

        self.rbf = rbf  # remember which radial basis function to use

        if onebyone:
            self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
            self.act = nn.ReLU(inplace=True)
            self.onebyone = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.onebyone = None

    # override the print func to print out rbf kernel as well
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        extra_lines[0] += ', rbf={}, eps_range=({}, {})'.format(self.rbf, self.eps_range[0], self.eps_range[1])

        if self.onebyone:
            extra_lines[0] += ', 1x1_conv=True'
        else:
            extra_lines[0] += ', 1x1_conv=False'

        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def forward(self, x):

        tmp = self.weight.repeat(self.fanout_factor, 1, 1, 1)

        if self.rbf == 'gaussian':
            w = torch.exp(-(torch.pow(tmp.mul(self.__getattr__('epsilon')), 2)))
        elif self.rbf == 'multiquadric':
            w = torch.sqrt(1 + torch.pow(tmp.mul(self.__getattr__('epsilon')), 2))
        elif self.rbf == 'inverse_quadratic':
            w = torch.reciprocal(1 + torch.pow(tmp.mul(self.__getattr__('epsilon')), 2))
        elif self.rbf == 'inverse_multiquadric':
            w = torch.reciprocal(torch.sqrt(1 + torch.pow(tmp.mul(self.__getattr__('epsilon')), 2)))
        else:
            raise NotImplementedError

        # weight standardization from https://github.com/joe-siyuan-qiao/WeightStandardization
        weight_mean = w.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = w - weight_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + _EPS
        w = w / std.expand_as(w)

        features = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return features


if __name__ == '__main__':
    from torchprofile import profile_macs
    import warnings

    warnings.filterwarnings("ignore")

    conv = ShieldMonomialConv2d(64, 64, num_seeds=4, exp_range=(1, 7),
                                fanout_factor=16, kernel_size=3, stride=1, padding=1, num_terms=1, onebyone=True)

    # conv = ShieldRBFFamilyConv2d(64, 64, num_seeds=4, eps_range=(1, 7), fanout_factor=16,
    #                              kernel_size=3, stride=1, padding=1, rbf='gaussian', onebyone=True)
    print(conv)

    param_count = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    param_count_full = sum(p.numel() for p in conv.parameters())

    print(param_count / 1)
    print(param_count_full / 1)

    data = torch.rand(1, 64, 32, 32)
    y = conv(data)
    try:
        for v in y[1]:
            print(v.size())
    except:
        print(y.size())

    flops = profile_macs(conv, data) / 1e6
    print(flops)