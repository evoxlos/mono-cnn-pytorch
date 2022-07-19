import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent


class LocalBinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, sparsity,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, mid_channels=None):

        super(LocalBinaryConv2d, self).__init__()

        self.sparsity = sparsity  # higher sparsity means more zeros
        _mid_channels = mid_channels if mid_channels else out_channels

        conv2d = nn.Conv2d(in_channels, _mid_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)

        w = torch.bernoulli(torch.ones(conv2d.weight.size()) * 0.5) * 2 - 1
        mask = torch.rand(conv2d.weight.size()) > sparsity
        conv2d.weight = torch.nn.Parameter(torch.mul(w, mask))
        conv2d.weight.requires_grad = False

        self.conv1 = conv2d
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(_mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

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

        child_lines[0] = child_lines[0][:-1]
        child_lines[0] += ', sparsity={})'.format(self.sparsity)

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
        return self.conv2(self.act(self.conv1(x)))
