import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent


class PerturbativeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, noise_level=0.01, noisy_train=False, noisy_eval=False):

        super(PerturbativeConv2d, self).__init__()

        self.noise_level = noise_level  # severity of the added noise
        self.noisy_train = noisy_train
        self.noisy_eval = noisy_eval

        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

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

        if self.noisy_train or self.noisy_eval:
            extra_lines.append('noise_level={}, noisy_train={}, noisy_eval={}'.format(
                self.noise_level, self.noisy_train, self.noisy_eval))
        else:
            extra_lines.append('noise_level={}'.format(self.noise_level))

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
        if not hasattr(self, 'noise'):
            self.register_buffer('noise', self.noise_level * torch.randn(x.data[0].size()).to(x.device))

        if (self.train and self.noisy_train) or self.noisy_eval:
            noise = self.noise_level * torch.randn(x.data[0].size())
        else:
            noise = self.__getattr__('noise')

        return self.conv1(self.act(torch.add(x, noise)))
