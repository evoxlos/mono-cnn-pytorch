'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.shift import ShiftConv2d
from models.layers.ghostconv import GhostConv2d
from models.layers.lbconv import LocalBinaryConv2d
from models.layers.perturbconv import PerturbativeConv2d
from models.layers.monoconv import ShieldMonomialConv2d, ShieldRBFFamilyConv2d


__all__ = [
    'resnet18', 'monomial_resnet18', 'gaussian_resnet18', 'multiquadric_resnet18',
    'inverse_quadratic_resnet18', 'inverse_multiquadric_resnet18',
    'local_binary_resnet18', 'perturbative_resnet18', 'shift_resnet18', 'ghost_resnet18',
    'resnet34', 'monomial_resnet34', 'local_binary_resnet34', 'perturbative_resnet34',
    'shift_resnet34', 'ghost_resnet34'
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MonomialBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_terms=3, exp_range=(1, 10), exp_factor=2, mono_bias=False,
                 onebyone=False):
        super(MonomialBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = ShieldMonomialConv2d(
                in_planes, planes, num_seeds=planes // exp_factor, num_terms=num_terms,
                exp_range=exp_range, fanout_factor=exp_factor, kernel_size=3, stride=1, padding=1,
                mono_bias=mono_bias, onebyone=onebyone
            )

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ShieldMonomialConv2d(
            planes, planes, num_seeds=planes // exp_factor, num_terms=num_terms,
            exp_range=exp_range, fanout_factor=exp_factor, kernel_size=3, stride=1, padding=1,
            mono_bias=mono_bias, onebyone=onebyone
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RBFFamilyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, exp_range=(1, 10), exp_factor=2, rbf='gaussian', onebyone=False):
        super(RBFFamilyBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = ShieldRBFFamilyConv2d(
                in_planes, planes, num_seeds=planes // exp_factor, eps_range=exp_range, fanout_factor=exp_factor,
                kernel_size=3, stride=1, padding=1, bias=False, rbf=rbf, onebyone=onebyone)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ShieldRBFFamilyConv2d(
            planes, planes, num_seeds=planes // exp_factor, eps_range=exp_range, fanout_factor=exp_factor,
            kernel_size=3, stride=1, padding=1, bias=False, rbf=rbf, onebyone=onebyone)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LocalBinaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.1):
        super(LocalBinaryBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = LocalBinaryConv2d(in_planes, planes, sparsity, kernel_size=3, stride=1, padding=1, bias=False,
                                           mid_channels=self.expansion*planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = LocalBinaryConv2d(planes, planes, sparsity, kernel_size=3, stride=1, padding=1, bias=False,
                                       mid_channels=self.expansion*planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GhostBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, ratio=4):
        super(GhostBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = GhostConv2d(in_planes, planes, ratio=ratio, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = GhostConv2d(planes, planes, ratio=ratio, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PerturbationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, noise_level=0.1, noisy_train=False, noisy_eval=False):
        super(PerturbationBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = PerturbativeConv2d(
                in_planes, planes, noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = PerturbativeConv2d(
            planes, planes, noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ShiftBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ShiftBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = ShiftConv2d(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = ShiftConv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, features=False, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.features = features

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.features and self.training:
            feats = []
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class MonomialResNet(nn.Module):
    def __init__(self,
                 block: MonomialBasicBlock,
                 num_blocks, num_terms, exp_range, exp_factor, mono_bias,
                 onebyone, in_channels=3, num_classes=10, features=False):
        super(MonomialResNet, self).__init__()
        self.in_planes = 64
        self.features = features

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, num_terms, exp_range, exp_factor[0], mono_bias,
                                       onebyone)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, num_terms, exp_range, exp_factor[1], mono_bias,
                                       onebyone)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, num_terms, exp_range, exp_factor[2], mono_bias,
                                       onebyone)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, num_terms, exp_range, exp_factor[3], mono_bias,
                                       onebyone)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self,
                    block: MonomialBasicBlock,
                    planes, num_blocks, stride, num_terms, exp_range, exp_factor, mono_bias, onebyone):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, num_terms, exp_range, exp_factor, mono_bias, onebyone))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.features and self.training:
            feats = []
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class RBFFamilyResNet(nn.Module):
    def __init__(self, block: RBFFamilyBasicBlock,
                 num_blocks, eps_range, exp_factor, rbf,
                 onebyone=True, in_channels=3, num_classes=10):
        super(RBFFamilyResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, eps_range, exp_factor[0], rbf,
                                       onebyone)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, eps_range, exp_factor[1], rbf,
                                       onebyone)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, eps_range, exp_factor[2], rbf,
                                       onebyone)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, eps_range, exp_factor[3], rbf,
                                       onebyone)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, eps_range, exp_factor, rbf,
                    onebyone):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, eps_range, exp_factor, rbf, onebyone))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class LocalBinaryResNet(nn.Module):
    def __init__(self, block: LocalBinaryBasicBlock,
                 num_blocks, in_channels=3, num_classes=10, sparsity=0.1, features=True):
        super(LocalBinaryResNet, self).__init__()
        self.in_planes = 64
        self.features = features

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, sparsity)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, sparsity)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, sparsity)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, sparsity)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, sparsity):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sparsity=sparsity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.features and self.training:
            feats = []
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class PerturbationResNet(nn.Module):
    def __init__(self, block: PerturbationBasicBlock,
                 num_blocks, noise_level, noisy_train, noisy_eval, in_channels=3, num_classes=10):
        super(PerturbationResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, noise_level, noisy_train, noisy_eval)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, noise_level, noisy_train, noisy_eval)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, noise_level, noisy_train, noisy_eval)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, noise_level, noisy_train, noisy_eval)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block: PerturbationBasicBlock,
                    planes, num_blocks, stride, noise_level, noisy_train, noisy_eval):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                noise_level=noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ShiftResNet(nn.Module):
    def __init__(self, block: ShiftBasicBlock,
                 num_blocks, in_channels=3, num_classes=10, features=False):
        super(ShiftResNet, self).__init__()
        self.in_planes = 64
        self.features = features

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block: ShiftBasicBlock,
                    planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.features and self.training:
            feats = []
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class GhostResNet(nn.Module):
    def __init__(self, block: GhostBasicBlock,
                 num_blocks, in_channels=3, num_classes=10, ratio=4):
        super(GhostResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, ratio)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, ratio)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, ratio)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, ratio)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ratio):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ratio=ratio))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# Basic Residual Block based models
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def monomial_resnet18(**kwargs):
    return MonomialResNet(MonomialBasicBlock, [2, 2, 2, 2], **kwargs)


def gaussian_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='gaussian', **kwargs)


def multiquadric_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='multiquadric', **kwargs)


def inverse_quadratic_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='inverse_quadratic', **kwargs)


def inverse_multiquadric_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='inverse_multiquadric', **kwargs)


def local_binary_resnet18(**kwargs):
    return LocalBinaryResNet(LocalBinaryBasicBlock, [2, 2, 2, 2], **kwargs)


def perturbative_resnet18(**kwargs):
    return PerturbationResNet(PerturbationBasicBlock, [2, 2, 2, 2], **kwargs)


def shift_resnet18(**kwargs):
    return ShiftResNet(ShiftBasicBlock, [2, 2, 2, 2], **kwargs)


def ghost_resnet18(**kwargs):
    return GhostResNet(GhostBasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def monomial_resnet34(**kwargs):
    return MonomialResNet(MonomialBasicBlock, [3, 4, 6, 3], **kwargs)


def local_binary_resnet34(**kwargs):
    return LocalBinaryResNet(LocalBinaryBasicBlock, [3, 4, 6, 3], **kwargs)


def perturbative_resnet34(**kwargs):
    return PerturbationResNet(PerturbationBasicBlock, [3, 4, 6, 3], **kwargs)


def shift_resnet34(**kwargs):
    return ShiftResNet(ShiftBasicBlock, [3, 4, 6, 3], **kwargs)


def ghost_resnet34(**kwargs):
    return GhostResNet(GhostBasicBlock, [3, 4, 6, 3], **kwargs)


def test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings("ignore")

    in_channels = 3  # for MNIST, 3 for SVHN and CIFAR-10
    H, W = 32, 32  # for MNIST, 32, 32 for SVHN and CIFAR-10

    model = resnet18(in_channels=in_channels, num_classes=10)

    # MonoCNNs
    # model = monomial_resnet18(
    #     in_channels=in_channels, num_terms=1, exp_range=(1, 7), exp_factor=[64, 128, 256, 512], mono_bias=False,
    #     onebyone=False)
    # model = monomial_resnet34(
    #     in_channels=in_channels, num_terms=1, exp_range=(1, 7), exp_factor=[64, 128, 256, 512], mono_bias=True,
    #     onebyone=False)

    # RBF family model
    # model = gaussian_resnet18(in_channels=in_channels, eps_range=(1, 7), exp_factor=[64, 128, 256, 512],
    #                           onebyone=False)
    # model = multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                               onebyone=True)
    # model = inverse_quadratic_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                    onebyone=False)
    # model = inverse_multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                       onebyone=True)

    # LBCNN
    # model = local_binary_resnet18(in_channels=in_channels, sparsity=0.1)

    # PNN
    # model = perturbative_resnet18(in_channels=in_channels, noise_level=0.1, noisy_train=False, noisy_eval=False)

    # ShiftNet
    # model = shift_resnet18(in_channels=in_channels)

    # GhostNet
    # model = ghost_resnet18(in_channels=in_channels, ratio=2, num_classes=10)

    print(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_count_full = sum(p.numel() for p in model.parameters())

    print(param_count / 1e6)
    print(param_count_full / 1e6)

    data = torch.rand(1, in_channels, H, W)
    y = model(data)
    try:
        for v in y[1]:
            print(v.size())
    except:
        print(y.size())

    flops = profile_macs(model, data) / 1e6
    print(flops)


if __name__ == '__main__':
    test()
