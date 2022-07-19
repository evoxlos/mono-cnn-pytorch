import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import numpy as np
from collections import OrderedDict

from datasets.cifar import CorruptedCIFAR
from datasets.cifar import AdditioanlCIFAR10
from utils import Normalize, cross_entropy_loss_with_soft_target


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--machine', type=str, default='rtx')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--data', default='../data', type=str, help='path to CIFAR-10/100 data')
# robustness evaluation related settings
parser.add_argument('--additional-data', default='../data', type=str, help='path to additional CIFAR-10 test data')
parser.add_argument('--corrupt-data', default='../data', type=str,
                    help='path to corrputed CIFAR-10-C / CIFAR-100-C test data')
parser.add_argument('--attacks', default='fgsm#pgd#square', type=str, help='name of the output dir to save')
parser.add_argument('--cifar100', action='store_true', default=False, help='use CIFAR-100 dataset')
parser.add_argument('--model', default='resnet18', type=str,
                    help='which model')
parser.add_argument('--variant', default=None, type=str, help='which variant')
# polynomial transformation CNN settings
parser.add_argument('--num-terms', default=5, type=int, help='number of poly terms')
parser.add_argument('--exp-range-lb', default=0, type=int, help='exponent min')
parser.add_argument('--exp-range-ub', default=10, type=int, help='exponent max')
parser.add_argument('--exp-factor', default=2, type=int, help='fan-out factor')
parser.add_argument('--mono-bias', action='store_true', default=False, help='add a bias term to poly trans')
parser.add_argument('--onebyone', action='store_true', default=False,
                    help='whether to use 1x1 conv to blend features')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='path to checkpoint')
# local binary CNN settings
parser.add_argument('--sparsity', default=0.1, type=float,
                    help='sparsity of local binary weights, higher number means more non-zero weights')
# pertubation CNN settings
parser.add_argument('--noise-level', default=0.0, type=float,
                    help='the severity of noise induced to features (PNN) and weights (PolyCNN)')
parser.add_argument('--noisy-train', action='store_true', default=False,
                    help='whether to use random noise during every training mini-batch')
parser.add_argument('--noisy-eval', action='store_true', default=False,
                    help='whether to use random noise during every evaluation mini-batch')
parser.add_argument('--output', default='tmp', type=str, help='name of the output dir to save')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
# knowledge distillation related
parser.add_argument('--kd', default=None, type=str,
                    help='path to the pretrained teacher model for knowledge distillation')
parser.add_argument('--kd-ratio', default=0.1, type=float, help='kd')
args = parser.parse_args()


args.output = './CIFAR10_Exps/{}'.format(args.output)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')


mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # omit the normalization to run Adversarial attack toolbox
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # omit the normalization to run Adversarial attack toolbox
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.cifar100:
    trainset = torchvision.datasets.CIFAR100(
        root=args.data, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=True, transform=transform_test)

    num_classes = 100

else:
    trainset = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test)

    num_classes = 10

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

corruption_types = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
    'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
    'speckle_noise', 'zoom_blur'
]


# Model
print('==> Building model..')
if args.model == 'resnet18':
    if args.variant == 'poly':
        from models import polynomial_resnet18
        exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
        net = polynomial_resnet18(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'polyv2':
        from models import polynomial_resnet18v2
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = polynomial_resnet18v2(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    elif args.variant == 'gaussian':
        from models import gaussian_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = gaussian_resnet18(num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
                                exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
                                noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'multiquadric':
        from models import multiquadric_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = multiquadric_resnet18(num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
                                    exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
                                    noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'inverse_quadratic':
        from models import inverse_quadratic_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = inverse_quadratic_resnet18(
            num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
            noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'inverse_multiquadric':
        from models import inverse_multiquadric_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = inverse_multiquadric_resnet18(
            num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
            noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'local_binary':
        from models import local_binary_resnet18
        net = local_binary_resnet18(num_classes=num_classes, sparsity=args.sparsity)

    elif args.variant == 'perturbative':
        from models import perturbative_resnet18
        net = perturbative_resnet18(num_classes=num_classes, noise_level=args.noise_level,
                                    noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    elif args.variant == 'shift':
        from models import shift_resnet18
        net = shift_resnet18(num_classes=num_classes)

    elif args.variant == 'ghost':
        from models import ghost_resnet18
        net = ghost_resnet18(num_classes=num_classes, ratio=4)

    else:
        from models import resnet18
        net = resnet18(num_classes=num_classes)

    if args.kd:
        from models import resnet18
        teacher = resnet18(num_classes=num_classes, features=True)
        ckpt = torch.load(args.kd, map_location='cpu')
        teacher = nn.Sequential(
            # include normalization as first layer
            Normalize(in_channels=3, mean=mean, std=std),
            teacher
        ).to(device)
        teacher.load_state_dict(ckpt['state_dict'])
        print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
        net.features = True

elif args.model == 'resnet34':
    if args.variant == 'poly':
        from models import polynomial_resnet34
        exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
        net = polynomial_resnet34(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'local_binary':
        from models import local_binary_resnet34
        net = local_binary_resnet34(num_classes=num_classes, sparsity=args.sparsity)

    elif args.variant == 'perturbative':
        from models import perturbative_resnet34
        net = perturbative_resnet34(num_classes=num_classes, noise_level=args.noise_level,
                                    noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    elif args.variant == 'shift':
        from models import shift_resnet34
        net = shift_resnet34(num_classes=num_classes)

    elif args.variant == 'ghost':
        from models import ghost_resnet34
        net = ghost_resnet34(num_classes=num_classes, ratio=4)

    else:
        from models import resnet34
        net = resnet34(num_classes=num_classes)

    if args.kd:
        from models import resnet34
        teacher = resnet34(num_classes=num_classes, features=True)
        ckpt = torch.load(args.kd, map_location='cpu')
        teacher = nn.Sequential(
            # include normalization as first layer
            Normalize(in_channels=3, mean=mean, std=std),
            teacher
        ).to(device)
        teacher.load_state_dict(ckpt['state_dict'])
        print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
        net.features = True

else:
    raise NotImplementedError

net = nn.Sequential(
    # include normalization as first layer
    Normalize(in_channels=3, mean=mean, std=std),
    net
).to(device)

checkpoint = torch.load(args.checkpoint, map_location='cpu')['state_dict']
net.load_state_dict(checkpoint)

print(net)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

parameters = filter(lambda p: p.requires_grad, net.parameters())

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

n_epoch = args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)


# Training
def train(epoch):
    print('\nEpoch: {:d}  lr: {:.4f}'.format(epoch, scheduler.get_last_lr()[0]))
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.kd:
            outputs, feats = net(inputs)
        else:
            outputs = net(inputs)

        if args.kd:
            with torch.no_grad():
                soft_logits, soft_feats = teacher(inputs)
                soft_logits = soft_logits.detach()
                soft_feats = [feat.detach() for feat in soft_feats]
                soft_label = F.softmax(soft_logits, dim=1)

            kd_loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
            loss = 0.4 * kd_loss + criterion(outputs, targets)

            for (_f, _sf) in zip(feats, soft_feats):
                loss += args.kd_ratio * F.mse_loss(_f, _sf)

        else:
            loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
                train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if torch.isnan(torch.tensor(loss.item())):
            return True

    return False


def test(net, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    correct_sample_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_sample_indices.extend(predicted.eq(targets).cpu().detach().numpy().tolist())
            correct += predicted.eq(targets).sum().item()

            if total >= 10000:
                break
    print('Test Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total, np.where(correct_sample_indices)[0]


save_data_dict = OrderedDict()
for epoch in range(start_epoch, start_epoch+n_epoch):
    terminate = train(epoch)
    acc, sampler_indices = test(net, testloader)

    corrupt_accs = []
    for c_type in corruption_types:
        print('corruption type: {}'.format(c_type))
        corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)
        corrupt_testloader = torch.utils.data.DataLoader(
            corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

        corrupt_acc, _ = test(net, corrupt_testloader)
        corrupt_accs.append(corrupt_acc)

    print(np.mean(corrupt_accs))

    scheduler.step()

    if terminate:
        break

save_data_dict['acc'] = acc
save_data_dict['state_dict'] = net.state_dict()

# evaluate for OOD Robustness
if not args.cifar100:
    print("\nEvaluating on additional CIFAR-10 testset")
    additional_testset = AdditioanlCIFAR10(root=args.additional_data, transform=transform_test)
    additional_testloader = torch.utils.data.DataLoader(
        additional_testset, batch_size=100, shuffle=False, num_workers=2)
    additional_acc, _ = test(net, additional_testloader)
    save_data_dict['additional_acc'] = additional_acc

print("\nEvaluating on corrupted CIFAR-10 testset")
for c_type in corruption_types:
    print('corruption type: {}'.format(c_type))
    corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)
    corrupt_testloader = torch.utils.data.DataLoader(
        corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

    corrupt_acc, _ = test(net, corrupt_testloader)
    save_data_dict['{}_acc'.format(c_type)] = corrupt_acc

