import argparse
import torch
from torchvision import datasets, transforms
import models
import torch.optim as optim
import numpy
from tensorboard import SummaryWriter
from collections import defaultdict
import os

def to_numeric(value):
    if type(value) == torch.autograd.variable.Variable:
        value = value.data[0]
    elif type(value) == int or type(value) == float:
        value = value
    else:
        raise Exception('argument must be of type Variable, float, or int, not "{}"'.format(type(value)))
    return value

class Logger():
    def __init__(self, root):
        self.writer = SummaryWriter(root)
        self.last_indexes = defaultdict(int)

    def scalar(self, key, value, index=None):
        index = index if index is not None else self.last_indexes[key]
        self.last_indexes[key] += 1

        value = to_numeric(value)

        self.writer.add_scalar(key, value, index)

    def from_stats(self, key_value_dictionary, index=None):
        for key in key_value_dictionary:
            self.scalar(key, key_value_dictionary[key], index)

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=12, metavar='NW',
                        help='number of threads (default: 12)')
    parser.add_argument('--log-root', type=str, default='.log', metavar='R',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    args.train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    args.validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    args.test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    args.model = models.Net()
    if args.cuda:
        args.model.cuda()

    args.optimizer = optim.SGD(args.model.parameters(), lr=args.lr, momentum=args.momentum)

    args.train_logger = Logger(os.path.join(args.log_root, 'train'))
    args.test_logger = Logger(os.path.join(args.log_root, 'test'))

    return args

def add_stats(dictionaryA, dictionaryB):
    return {key:   (to_numeric(dictionaryA[key]) if key in dictionaryA else 0)
                 + (to_numeric(dictionaryB[key]) if key in dictionaryB else 0)
            for key in list(dictionaryA.keys()) + list(dictionaryB.keys())}

def divide_stats_by_constant(dictionary, constant):
    return {key: to_numeric(dictionary[key]) / constant for key in dictionary}


def stats_to_string(dictionary, prepend=""):
    return " ".join(["{}{}:{:.4f}".format(prepend, key, to_numeric(dictionary[key])) for key in dictionary])