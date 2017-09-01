import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import pyquaternion as pyq

# 3D-Unet code from: https://github.com/shiba24/3d-unet/blob/master/pytorch/model.py
class Net(nn.Module):
    def __init__(self, train_shapes, validation_shapes, test_shapes, args):
        super(Net, self).__init__()

        channels, sequence_length, height, width = train_shapes[0]
        sequence_length, state_dim = train_shapes[1]

        test_input = Variable(torch.ones(1, *(train_shapes[0])))
        sensor_dim = 512

        self.ec0 = self.encoder(channels,  32, kernel_size=(sequence_length, 3, 3), padding=(1, 0, 0), bias=False, batchnorm=True)
        self.ec1 = self.encoder(      32,  64, kernel_size=(sequence_length, 3, 3), padding=(1, 0, 0), bias=False, batchnorm=True)
        self.ec2 = self.encoder(      64,  64, kernel_size=(sequence_length, 3, 3), padding=(1, 0, 0), bias=False, batchnorm=True)
        self.ec3 = self.encoder(      64, 128, kernel_size=(sequence_length, 3, 3), padding=(1, 0, 0), bias=False, batchnorm=True)
        self.ec4 = self.encoder(     128, 128, kernel_size=(sequence_length, 3, 3), padding=(0, 0, 0), bias=False, batchnorm=True)
        self.ec5 = self.encoder(     128, 256,               kernel_size=(1, 3, 3), padding=(0, 0, 0), bias=False, batchnorm=True)
        self.ec6 = self.encoder(     256, 256,               kernel_size=(1, 3, 3), padding=(0, 0, 0), bias=False, batchnorm=True)
        self.ec7 = self.encoder(     256, sensor_dim,        kernel_size=(1, 3, 3), padding=(0, 0, 0), bias=False, batchnorm=True)

        self.pool0 = nn.MaxPool3d((1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.h_pos = nn.Sequential(
            nn.Linear(sensor_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3))

        self.h_vel = nn.Sequential(
            nn.Linear(sensor_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3))

        self.h_rot = nn.Sequential(
            nn.Linear(sensor_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4))

        # sobel texture gradient
        # b-vae

        self.quaterion_unit_inverse_mask = Variable(torch.from_numpy(np.array([1., -1., -1., -1.], dtype=np.float32)),
                                                    requires_grad=False)
        if args.cuda:
            self.quaterion_unit_inverse_mask = self.quaterion_unit_inverse_mask.cuda()

        self.reconstruction_loss = nn.MSELoss()

    def quaterion_unit_inverse(self, q):
        return q * self.quaterion_unit_inverse_mask

    def quaterion_mul(self, a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                          aw * bx + ax * bw - ay * bz + az * by,
                          aw * by + ax * bz + ay * bw - az * bx,
                          aw * bz - ax * by + ay * bx + az * bw], 1)

    def logq(self, q):
        n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
        return (2 * torch.atan2(n, q[:, 0].unsqueeze(1))) * q[:, 1:] / n

    def box_minus(self, a, b, safe=True):
        result = self.logq(self.quaterion_mul(a, self.quaterion_unit_inverse(b)))
        if safe:
            result[result != result] = 0

        return result

    def encoder(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def preprocess(self, x):
        return x.float() / 255.0

    def forward(self, x):
        frames, xd, xdd, omegas, previous_state = x

        # normalize
        x = self.preprocess(frames)

        # unet encoder with batch norm
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        # global average pooling
        batch, channel, sequence, height, width = e7.size()
        gap = e7.view(batch, channel, sequence * height * width).mean(2)

        # sensor-to-measuement transform
        delta_pos = self.h_pos(gap)
        delta_vel = self.h_vel(gap)
        rot = self.h_rot(gap)
        rot /= torch.norm(rot, p=2, dim=1, keepdim=True)

        previous_pos, previous_vel, previous_rot = previous_state[:, :3], previous_state[:, 3:6], previous_state[:, 6:]
        measurement = (previous_pos + delta_pos, previous_vel + delta_vel, rot)
        # TODO: do we want to predict the delta to the quaterion vector?

        estimate = measurement

        P = Variable(torch.eye(9)).cuda()

        return (estimate, P)

    def loss(self, output, target):
        t_pos, t_vel, t_rot = target[:, -1, :3], target[:, -1, 3:6], target[:, -1, 6:]
        (pos, vel, rot), P = output

        x_delta = torch.cat([pos - t_pos, vel - t_vel, self.box_minus(rot, t_rot)], dim=1).unsqueeze(2)
        likelihood = torch.matmul(x_delta.transpose(1, 2), P).bmm(x_delta)

        loss = torch.log(likelihood.mean())

        return loss, {'loss': loss}
