import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyq

# Thank's Atila!
# https://gist.github.com/atilaorh/97c16b796c1d03138ef72bb80d9b97d7
class SobelGradRepeated2D(nn.Module):
    def __init__(self, in_channels, in_time):
        super(SobelGradRepeated2D, self).__init__()
        assert in_time % 2 != 0, 'in_time should be an odd number to maintain input and output sizes'

        gx_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        gy_kernel = gx_kernel.T.copy()

        gx_kernel = np.tile(gx_kernel, [1, in_channels, in_time, 1, 1])  # (out_channels, in_channels, kT, kH, kW)
        gy_kernel = np.tile(gy_kernel, [1, in_channels, in_time, 1, 1])

        gx_kernel = Variable(torch.from_numpy(gx_kernel), requires_grad=False)
        gy_kernel = Variable(torch.from_numpy(gy_kernel), requires_grad=False)

        self.register_buffer('gx_kernel', gx_kernel)
        self.register_buffer('gy_kernel', gy_kernel)

        self.in_time = in_time
        self.in_channels = in_channels


    def forward(self, im):
        gx = torch.nn.functional.conv3d(im, self.gx_kernel, bias=None, stride=1, padding=(self.in_time // 2, 1, 1))
        gy = torch.nn.functional.conv3d(im, self.gy_kernel, bias=None, stride=1, padding=(self.in_time // 2, 1, 1))

        grad = torch.sqrt(gx * gx + gy * gy)
        grad /= 3.

        return grad

# 3D-Unet code from: https://github.com/shiba24/3d-unet/blob/master/pytorch/model.py
class Net(nn.Module):
    def __init__(self, train_shapes, validation_shapes, test_shapes, args):
        super(Net, self).__init__()

        channels, sequence_length, height, width = train_shapes[0]
        sequence_length, state_dim = train_shapes[1]

        test_input = Variable(torch.ones(1, *(train_shapes[0])))
        sensor_dim = 512

        self.sobel_gradient = SobelGradRepeated2D(channels, sequence_length)

        hsl = sequence_length // 2  # half sequence length

        self.ec0 = self.encoder(       1,  32, kernel_size=(sequence_length, 3, 3), padding=(hsl, 0, 0), bias=True, batchnorm=False)
        self.ec1 = self.encoder(      32,  64, kernel_size=(sequence_length, 3, 3), padding=(hsl, 0, 0), bias=True, batchnorm=False)
        self.ec2 = self.encoder(      64,  64, kernel_size=(sequence_length, 3, 3), padding=(hsl, 0, 0), bias=True, batchnorm=False)
        self.ec3 = self.encoder(      64, 128, kernel_size=(sequence_length, 3, 3), padding=(hsl, 0, 0), bias=True, batchnorm=False)
        self.ec4 = self.encoder(     128, 128, kernel_size=(sequence_length, 3, 3), padding=(0, 0, 0),   bias=True, batchnorm=False)
        self.ec5 = self.encoder(     128, 256,               kernel_size=(1, 3, 3), padding=(0, 0, 0),   bias=True, batchnorm=False)
        self.ec6 = self.encoder(     256, 256,               kernel_size=(1, 3, 3), padding=(0, 0, 0),   bias=True, batchnorm=False)
        self.ec7 = self.encoder(     256, sensor_dim,        kernel_size=(1, 3, 3), padding=(0, 0, 0),   bias=True, batchnorm=False)

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
            nn.Linear(1024, 3))

        # b-vae

        self.gravity = Variable(torch.from_numpy(np.array([[0, 0, 1.]], dtype=np.float32))).cuda()
        self.quaterion_unit_inverse_mask = Variable(torch.from_numpy(np.array([1., -1., -1., -1.], dtype=np.float32)), requires_grad=False).cuda()

        self.epsilon = 1e-4

        # symmetric and positive definite, possibly just diagonal
        self.Q = nn.Parameter(torch.ones(9))

        self.reconstruction_loss = nn.L1Loss()

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.children():
            if type(module) == nn.Conv3d:
                torch.nn.init.xavier_normal(module.weight, gain=1)

    def quaterion_unit_inverse(self, q):
        return q * self.quaterion_unit_inverse_mask

    def quaterion_mul(self, a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                          aw * bx + ax * bw - ay * bz + az * by,
                          aw * by + ax * bz + ay * bw - az * bx,
                          aw * bz - ax * by + ay * bx + az * bw], 1)

    def quaternion_rot(self, vector, q):
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx, vy, vz = vector[:, 0], vector[:, 1], vector[:, 2]

        return torch.stack(
            [(1. - 2. * qy * qy - 2. * qz * qz) * vx + (2. * (qx * qy + qw * qz)) * vy + 2. * (qx * qz - qw * qy) * vz,
             (2. * (qx * qy - qw * qz)) * vx + (1. - 2. * qx * qx - 2. * qz * qz) * vy + 2. * (qy * qz + qw * qx) * vz,
             (2. * (qx * qz + qw * qy)) * vx + 2. * (qy * qz - qw * qx) * vy + (1. - 2. * qx * qx - 2. * qy * qy) * vz], 1)

    def expq(self, three_vector):
        # clamping to address https://github.com/pytorch/pytorch/issues/2421
        three_vector = torch.clamp(three_vector, self.epsilon)

        n = torch.norm(three_vector, p=2, dim=1, keepdim=True)
        half_norm = n / 2.0
        q = torch.cat([torch.cos(half_norm), torch.sin(half_norm) * three_vector / n], dim=1)

        # when norm(three_vector) is close to zero it creates a numerical instability,
        # so we use a different approximation (5.39)
        # https://www.research-collection.ethz.ch/handle/20.500.11850/129873
        mask = n <= self.epsilon
        if torch.sum(mask).data[0] > 0:
            q[mask] = torch.cat([n * 0.0 + 1, three_vector / 2], dim=1)

        return q

    def logq(self, q):
        # clamping to address https://github.com/pytorch/pytorch/issues/2421
        q = torch.clamp(q, self.epsilon)

        n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
        three_vector = (2 * torch.atan2(n, q[:, 0:1])) * q[:, 1:] / n

        # when norm(q) is close to zero it's numerically unstable, so we use 5.41 from
        # https://www.research-collection.ethz.ch/handle/20.500.11850/129873
        mask = n <= self.epsilon
        if torch.sum(mask).data[0] > 0:
            three_vector[mask] = torch.sign(q[:, 0:1]) * q[:, 1:]

        return three_vector

    def box_minus(self, a, b):
        return self.logq(self.quaterion_mul(a, self.quaterion_unit_inverse(b)))

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
        x = x.float() / 255.0
        x = self.sobel_gradient(x)

        return x

    def forward(self, x):
        dts, frames, xd, xdd, omegas, previous_state = x

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
        rot = self.expq(rot)

        # Extended Kalman Filter
        pos_dot = self.quaternion_rot(delta_pos, self.quaterion_unit_inverse(rot))
        vel_dot = self.quaternion_rot(self.gravity, self.quaterion_unit_inverse(rot)) + xdd[:, -1]

        previous_pos, previous_vel, previous_rot = previous_state[:, :3], previous_state[:, 3:6], previous_state[:, 6:]

        dt = dts[:, -1:]
        pos_t = previous_pos + pos_dot * dt
        vel_t = previous_vel + vel_dot * dt
        quat_t = self.quaterion_mul(rot, self.expq(omegas[:, -1] * dt))  #TODO: rot boxplus omega * dt

        estimate = (pos_t, vel_t, quat_t)

        A = Variable(torch.eye(9), requires_grad=False).cuda().unsqueeze(0).repeat(batch, 1, 1)  # function of x
        P = Variable(torch.eye(9), requires_grad=False).cuda().unsqueeze(0).repeat(batch, 1, 1)

        P = A.bmm(P).bmm(A.transpose(1, 2)) + self.Q.diag()  # page 42.3

        return (estimate, P)

    def loss(self, output, target, input=None):
        t_pos, t_vel, t_rot = target[:, -1, :3], target[:, -1, 3:6], target[:, -1, 6:]
        (pos, vel, rot), P = output

        x_delta = torch.cat([pos - t_pos, vel - t_vel, self.box_minus(rot, t_rot)], dim=1).unsqueeze(2)

        likelihood = x_delta.transpose(1, 2).bmm(P).bmm(x_delta)

        # TODO: this math is wrong until i can get the right formula for likelihood
        loss = likelihood.abs().mean() + self.Q.abs().sum()

        return loss, {'loss': loss}
