# -*- coding: utf-8 -*-
"""
@Project:
@File: sdf2d.py

@Description:

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = "François Lauze, University of Copenhagen"
__date__ = "7/6/23"
__version__ = "0.0.1"

from matplotlib import pyplot as plt

import utils
from grids import Grid2D


class SDF2(nn.Module):
    """
    A simple SDF network in dimension 2.
    To be close to the usage, that I have seen so far,                                                                                                                    this is a MLP with a skip connection.
    """
    def __init__(self,
                 num_entries=2,
                 hidden_size=256,
                 num_hidden_layers=4,
                 skip_connection=(3,),
                 weight_norm=True,
                 bias = 0.5
                 ) -> None:
        """
        
        :param hidden_size: siz oi the hidden layers
        :param num_hidden_layers: amount of hidden layers
        :param skip_connection: list of layers where the skip connection is applied
        :param weight_norm: if True, weight normalization is applied
        :param bias: bias of the last layer
        """
        super(SDF2, self).__init__()
        self.skip_connection = skip_connection
        #self.activation = nn.Softplus(beta=100)
        self.activation = nn.Tanh()
        dims = [num_entries] + [hidden_size] * num_hidden_layers + [1]
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1] - dims[0] if l + 1 in self.skip_connection else dims[l + 1]
            
            lin = nn.Linear(dims[l], out_dim)
            if l == self.num_layers - 2:
                nn.init.normal_(lin.weight, mean=np.sqrt(np.pi / dims[l]), std=0.0001)
                nn.init.constant_(lin.bias, bias)
                
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(l), lin)
            
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        return value of the function
        :param points: input tensor, size (n, 2) where n is the amount of sampled points
        :return: output tensor
        """
        x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
    
            if l in self.skip_connection:
                x = torch.cat([x, points], 1) / np.sqrt(2)
    
            x = lin(x)
    
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x
    
    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        SDF gradient
        :param points: points tensor, size (n, 2)
        :return: gradient tensor, size (n, 2)
        """
        points.requires_grad_(True)
        y = self.forward(points)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        grad = torch.autograd.grad(outputs=y, inputs=points, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return grad.unsqueeze(1)



class SdfDrawer(utils.Drawer):

    def draw(self, points):
        image = points.reshape(self.grid.m,  self.grid.n)
        self.ax.clear()
        self.ax.imshow(image, extent=self.grid.extent, origin='lower')
        self.ax.contour(self.grid.X, self.grid.Y, image, levels=[-0.2, 0, 0.2], colors='r')
        self.ax.set_title(self.title)
        plt.draw()
        plt.pause(0.001)

    def draw_sdf(self, sdf: SDF2, device: torch.device):
        points = encoding(self.grid.tensor(device).float())
        output = sdf(points)
        self.draw(output.detach().cpu().numpy())


def encoding(positions, num_frequencies=4):
    frequencies = 2.0 ** torch.arange(0, num_frequencies, device=positions.device)
    encodings = [positions]
    for freq in frequencies:
        encodings.append(torch.sin(freq * positions))
        encodings.append(torch.cos(freq * positions))
    return torch.cat(encodings, dim=-1)


def eikonal_loss(points: torch.Tensor, sdf: SDF2) -> torch.Tensor:
    """
    The eikonal loss is the square of the gradient of the sdf
    :param sdf: the sdf
    :param points: the points
    :return: the eikonal loss
    """
    # compute the gradient of the sdf
    sdf_grad = sdf.gradient(points)
    # compute the norm of the gradient
    sdf_grad_norm = torch.linalg.norm(sdf_grad, ord=2, dim=1)
    # compute the eikonal loss
    return torch.mean((sdf_grad_norm - 1.0)**2)


def train_sdf(grid: Grid2D, true_sdf, net: SDF2 = None, epochs=100, weight=1, rate=1e-3, drawer: SdfDrawer = None) \
        -> SDF2:
    device = true_sdf.device
    points = encoding(grid.tensor(device).float())
    if net is None:
        net = SDF2(num_entries=points.shape[1])
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    loss_value = nn.L1Loss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = net(points)
        loss_v = loss_value(output, true_sdf)
        loss_e = eikonal_loss(points, net)
        e_w = 0.0 if epoch < epochs / 2 else weight * epoch / epochs
        loss = loss_v + e_w * loss_e
        loss.backward()
        optimizer.step()
        if drawer is not None:
            drawer.set_title(str(epoch))
            drawer.draw(output.detach().cpu().numpy())
    return net

