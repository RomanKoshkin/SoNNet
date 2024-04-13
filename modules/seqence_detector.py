from tkinter import Spinbox
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from .constants import bar_format


class DetectKnown(nn.Module):

    def __init__(self, KER_WIDTHS=[50, 100, 200], sigma=5.0, device='cuda:0', reverse=False):
        super(DetectKnown, self).__init__()
        self.device = device
        self.reverse = reverse
        self.KER = []
        self.KER_ = []
        self.NE = 400
        self.nass = 20
        self.ca_size = self.NE // self.nass
        self.sigma = torch.FloatTensor([sigma]).to(device)
        for ker_width in KER_WIDTHS:
            self.KER.append(self._make_kernel(ker_width).to(device))

        # kernels for estimating the number of spikes in the window
        for ker in self.KER:
            ker_ = torch.ones_like(ker)
            # ker_ = ker_ / ker_.sum()
            self.KER_.append(ker_)

    @staticmethod
    def _gauss(x_values, mu, sigma):
        return torch.exp(-torch.pow(x_values - mu, 2.) / (2 * torch.pow(sigma, 2.)))

    def _make_kernel(self, ker_width):
        shift = torch.FloatTensor([ker_width // self.ca_size]).to(device=self.device)
        ker = torch.zeros((self.NE, ker_width))
        x_values = torch.arange(ker_width).to(device=self.device, dtype=torch.float)

        for j, i in enumerate(np.arange(0, self.NE, self.nass)):
            if self.reverse:
                mu = shift * (self.nass - j)
            else:
                mu = shift * j
            g = self._gauss(x_values, mu, self.sigma)
            ker[i:i + self.ca_size, :] = g
        return ker.view(1, 1, self.NE, -1)

    def _make_rand_kernel(self, ker_width):
        shift = torch.FloatTensor([ker_width // self.ca_size]).to(device=self.device)
        ker = torch.zeros((self.NE, ker_width))
        x_values = torch.arange(ker_width).to(device=self.device, dtype=torch.float)

        means = np.arange(0, self.NE, self.nass)
        np.random.shuffle(means)

        for j, i in enumerate(means):
            if self.reverse:
                mu = shift * (self.nass - j)
            else:
                mu = shift * j
            g = self._gauss(x_values, mu, self.sigma)
            ker[i:i + self.ca_size, :] = g
        return ker.view(1, 1, self.NE, -1)

    def boostrapCIs(self, X, num_perms, ker_width):
        with torch.no_grad():
            STD = torch.zeros(size=(num_perms,))
            for i in range(num_perms):
                ker = self._make_rand_kernel(ker_width).to(self.device)
                conv = F.conv2d(X, weight=ker, padding=(0, ker.shape[-1] // 2), bias=None).squeeze()
                STD[i] = conv.std()
        return STD

    def forward(self, X):
        OUT = []
        SP_COUNTS = []
        for ker, ker_ in zip(self.KER, self.KER_):
            conv = F.conv2d(X, weight=ker, padding=(0, ker.shape[-1] // 2), bias=None)[:, :, :, :-1]
            sp_count = F.conv2d(X, weight=ker_, padding=(0, ker.shape[-1] // 2), bias=None)[:, :, :, :-1]
            sp_count[sp_count == 0.0] = 1.0  # if you have zero spikes, assume 1
            # OUT.append(conv / sp_count)  # NOTE: adjust covolution curve height for the number of spike in the window
            OUT.append(conv)  # NOTE: adjust covolution curve height for the number of spike in the window
            SP_COUNTS.append(sp_count)
        return torch.stack(OUT), torch.stack(SP_COUNTS)