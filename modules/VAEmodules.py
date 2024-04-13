from asyncio import base_tasks
from scipy.sparse import lil_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from itertools import repeat
from einops import rearrange
from termcolor import cprint


class Dataset(torch.utils.data.Dataset):
    """
    ARGS:
        sp: dataframe of spike times.
        winlen: window length, in samples
        stride: window stride, in samples
    OUTPUT:
        dense array of float16 of the spikes
    """

    def __init__(self, sp, winlen, stride, dtype=torch.float32):
        self.dtype = dtype
        self.stride = stride
        self.winlen = winlen

        cprint('not using downsampling, just rounding to integer ms', color='red')
        self.T_samples = np.round(sp.spiketime.max()).astype(int) + 1
        sp_samples = np.round(sp.spiketime.to_numpy()).astype(int)
        self.lm = lil_matrix((400, self.T_samples + 1)).astype('uint8')
        for t, nid in tqdm(zip(sp_samples, sp.neuronid.to_numpy())):
            if nid < 400:
                self.lm[nid, t] = 1

        self.index = list(range(0, self.T_samples - self.winlen - self.stride, self.stride))

    def __getitem__(self, i):
        addr = self.index[i]
        return torch.from_numpy(self.lm[:, addr:addr + self.winlen].toarray()).to(self.dtype)

    def __len__(self):
        return len(self.index)


class Filter(nn.Module):

    def __init__(self, winlen, DEVICE):
        super(Filter, self).__init__()
        self.DEVICE = DEVICE
        self.winlen = winlen
        self.x_values = torch.linspace(0, winlen, winlen, device=DEVICE)
        self.mus = nn.Parameter(torch.randint(0, winlen, size=(400,), device=DEVICE, dtype=torch.float))
        self.sigma = nn.Parameter(torch.tensor([16.0], device=DEVICE), requires_grad=False)
        self._rebuild()

    def _rebuild(self):
        self.filt = self._gaussian()

    def _gaussian(self):
        return torch.exp(-torch.pow(self.x_values.repeat(400, 1) - self.mus.unsqueeze(-1), 2.) /
                         (2 * torch.pow(self.sigma, 2.)))

    def forward(self, X):
        return torch.einsum('bij,ij->b', X, self.filt) / self.winlen


class Filter_xp(nn.Module):

    def __init__(self, winlen, DEVICE):
        super(Filter_xp, self).__init__()
        self.DEVICE = DEVICE
        self.winlen = winlen
        self.x_values = torch.linspace(0, winlen, winlen, device=DEVICE)
        self.mus = nn.Parameter(torch.randint(0, winlen, size=(400,), device=DEVICE, dtype=torch.float))
        self.amps = nn.Parameter(torch.rand(size=(400,), device=DEVICE, dtype=torch.float))
        self.sigma = nn.Parameter(torch.tensor([16.0], device=DEVICE), requires_grad=False)
        self._rebuild()

    def _rebuild(self):
        self.filt = self._gaussian()

    def _gaussian(self):
        return torch.exp(-torch.pow(self.x_values.repeat(400, 1) - self.mus.unsqueeze(-1), 2.) /
                         (2 * torch.pow(self.sigma, 2.))) * self.amps.unsqueeze(-1)

    def forward(self, X):
        return torch.einsum('bij,ij->b', X, self.filt) / self.winlen


class GaussianReconst(nn.Module):

    def __init__(self, winlen, DEVICE, batch_size):
        super(GaussianReconst, self).__init__()
        self.DEVICE = DEVICE
        self.winlen = winlen
        self.batch_size = batch_size
        self.x_values = torch.linspace(0, winlen, winlen, device=DEVICE).repeat(batch_size, 1)
        self.sigma = torch.tensor([10.0], device=DEVICE).repeat(batch_size, 1)

    def _rebuild(self, mus):
        filt = torch.zeros(size=(self.batch_size, 400, self.winlen), device=self.DEVICE)
        mus = torch.clamp(mus.clone(), min=0.0, max=self.winlen - 1)
        return self._gaussian(filt, mus.unsqueeze(-1))

    def _gaussian(self, filt, mus):
        for i in range(400):
            filt[:, i, :] = torch.exp(-torch.pow(self.x_values - mus[:, i], 2.) / (2 * torch.pow(self.sigma, 2.)))
        return filt

    def forward(self, mus):
        return self._rebuild(mus)


class SpikeVAE(nn.Module):

    def __init__(self, winlen, num_filters, latent_dim, dtype=torch.float, DEVICE='cuda:0', batch_size=100):
        super(SpikeVAE, self).__init__()
        self.DEVICE = DEVICE
        self.winlen = winlen
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        # self.filters = nn.ModuleList([Filter(winlen, DEVICE) for i in range(self.num_filters)])
        self.filters = nn.ModuleList([Filter_xp(winlen, DEVICE) for i in range(self.num_filters)])
        print('USING Fileter_xp')

        self.int0_mu = nn.Linear(self.num_filters, self.latent_dim, dtype=dtype)
        self.int0_sig = nn.Linear(self.num_filters, self.latent_dim, dtype=dtype)

        self.lat_int1 = nn.Linear(self.latent_dim, 400, dtype=dtype)
        self.lat_int2 = nn.Linear(self.latent_dim, 400, dtype=dtype)
        self.int01 = nn.Linear(self.latent_dim, self.latent_dim, dtype=dtype)
        self.bn = nn.BatchNorm1d(400, affine=True)
        self.bn1 = nn.BatchNorm1d(400, affine=True)
        self.gr = GaussianReconst(winlen, DEVICE, batch_size)

        self.info()

    def forward(self, visual_input, direct_mu=None):
        # NOTE: you must re-allocate this tensor to avoid "trying to backward through the graph a second time..."
        self.filtered = torch.zeros(size=(self.batch_size, self.num_filters), device=self.DEVICE)
        for i, f in enumerate(self.filters):
            self.filtered[:, i] = f(visual_input)

        if direct_mu is None:
            # compute mu & sigma
            mu = F.elu(self.int0_mu(self.filtered))
            log_sig = F.elu(self.int0_sig(self.filtered))

            # get the stochastic latent representation
            latent = self.reparameterize(mu, log_sig)

            # project the latent
            laten2int_com = F.gelu(self.int01(latent))

            # predict the COMs for spikes in each row
            # coms = F.sigmoid(self.bn(self.lat_int1(laten2int_com))) * (self.winlen - 1)
            coms = F.gelu(self.lat_int1(laten2int_com))

            # predict the number of spikes for gaussians in each row
            # pred_num_spikes = F.sigmoid(self.bn1(self.lat_int2(laten2int_com)))
            pred_num_spikes = F.gelu(self.lat_int2(laten2int_com))

            # coms = self.gr(coms)
        else:
            # compute mu & sigma
            mu = F.elu(self.int0_mu(self.filtered))

            # project mu (insteaad of the stochastic latent)
            laten2int_com = F.gelu(self.int01(mu))

            # predict the COMs for spikes in each row
            # coms = F.sigmoid(self.bn(self.lat_int1(laten2int_com))) * (self.winlen - 1)
            coms = F.gelu(self.lat_int1(laten2int_com))

            # predict the number of spikes for gaussians in each row
            # pred_num_spikes = F.sigmoid(self.bn1(self.lat_int2(laten2int_com)))
            pred_num_spikes = F.gelu(self.lat_int2(laten2int_com))

            # decoded = self.gr(coms)
            log_sig = 0
        return coms, pred_num_spikes, mu, log_sig

    def reparameterize(self, mu, log_sig):
        std = torch.exp(log_sig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_gumbel(self, logits):
        return F.gumbel_softmax(logits, dim=-1, hard=True)

    def KL(self, z_mean, z_log_sigma):
        kl_loss = -0.5 * torch.mean(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
        return kl_loss

    def info(self):
        np = 0
        for p in self.parameters():
            np += torch.prod(torch.tensor(p.shape))
        print(f'Number of parameters: {np}')


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.im_int0 = nn.Linear(784, 256)
        self.int0_mu = nn.Linear(256, 5)
        self.int0_sig = nn.Linear(256, 5)
        self.lat_int1 = nn.Linear(5, 256)
        self.int1_im = nn.Linear(256, 784)
        self.info()

    def forward(self, visual_input, direct_mu=None):
        if direct_mu is None:
            int0 = F.relu(self.im_int0(visual_input.view(-1, 784)))
            mu = self.int0_mu(int0)
            log_sig = self.int0_sig(int0)
            latent = self.reparameterize(mu, log_sig)
            int1 = F.relu(self.lat_int1(latent))
            decoded = torch.sigmoid(self.int1_im(int1))
        else:
            int0 = F.relu(self.im_int0(visual_input.view(-1, 784)))
            mu = self.int0_mu(int0)
            int1 = F.relu(self.lat_int1(mu))
            decoded = torch.sigmoid(self.int1_im(int1))
            log_sig = 0
        return decoded, mu, log_sig

    def reparameterize(self, mu, log_sig):
        std = torch.exp(log_sig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def KL(self, z_mean, z_log_sigma):
        kl_loss = -0.5 * torch.sum(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
        return kl_loss

    def info(self):
        np = 0
        for p in self.parameters():
            np += torch.prod(torch.tensor(p.shape))
        print(f'Number of parameters: {np}')


class Xent(nn.Module):

    def __init__(self, DEVICE):
        super(Xent, self).__init__()
        x_values = torch.linspace(0, 200, 200, device=DEVICE)
        sigma = torch.tensor([10.0], device=DEVICE)
        # 99.5 is to make sure the peak of the kernel is in the middle of its support
        self.ker = torch.exp(-torch.pow(x_values - 99.5, 2.) / (2 * torch.pow(sigma, 2.))).repeat(400, 1, 1, 1)
        self.cr = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, prediction, X):
        # calculate pseudo-labels
        centerOfMass = F.conv2d(
            input=X.unsqueeze(2),
            weight=self.ker,
            groups=400,
            padding='same',
        ).squeeze().argmax(dim=-1)

        # remember the batch id and row where there's at least one spike
        NotToSkip = torch.where(X.max(axis=-1)[0] > 0)

        # only take into account the rows, for which there is at least one spike
        return self.cr(X[NotToSkip], centerOfMass[NotToSkip]), centerOfMass, NotToSkip