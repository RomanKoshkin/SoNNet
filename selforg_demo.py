import os, sys
sys.path.append('../')
print(f'PID: {os.getpid()}')
from modules.utils import plotSTDP


from modules_pybind.python_wrapper import Model # here will wrap the c++ object with Python
from modules_pybind.cpp_modules import truncated_normal_ab_sample, generateRandom32bitInt # directly import a c++ function

from modules.utils import *
from modules.constants import ROOT
# from modules.environment import InfinitePong
# from modules.trainer import Trainer
# from modules.exp1 import Exp1
# from modules.VAEmodules import GaussianReconst, Dataset, VAE, SpikeVAE
# from modules.seqence_detector import DetectKnown

import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler

from itertools import permutations, groupby, chain
from collections import Counter
import seaborn as sns



import sys, time, threading, json
import os, subprocess, pickle, shutil, itertools, string, warnings
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import jensenshannon as JS
from multiprocessing import Pool
import scipy
from scipy import signal
import numba
from numba import njit, jit, prange
from numba.typed import List

import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from termcolor import cprint
from pprint import pprint

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

from tqdm import trange, tqdm
from itertools import repeat

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# from IPython.display import Video
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import umap



cell_id = 0

lif = False

params = load_config(f'{ROOT}/configs/config_sequential.yaml')
matplotlib.rcParams.update(load_config(f'{ROOT}/configs/rcparams.yaml'))
datafolder = 'data'

dt = params['dt']
NEo = params['NEo']
NM = params['NM']


NE = 400
NI = 80

case = params['case']
wide = params['wide']
N = NE + NI


symmetric = True
HAGA = True
astro = True
stp_on_I = False
dump_xy = False
dump_dw = False
RESET_CONNECTION_PROBS = True # NOTE: thie is important, otherwise cEE, cEI, cIE, cII will remain at defaults


UU = simple_make_UU(NE, case, wide)
for k,v in {'HAGA': HAGA,
            'symmetric': symmetric,
            'U': np.mean(UU),
            'cEE': 0.4,
            'cEI': 0.5,
            'cIE': 0.2,
            'cII': 0.5,   
            'Cp': 0.14,
            'Cd': 0.02,
            'taustf': 350.0,
            'taustd': 250.0,
            'tpp': 15.1549,
            'tpd': 120.4221,
           }.items():
    params[k] = v

# create the model
m = Model(NE, NI, NEo, cell_id)
m.m.datafolder = datafolder


m.setParams(params, reset_connection_probs=RESET_CONNECTION_PROBS)
m.saveSpikes(True)


m.m.dump_xy = dump_xy # save input patterns and outputs of each neuron at each time step
m.m.stp_on_I = stp_on_I
m.m.dump_dw = dump_dw


UU = np.random.choice(UU, size=N)
if astro:
    UU[NE:] = 1.0
else:
    UU[:NE] = np.mean(UU) # <<<<<<<<<<
    UU[NE:] = 1.0
m.setUU(UU)


m.set_useThetas(False)
m.get_useThetas()
np.save('weights', np.copy(m.getWeights()))
print(m.getState().Jmax)

fig, ax = plt.subplots(1, 2, figsize=(10,4))
stdp_dict = plotSTDP(params, ax[0], disp=0)


# Freeze all but E-E weights


fig, ax = plt.subplots(1,2,figsize=(12,4))
frozens = m.getFrozens()
ax[0].imshow(frozens)

frozens[:, NE:] = True
frozens[NE:,:] = True

m.setFrozens(frozens)

frozens = m.getFrozens()
ax[1].imshow(frozens)






# m.set_Jmax(2.0)
lif = False


nass = 20
overlap_frac = 0

patternLenMs = 10
stim_strength = 1.0

stimulator = Stimulator(
    m, 
    stim_strength=0.01 if lif else stim_strength,
    nass=nass,
    overlap=int(NE / nass * overlap_frac),
    rotate_every_ms=patternLenMs,
    cell_id=cell_id, 
    dump_stats=1000,
    lif=lif)

callback = None 
STIM_ONSET = []


rwPatID = None
random_order_of_patterns = False
m.set_mex(0.3)    # 

# m.set_mex(0.17) # for (sym/HAGA), (asym/noHAGA) sequence learning and replay
# m.set_mex(0.0001 if lif else 0.2) # for Hiratani's binary SNN
# m.m.mex = 0.00001    # for LIF SNN (probability that a neuron is forced to fire due to external random input)
                    # NOTE: if h = 0.1 ms, then in 1s you have 10000 steps and with mex=0.0001 your external
                    # stimulation is @ 1 Hz

# for i in range(5):
#     STIM_ONSET.append(m.getState().t)
#     stimulator.ping(patternLenMs=200, pattern_id=2, msPerEpoch=2, plasticity=False)
#     stimulator.sham(1600, plasticity=False, clusterize_=True, callback=callback)

# stimulator.ping(patternLenMs=20, pattern_id=2, msPerEpoch=2, plasticity=False)

# stimulator.perturb_randM(1000, M=100, plasticity=True, clusterize_=True, callback=None, saveFD=False)
# stimulator.sham(1000, plasticity=False, clusterize_=True, callback=callback)

# stimulator.train(40000, patternLenMs=patternLenMs, clusterize_=True, callback=callback, rwPatID=rwPatID, random_order_of_patterns=random_order_of_patterns)
# stimulator.sham(20000, plasticity=False, clusterize_=True, callback=callback)
stimulator.sham(30000, plasticity=True, clusterize_=True, callback=callback)


# stimulator.perturb(1000, stID=0, enID=100, plasticity=True, clusterize_=False, callback=None)
# stimulator.sham(100000, plasticity=False, clusterize_=True, callback=callback)



span_ms = 2000 # plot last 2000 ms of spikes

# load the spikes and get the last span_ms of them
path = f'data/spike_times_{cell_id}' 
sp = pd.read_csv(path, delimiter=' ', header=None, engine='python')
sp.columns = ['spiketime', 'neuronid']
sp = sp[(sp.spiketime > sp.spiketime.max() - span_ms)]

# clusterize the neurons to expose the cell assemblies
w_, labels, counts, mod, newids  = clusterize(m.getWeights()[:NE, :NE])
map_dict = {j:i for i,j in enumerate(newids)}
spsrt = sp.copy()
spsrt.neuronid = spsrt.neuronid.map(map_dict)

# get the number of cell assemblies
numCA = len(np.unique(labels))

# plot cell assmebly #0
ca_id = 0

caXnids = np.where(labels[newids] == ca_id)[0] # get the CA ids of each neuron after sorting
st = caXnids.min() # delimit the cell assembly (get the st and end neuron index)
en = caXnids.max()

fig, ax = plt.subplots(2, 1, figsize=(18,8), sharex=True, dpi=300)
spsrt.plot(x='spiketime', y='neuronid', style='bo', markersize=0.4, ax=ax[0])
ax[0].axhspan(st, en, alpha=0.3)

# get only the neurons in cell assembly #0 and plot them
tmp = spsrt[(spsrt.neuronid >= st) & (spsrt.neuronid < en)]
tmp.plot(x='spiketime', y='neuronid', style='bo', ax=ax[1])
plt.savefig('assets/ca.png', dpi=300)
