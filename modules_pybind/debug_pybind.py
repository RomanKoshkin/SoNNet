# NOTE: make sure to use `engine='python'` if you use `pandas.read_csv`
#
import os, sys, warnings
from pprint import pprint

sys.path.append('../')
print(f'PID: {os.getpid()}')

from modules_pybind.python_wrapper import Model  # here will wrap the c++ object with Python

from modules.utils import *
from modules.constants import ROOT

import numpy as np  # import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------

cell_id = 0
HAGA = True
astro = True
symmetric = True
t_train = 0
t_sham = 100000

# --------------------------------------------------------------------------------------------------

# params = load_config(f'{ROOT}/configs/config_istdp.yaml')
params = load_config(f'{ROOT}/configs/config_istdp_lif.yaml')

dt = params['dt']
NEo = params['NEo']
NM = params['NM']

NE = params['NE']
NI = params['NI']

case = params['case']
wide = params['wide']

UU = simple_make_UU(NE, case, wide)

HAGA = True
symmetric = True
astro = False

for k, v in {
        'HAGA': HAGA,
        'symmetric': symmetric,
        'U': np.mean(UU),
        'Cp': 0.14,
        'Cd': 0.01,
        'taustf': 350.0,
        'taustd': 250.0,
        'tpp': 15.1549,  #5.1549, #
        'tpd': 120.4221  #5.4221, #
}.items():
    params[k] = v

m = Model(NE, NI, NEo, cell_id)
m.setParams(params)
m.saveSpikes(True)

if astro:
    m.setUU(UU)

m.set_useThetas(False)
m.get_useThetas()

m.m.sim_lif(2200)  # to debug LIF

patternLenMs = 20
nass = 20

# NOTE: change E->E weights
w = m.getWeights()
for i in range(NE):
    for j in range(NE):
        if w[i, j] > 0.001:
            w[i, j] = 0.15 * (4.0 + 0.8 * np.random.randn())
m.setWeights(w)

m.set_Jmax(9.0)  # NOTE: prohibit clipping

m.set_mex(0.3)

# NOTE: frezing all wts except EE
frozens = m.getFrozens()
frozens[:, NE:] = True
frozens[NE:, :] = True
m.setFrozens(frozens)

stimulator = Stimulator(
    m,
    stim_strength=1.0,
    nass=nass,
    rotate_every_ms=patternLenMs,
    cell_id=cell_id,
    dump_stats=1000,
)

# NOTE: clusterize_=True saves .npy snapshots. Load them with SnapshotLoader
stimulator.sham(250000, plasticity=True, clusterize_=True)
