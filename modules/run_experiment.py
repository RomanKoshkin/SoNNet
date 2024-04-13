import wandb
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler

# For Arial
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams["text.usetex"] = False

# for latex:
# matplotlib.rcParams['font.family'] = "DejaVu Sans"
# matplotlib.rcParams['font.serif'] = "Computer Modern"
# matplotlib.rcParams["text.usetex"] = True

matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rcParams["axes.labelsize"] = 22
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["legend.fontsize"] = 22
matplotlib.rcParams["axes.titlesize"] = 22
matplotlib.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["figure.autolayout"] = True

from itertools import permutations, groupby, chain
from collections import Counter
import seaborn as sns
from sklearn.datasets import fetch_openml


import sys, time, threading, json
print(f'Python version: {sys.version_info[0]}')
import os, subprocess, pickle, shutil, itertools, string, warnings
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
import scipy
from scipy import signal
import numba
from numba import njit, jit, prange
from numba.typed import List

from cClasses_10b import cClassOne
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from termcolor import cprint
from pprint import pprint

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

from sknetwork.clustering import Louvain, modularity
from sknetwork.linalg import normalize
from sknetwork.utils import bipartite2undirected, membership_matrix

from tqdm import trange, tqdm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar




def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lo_cutoff, hi_cutoff, fs=7.5, order=5):
    b, a = butter_highpass(lo_cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    b, a = butter_lowpass(hi_cutoff, fs, order=order)
    y = signal.filtfilt(b, a, y)
    return y

def calcAssWgts(t):
    global m, labels
    res = dict()
    res['t'] = t
    for i in np.unique(labels):
        for j in np.unique(labels):
            res[f'{i}-{j}'] = m.calcMeanW(np.where(labels==i)[0].astype('int32'), np.where(labels==j)[0].astype('int32'))
    return res

def calcF(t):
    global m, labels
    F = dict()
    F['t'] = t
    for i in np.unique(labels):
        F[f'{i}'] = m.calcMeanF(np.where(labels==i)[0].astype('int32'))
    return F

def calcD(t):
    global m, labels
    D = dict()
    D['t'] = t
    for i in np.unique(labels):
        D[f'{i}'] = m.calcMeanD(np.where(labels==i)[0].astype('int32'))
    return D

def plot_STDP(m):
    # see STDP_explained.ipynb for details
    def fd(x, alpha, JEE):
        return np.log(1+x*alpha/JEE)/np.log(1+alpha)
    states = m.getState()
    current_spike_t = 0
    spts = np.linspace(-250, 0, 200)
    dwd = states.Cp * np.exp(-(current_spike_t - spts)/states.tpp ) - fd(0.1, states.alpha, states.JEE) * states.Cd * np.exp(-(current_spike_t - spts)/states.tpd)
    dwp = states.Cp * np.exp(-(current_spike_t - spts)/states.tpp ) - fd(0.1, states.alpha, states.JEE) * states.Cd * np.exp(-(current_spike_t - spts)/states.tpd)
    plt.plot(spts-current_spike_t, dwd, label='LTD')
    plt.plot(current_spike_t-spts, dwp, label='LTP')
    plt.grid()
    plt.xlabel('$|t^i - t^j|$ [ms]', fontsize=18)
    
@numba.jit(fastmath=True)
def getMeanAssWgt(t):
    global m, labels, params, WgtEvo, FDevo, NE, NI
    d = dict()
    fd = dict()
    d['t'] = t
    fd['t'] = t
    W = m.getWeights()[:NE,:NE]
    F = np.copy(m.getF())
    D = np.copy(m.getD())
    for assID in np.unique(labels):
        wacc = 0
        c = 0
        assNids = np.where(labels==assID)[0]
        fd['{}.f'.format(assID)] = F[assNids].mean()
        fd['{}.d'.format(assID)] = D[assNids].mean()
        for i in assNids:
            for j in assNids:
                if W[i, j] > params['Jepsilon']:
                    c += 1
                    wacc += W[i, j]
        wacc /= c
        d[assID] = wacc
    WgtEvo.append(d)
    FDevo.append(fd)
    
@numba.jit(fastmath=True)
def getMeanBetweenAssWgt(t, assAid, assBid):
    global m, labels, params, D, NE, NI
    d = dict()
    d['t'] = t
    W = m.getWeights()[:NE,:NE]
   
    waccAB = 0
    cAB = 0
    waccBA = 0
    cBA = 0
    assANids = np.where(labels==assAid)[0]
    assBNids = np.where(labels==assBid)[0]
    for i in assANids:
        for j in assBNids:
            if W[j, i] > params['Jepsilon']:
                cAB += 1
                waccAB += W[j, i]
            if W[i, j] > params['Jepsilon']:
                cBA += 1
                waccBA += W[i, j]
    waccAB /= cAB
    waccBA /= cBA
    d['AB'] = waccAB
    d['BA'] = waccBA
    D.append(d)

# iterate over each time bin in the dataframe
def par_loop(i):
    global tmp, step_ms
    sp_count = tmp[(tmp.spiketime >= i) & (tmp.spiketime < i + step_ms)].shape[0]
    bin_middle = i + step_ms/2
    return sp_count, bin_middle

def DetermineUpstateLenghs(cell_ass_id, sub_sorted):
    
    global tmp, step_ms
    
    ref_vline_spacing_ms = 50 # for plotting
    plotting = False
    interval_ms_ = 160000
    
    step_ms = 5 # width of a time bin in which we count spikes (high spike count indicates an upstate)
    trigger_level = 10**0.9 # fine tune this parameter to determine the upstates (see plot below)
    sp_count, bin_middle = [], [] # list of spike counts for each bin, middle times for each bin
    

    
    s_ = np.where(labels[newids] == cell_ass_id)[0].min()
    e_ = np.where(labels[newids] == cell_ass_id)[0].max()


    tmp = sub_sorted[(sub_sorted.neuronid > s_) & (sub_sorted.neuronid < e_)]
    tmp = tmp[tmp.spiketime < tmp.spiketime.min() + interval_ms_].reset_index(drop=True)
    tmp.neuronid -= tmp.neuronid.min()
    


    if plotting:
        fig, ax = plt.subplots(4, 1, figsize=(18,15), sharex=True)

        ax[0].plot(sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].spiketime,
                   sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].neuronid, 'bo', ms=0.2)
        ax[0].axhspan(s_, e_, alpha=0.3, color='red')
        ax[0].set_ylabel(f'Neuron IDs')

        ax[1].plot(tmp.spiketime, tmp.neuronid, 'bo', ms=0.2)
        ax[1].set_title(f'Cell assembly id: {cell_ass_id}')
        for i in np.arange(st, 
                           sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].spiketime.max(),
                           ref_vline_spacing_ms):
            ax[1].axvline(i, linewidth=0.5)
        ax[1].set_ylabel(f'Neuron IDs')



    pool_list = np.arange(tmp.spiketime.min(), tmp.spiketime.min() + interval_ms_ , step_ms)
    with Pool(25) as p:
        res = list(p.map(par_loop, pool_list)) # without a progress bar


    sp_count = np.array([i[0] for i in res])
    bin_middle = np.array([i[1] for i in res])



    if plotting:
        ax[2].plot(bin_middle, sp_count)
        ax[2].set_yscale('log')
        ax[2].grid(True, which="both")
        ax[2].axhline(trigger_level, color='red', linewidth=2, label='trigger level')
        ax[2].set_ylabel(f'Number of spikes in {step_ms}-ms windows')
        ax[2].legend()

    # determine the rectangular signal showing where upstates are
    up_state_test = np.zeros_like(sp_count)
    up_state_test[sp_count > trigger_level] = 1

    if plotting:
        ax[3].plot(bin_middle, up_state_test)
        ax[3].set_xlabel(f'Time, ms')
        ax[3].set_ylabel(f'Upstate (yes, no)')

        plt.figure()

    # split the rectangular signal of contiguous 1 and 0s into strings of contiguous ones and zeros.
    # count their lengths. Each 1 corresponds to one binwidth in which an upstate was detected
    idx_at_which_to_split = np.where((np.diff(up_state_test) == 1 ) | (np.diff(up_state_test) == -1 ))[0] + 1
    # compute the lengths of each upstate in milliseconds
    consec = np.split(up_state_test, idx_at_which_to_split) # indicators that indicate if the timebin is up or downstate
    upstate_lengts_ms = [i.sum()*step_ms for i in consec if i.sum() > 0]

    muFR, us_nogap = get_mean_FR_in_assembly(tmp)
    return np.mean(upstate_lengts_ms), e_-s_, muFR, cell_ass_id, s_, e_

def get_mean_FR_in_assembly(tmp):
    
    window = 10 # in ms
    FR_min = 20 # firing rate within a window, below which we drop the data as belonging to a down state
    mingap = 50  # minimum length of downstate that you want to delete
    standard_gap = 0.1 # the length of gap you want to truncate gaps larger than mingap (must be smaller than mingap)

    ss, df_, us_nogap = delimit_upstates(tmp, FR_min, window, mingap, standard_gap)

    FR = []
    for i in range(us_nogap.neuronid.max()):
        FR.append(us_nogap[us_nogap.neuronid == i].shape[0] / ((us_nogap.spiketime.max() - us_nogap.spiketime.min()) / 1000))
    return np.mean(FR), us_nogap

def delimit_upstates(tmp, FR_min, window, mingap, standard_gap):
    """
    PARAMETERS:
    
    tmp - dataframe of spikes in ONE cell assembly
    FR_min - threshold, below which we consider there to be a downstate
    window - interval, ms, in which we count spikes for FR
    mingap - minimum length of downstate that you want to delete
    standard_gap - the length of gap you want to truncate gaps larger than mingap (must be smaller than mingap)
    
    RETURNS
    ss - dataframe of upstate boundaries
    df_ - original spikes with a column that marks which spikes are not part of an upstate
    us_nogap - original dataframe with downstates dropped """

    @jit(fastmath=True, nopython=True, parallel=True, nogil=True)
    def fast_(spts, x, window):
        ma1 = np.zeros_like(spts)
        for i in prange(len(spts)):
            ma1[i] = np.sum((x >= (spts[i]- window/2)) & (x < (spts[i]+window/2)))
        return ma1
    
    df_ = tmp.copy()
    spts = df_.spiketime.to_numpy()
    x = df_.spiketime.to_numpy()
    t = time.time()

    ma1 = fast_(spts, x, window)
    df_['drop'] = ma1 < FR_min                    # mark down-states for dropping

    # get upstate start and end times in a dataframe:
    ss_ = []
    seqstart = False
    for i, fr in enumerate(ma1):
        if fr > FR_min and not seqstart:
            start = spts[i]
            seqstart = True
        if fr <= FR_min and seqstart:
            seqstart = False
            stop = spts[i]
            ss_.append((start, stop, stop-start))
    ss = pd.DataFrame(ss_, columns=['start', 'stop', 'leng'])
    
    
    # drop the downstates:
    upstates = df_[df_['drop'] == False]
    x = upstates.spiketime.diff().values
    y = [i if i>mingap else 0 for i in x]
    y_ = [i-standard_gap if i>mingap else i for i in y]

    # new dataframe of spikes without downstates
    us_nogap = upstates.copy()
    us_nogap.reset_index(drop=True, inplace=True)
    us_nogap.spiketime -= np.cumsum(y_)
    us_nogap.spiketime -= us_nogap.spiketime.min()
    
    return ss, df_, us_nogap

def get_chain(t):
    global m, wchain, chain, NE, NI
    x = np.copy(m.getWeights()[:NE, :NE])

    wchain_ = []
    for link in chain:
        wchain_.append(x[link[::-1]])
    plt.figure(figsize=(16,4))
    plt.plot(wchain)
    plt.plot(wchain_)
    plt.ylim(0, 0.8)
    plt.grid()
    plt.title(f'Time: {t}')
    plt.gca().get_xticks()
    plt.xticks(ticks=range(len(chain)), labels=chain)
    
    plt.savefig(f'snap/snap_{str(t).zfill(4)}.png', dpi=300)
    plt.close()

def clusterize(w):
    global NE, NI
    x = np.copy(w[:NE, :NE])
    G = nx.from_numpy_matrix(x)
    adj_mat = nx.to_numpy_array(G)
    louvain = Louvain()
    labels = louvain.fit_transform(adj_mat)
    mod = modularity(adj_mat, labels)

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup:tup[1], reverse=True)
    newids = [i[0] for i in tmp]

    W_ = x
    W_ = W_[newids, :]
    W_ = W_[:, newids]
    return W_, labels, counts, mod, newids

def bruteforce_sequences(candidate):
    global tmp
    S = List()
    for nid in candidate:
        S.append(tmp[tmp.neuronid == nid].spiketime.to_numpy())

    c, duplicates, not_duplicates = new_find_seq(S)
    return c, duplicates, not_duplicates, candidate

@njit
def new_find_seq(S: List) -> (list, set, set):
    A = [] # links
    for it, I in enumerate(S[1:]):
        if it == 0:
            J = S[0]
        a = np.zeros((len(I), len(J)))
        for iid, i in enumerate(I):
            for jid, j in enumerate(J):
                a[iid, jid] = i-j
        good_iids, good_jids = np.where((a > 0.0) & (a < 5.0))
        A.append([[j, i] for i,j in zip(I[good_iids], J[good_jids])])
        J = np.unique(I[good_iids])


    # assemble links end to start:
    a_ = A[0]
    c = []  # assembled links
    for b_ in A[1:]:
        for a in a_:
            for b in b_:
                if a[-1] == b[0]:
                    c.append(a + [b[1]])
        a_ = c


    # remove shorter sequences that overlap with longer ones entirely
    duplicates, not_duplicates = [], []
    for idx, x in enumerate(c):
        for y in c:
            if len(x) < len(y):
                acc = 0.0
                for i in range(len(x)):
                    acc += y[i] - x[i]
                if acc == 0.0:
                    duplicates.append(idx)
                    break
    not_duplicates = set(range(len(c))) - set(duplicates)
    return [c[i] for i in not_duplicates], set(duplicates), not_duplicates

def clusterize2(x):
    G = nx.from_numpy_matrix(x)
    adj_mat = nx.to_numpy_array(G)
    louvain = Louvain(resolution=0.001)
    labels = louvain.fit_transform(adj_mat)
    mod = 0

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup:tup[1], reverse=True)
    newids = [i[0] for i in tmp]

    return labels, counts, mod, newids

def cluster_sequences(SEQ):
    X = np.zeros((len(SEQ), len(SEQ)))
    x = np.zeros((len(SEQ), len(SEQ)))
    
    for i in range(len(SEQ)):
        for j in range(len(SEQ)):
            if i == j:
                X[i, j] = 100
                x[i, j] = np.sum(np.abs(np.array(SEQ[i]) - np.array(SEQ[j])))
            else:
                X[i, j] = 1.0/(np.linalg.norm(np.array(SEQ[i]) - np.array(SEQ[j])))
                x[i, j] = np.sum(np.abs(np.array(SEQ[i]) - np.array(SEQ[j])))
    
    labels, counts, mod, newids = clusterize2(X)
    X_ = x
    X_ = X_[newids, :]
    X_ = X_[:, newids]
    
    return X_, labels, counts, mod, newids

@njit
def accfunc(neurons_i, neurons_j, Jepsilon):
    global W
    acc = []
    for neuron_i in neurons_i:
        for neuron_j in neurons_j:
            if W[neuron_i, neuron_j] > Jepsilon:
                acc.append(W[neuron_i, neuron_j])
    return acc

def make_UU(asInPaper, narrow=True):
    global m
    if asInPaper == 3:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, False
    elif asInPaper == 2:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13.3, 26.6, 1.0, 133, False
    elif asInPaper == 1:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, True
    elif asInPaper == 0:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13, 26, 1.0, 133, True
    else:
        raise

    N_ = int(8e6) # create an array with enough margin to delete values sampled above 1.0
    try:
        with open(f'/flash/FukaiU/roman/RelProbTestSims/RelProbTest_{case}_1_1/wts_wi_0.5.p', 'rb') as f:
            ww = pickle.load(f)[:2500, :2500]
        i, j = np.where(ww > 0.001)
    except:
        ww = np.copy(m.getWeights())[:2500, :2500]
        i, j = np.where(ww > params['Jepsilon'])
        

    if not narrow:
        x = np.random.gamma(shape_wide, scale_wide, N_)/divisor
        x = np.delete(x, np.where(x > 1))
        if not flip:
            x = (x[:2500*2500]).reshape(2500, 2500)
        else:
            x = - (x[:2500*2500]).reshape(2500, 2500) + 1
        UU_wide = np.zeros((2500, 2500))
        UU_wide[i, j] = x[i, j]
        UU_wide_mean = UU_wide[i, j].mean()
        return UU_wide, i, j

    else:    
        x = np.random.gamma(shape_narrow, scale_narrow, N_)/divisor
        x = np.delete(x, np.where(x > 1))
        if not flip:
            x = (x[:2500*2500]).reshape(2500, 2500)
        else:
            x = - (x[:2500*2500]).reshape(2500, 2500) + 1
        UU_narrow = np.zeros((2500, 2500))
        UU_narrow[i, j] = x[i, j]
        UU_narrow_mean = UU_narrow[i, j].mean()
        return UU_narrow, i, j


relprob_dist = bool(int(sys.argv[1]))
case = int(sys.argv[2])
narrow = bool(int(sys.argv[3]))
HAGA = bool(int(sys.argv[4]))
NneurStim = int(sys.argv[5])
stimDur = int(sys.argv[6])

cprint('****************** SANITY CHECK *******************', 'blue', 'on_yellow')
pprint( {'relprob_dist': relprob_dist,
         'case': case,
         'narrow': narrow,
         'HAGA': HAGA,
         'NneurStim': NneurStim,
         'stimDur': stimDur})


cell_id = 6347
pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
# pathToBucket = '/home/roman/bucket/slurm_big_HAGAwoFD/'

dt = 0.01
NEo = 0
NE = 400 + NEo
NI = 80  #157


df = pd.read_pickle('df.pkl')

# the order of keys IS IMPORTANT for the cClasses not to break down
params = {
    "alpha": 50.0,    # Degree of log-STDP (50.0)
    "JEI": 0.15,      # 0.15 or 0.20

    "T": 1800*1000.0,   # simulation time, ms
    "h": 0.01,          # time step, ms ??????

    # probability of connection
    "cEE": 0.4, # 
    "cIE": 0.2, #
    "cEI": 0.5, #
    "cII": 0.5, #

    # Synaptic weights
    "JEE": 0.15, #
    "JEEinit": 0.16, # ?????????????
    "JIE": 0.15, # 
    "JII": 0.06, #
    
    #initial conditions of synaptic weights
    "JEEh": 0.15, # Standard synaptic weight E-E
    "sigJ": 0.3,  #

    "Jtmax": 0.25, # J_maxˆtot
    "Jtmin": 0.01, # J_minˆtot # ??? NOT IN THE PAPER

    # Thresholds of update
    "hE": 1.0, # Threshold of update of excitatory neurons
    "hI": 1.0, # Threshold of update of inhibotory neurons

    "IEex": 2.0, # Amplitude of steady external input to excitatory neurons
    "IIex": 0.5, # Amplitude of steady external input to inhibitory neurons
    "mex": 0.3,        # mean of external input
    "sigex": 0.1,      # variance of external input

    # Average intervals of update, ms
    "tmE": 2.5,  #t_Eud EXCITATORY <<<<<<<<<<<<<<<<<< decreased by a factor of 5
    "tmI": 2.5,  #t_Iud INHIBITORY <<<<<<<<<<<<<<<<<< decreased by a factor of 5
    
    #Short-Term Depression
    "trec": 600.0,     # recovery time constant (tau_sd, p.13 and p.12)
    "Jepsilon": 0.001, # ????????
    
    # Time constants of STDP decay
    "tpp": 20.0,  # tau_p
    "tpd": 40.0,  # tau_d
    "twnd": 500.0, # STDP window lenght, ms
    
    "g": 1.25,        # ??????
    
    #homeostatic
    "itauh": 100,       # decay time of homeostatic plasticity, (100s)
    "hsd": 0.1,
    "hh": 10.0,  # SOME MYSTERIOUS PARAMETER
    "Ip": 1.0,   # External current applied to randomly chosen excitatory neurons
    "a": 0.20,   # Fraction of neurons to which this external current is applied
    
    "xEinit": 0.02, # the probability that an excitatory neurons spikes at the beginning of the simulation
    "xIinit": 0.01, # the probability that an inhibitory neurons spikes at the beginning of the simulation
    "tinit": 100.00, # period of time after which STDP kicks in (100.0)
    "U": 0.6,
    "taustf": 200,
    "taustd": 500,
    "Cp": 0.01875,
    "Cd": 0.0075,
    "HAGA": bool(HAGA)} 


params['HAGA'] = HAGA
params['JEE'] = 0.45

params["U"] = df.iloc[cell_id]['U']
params['g'] = 2.5
params["tinit"] = 100
params["JEEinit"] = 0.46 # 0.16
params["Cp"] = df.iloc[cell_id]['Cp']
params["Cd"] = df.iloc[cell_id]['Cd']
params["tpp"] = df.iloc[cell_id]['tpp']
params["tpd"] = df.iloc[cell_id]['tpd']
params["taustf"] = df.iloc[cell_id]['taustf']
params["taustd"] = df.iloc[cell_id]['taustd']
params["alpha"] = 50.00
params["itauh"] = 100


# set params that are good for woFD self-organization:
if not HAGA:
    for k,v in {'Cp': 0.06, 'Cd': 0.01, 'taustf': 450.0, 'taustd': 200.0, 'tpp': 15.1549, 'tpd': 120.4221}.items():
        params[k] = v


# build model
m = cClassOne(NE, NI, NEo, cell_id)
m.setParams(params)
m.saveSpikes(1)

m.set_STDP(True)
m.set_mex(0.3)
m.set_hEhI(1.0, 1.0)
m.set_HAGA(HAGA)


# sample release probs from the distribution you need
UU, i, j = make_UU(case, narrow=narrow)
UU = UU[i,j][:NE].tolist()


if relprob_dist:
    # set these relsease probs
    m.setUU(np.ascontiguousarray(UU))



# remember the newids with which to sort the raster and weights
W = np.copy(m.getWeights())
w_, labels, counts, mod, newids = clusterize(W)    
map_dict = {j:i for i,j in enumerate(newids)}


WgtEvo, FEvo, DEvo, MOD, NASS = [], [], [], [], []

wandb.config = {k: v for k, v in params.items()}
wandb.config['relprob_dist'] = relprob_dist
wandb.config['case'] = case
wandb.config['narrow'] = narrow
wandb.config['NneurStim'] = NneurStim
wandb.config['stimDur'] = stimDur
wandb.init(project="bmm_small_with_perturb_2", entity="nightdude", config=wandb.config)
wandb.run.name = f"{os.getcwd().split('/')[-1]}_{wandb.run.id}"



cell_id = 6347
pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
# pathToBucket = '/home/roman/bucket/slurm_big_HAGAwoFD/'

dt = 0.01
NEo = 0
NE = 400 + NEo
NI = 80  #157

for aii, niter, stimulate_ in [(0, 100, False), (1, stimDur, True), (2, 100, False)]:
    m.set_hEhI(1.0, 1.0)
    m.set_STDP(True)
    m.set_homeostatic(True)
    stimulate = stimulate_
    print(f'\n\
          STDP: {"on" if m.get_STDP() else "off"}\n\
          Homeostatic: {"on" if m.get_homeostatic() else "off"}\n\
          hE: {m.getState().hE}\n\
          hI: {m.getState().hI}\n\
          stim: {"on" if stimulate else "off"}\n\
          HAGA: {HAGA}\n\
          relprob_dist: {relprob_dist}\n\
          case: {case}\n\
          narrow: {narrow}\n')


    if stimulate:
        x = np.zeros((NE,)).astype('int32')
        x[NE-NEo:] = 1
        m.setStim(x)
        x1 = np.zeros((NE,)).astype(float)
        m.setStimIntensity(x1)
    else:
        x = np.zeros((NE,)).astype('int32')
        m.setStim(x)
        x1 = np.zeros((NE,)).astype(float)
        m.setStimIntensity(x1)

    pbar = trange(niter, desc=f'Time : {m.getState().t:.2f}') # 2000 for 400 s
    for i in pbar:

        if stimulate:
            x = np.zeros((NE,)).astype('int32') 
            x1 = np.zeros((NE,)).astype(float)
            x[0:NneurStim] = 1
            x1[0:NneurStim] = 1.0
            m.setStim(x)
            m.setStimIntensity(x1)
            m.sim(10000)
            
            # no stim interval
            x1 = np.zeros((NE,)).astype(float)
            m.setStimIntensity(x1)
            m.sim(10000)
        else:
            # no stim interval
            x1 = np.zeros((NE,)).astype(float)
            m.setStimIntensity(x1)
            m.sim(20000)
        
        if i % 1 == 0:
            W = np.copy(m.getWeights())[:NE-NEo, :NE-NEo]
            w_, labels, counts, mod, newids = clusterize(W)
            map_dict = {j:i for i,j in enumerate(newids)}

            t = m.getState().t
            n_ass = len(np.unique(labels))
            pbar.set_description(f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {n_ass}')
            WgtEvo.append(calcAssWgts(t))
            FEvo.append(calcF(t))
            DEvo.append(calcD(t))
            MOD.append(mod)
            NASS.append(n_ass)
            wandb.log({'NASS': n_ass,
                       'MOD': mod,
                       't': t
                       })

    if aii == 0: 
        W_bak = np.copy(m.getWeights())
        with open(f'W_bak_{cell_id}.p', 'wb') as f:
            pickle.dump(W_bak, f)
        m.dumpSpikeStates()
        states = m.getState()
        p = pd.DataFrame([(p, states.__getattribute__(p)) for p in dir(states) if not p.startswith('_')])
        p.columns = ['param', 'value']
        p.index = p.param
        p.drop(columns='param', inplace=True)
        with open(f'states_{cell_id}.json', 'w') as f:
            f.write(json.dumps(p.value.to_dict()))

with open('MOD', 'w') as f:
    f.writelines([str(it) + '\n' for it in MOD])
with open('NASS', 'w') as f:
    f.writelines([str(it) + '\n' for it in NASS])

fig, ax = plt.subplots(1, 4, figsize=(18,4))
ax[0].plot(m.getTheta())
ax[1].plot(x1)
ax[2].plot(MOD)
b = [0.1]
for i in range(1000):
    b.append(b[-1]*0.99)
ax[3].plot(b)
plt.savefig('fig2.jpg', dpi=300)
wandb.log({"fig2": wandb.Image(fig)})
plt.close('all')





fig, ax = plt.subplots(2, 3, figsize=(18,12))

w_, labels, counts, mod, newids = clusterize(W_bak)
map_dict = {j:i for i,j in enumerate(newids)}
ax[0,1].imshow(w_)
ax[0,1].set_title(f'Before perturbation, N = {len(np.unique(labels))}\nsorted')
ax[0,1].set_xticks([0, 399])
ax[0,1].set_yticks([0, 399])

pd.DataFrame(WgtEvo).plot(x='t', y=[f'{i}-{i}' for i in np.unique(labels)], ax=ax[0,0], lw=1.5)
pd.DataFrame(WgtEvo).plot(x='t', y=[f'{0}-{i}' for i in np.unique(labels)[1:]], ax=ax[0,0], c='k', lw=0.5)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)


ax[0,0].axvspan(412000, 412000+60*200, label='perturb on', color='red', alpha=0.3)
ax[0,0].legend(ncol=2, fontsize=10, loc='lower right')
scalebar = AnchoredSizeBar(ax[0,0].transData, 
                           size=10000, label='10 s', loc='lower left', 
                           pad=1.3, frameon=False, size_vertical=0.015, color='black',
                           fontproperties=matplotlib.font_manager.FontProperties(size=22))
ax[0,0].add_artist(scalebar)
ax[0,0].set_ylim(0, 2.5)
# ax[0].get_legend().remove()

W_now = np.copy(m.getWeights())[:NE-NEo, :NE-NEo]
w_now, labels_now, counts_now, mod_now, newids_now = clusterize(W_now)
ax[0,2].imshow(w_now)
ax[0,2].set_xticks([0, 399])
ax[0,2].set_yticks([0, 399])

# ax[0].legend(loc='upper right', ncol=3)
# _ = plt.setp(ax[0].get_legend().get_texts(),fontsize=10)


ax[0,2].set_title(f'After perturbation, N = {len(np.unique(labels_now))}')


W_now = np.copy(m.getWeights())[:NE-NEo, :NE-NEo]
w_now, labels_now, counts_now, mod_now, newids_now = clusterize(W_now)
w_before, labels_before, counts_before, mod_before, newids_before = clusterize(W_bak)
map_dict_before = {j:i for i,j in enumerate(newids)}

ax[1,0].imshow(W_now)
ax[1,0].set_title(f'After perturbation, N = {len(np.unique(labels))}\nunsorted')
ax[1,0].set_xticks([0, 399])
ax[1,0].set_yticks([0, 399])

ax[1,1].imshow(W_now[:, newids_before][newids_before, :])
ax[1,1].set_title(f'After perturbation, N = {len(np.unique(labels))}\nsorted using the previous labels')
ax[1,1].set_xticks([0, 399])
ax[1,1].set_yticks([0, 399])

ax[1,2].imshow(W_bak[:NE-NEo, :NE-NEo])
ax[1,2].set_title(f'Before perturbation, N = {len(np.unique(labels))}\nunsorted')
ax[1,2].set_xticks([0, 399])
ax[1,2].set_yticks([0, 399])

plt.savefig('fig3.jpg', dpi=300)
wandb.log({"fig3": wandb.Image(fig)})
plt.close('all')





fig, ax = plt.subplots(3,1, figsize=(15,7), sharex=True)

pd.DataFrame(WgtEvo).plot(x='t', y=[f'{i}-{i}' for i in range(6)], ax=ax[0])
ax[0].legend(loc='lower right', ncol=2)
plt.setp(ax[0].get_legend().get_texts(),fontsize=14)
ax[0].set_title('Average within-assebmly weights', fontsize=16)

pd.DataFrame(WgtEvo).plot(x='t', y=[f'{i}-{i+1}' for i in range(5)], ax=ax[1])
ax[1].legend(loc='upper right', ncol=2)
plt.setp(ax[1].get_legend().get_texts(),fontsize=14)
ax[1].set_title('Average between-assebmly weights', fontsize=16)


pd.DataFrame(WgtEvo).plot(x='t', y=[f'{i+1}-{i}' for i in range(5)], ax=ax[2])
ax[2].legend(loc='upper right', ncol=2)
plt.setp(ax[2].get_legend().get_texts(),fontsize=14)
ax[2].set_title('Average between-assebmly weights', fontsize=16)


for i in range(3):
    ax[i].tick_params(axis='x', labelsize=16)
    ax[i].tick_params(axis='y', colors='red', labelsize=16)
    ax[i].tick_params(axis='y', colors='k', labelsize=16)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].set_xlim(pd.DataFrame(WgtEvo).t.min(), pd.DataFrame(WgtEvo).t.max())
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
scalebar = AnchoredSizeBar(ax[0].transData, 
                           size=100000, label='100 s', loc='lower left', 
                           pad=0.1, frameon=False, size_vertical=0.005, color='black',
                           fontproperties=matplotlib.font_manager.FontProperties(size=22))
ax[0].add_artist(scalebar)

ax[2].set_xlabel('Time [ms]')

plt.savefig('fig4.jpg', dpi=300)
wandb.log({"fig4": wandb.Image(fig)})
plt.close('all')

wandb.finish()