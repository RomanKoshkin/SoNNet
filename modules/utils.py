import enum
from itertools import chain, groupby, repeat
from collections import Counter
import torch
import torch.nn.functional as F

import sys, time, json, yaml, subprocess, copy
from tqdm import trange, tqdm
from modules.constants import bar_format

import os, pickle, shutil, warnings, shutil
import numpy as np
from multiprocessing import Pool, Array
from ctypes import c_double, c_int
from scipy import signal
import numba, string
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
from collections import namedtuple

from scipy.sparse import lil_matrix
from .constants import COLORS

try:
    import sknetwork
    if sknetwork.__version__ == '0.20.0':
        SKNETWORK_OLD = True
        from sknetwork.clustering import Louvain, modularity
    elif sknetwork.__version__ == '0.30.0':
        from sknetwork.clustering import Louvain
        from sknetwork.clustering import get_modularity as modularity
        SKNETWORK_OLD = False
    else:
        raise ValueError("You must have scikit-network of the following versions: 0.20.0 or 0.30.0")
except Exception as e:
    cprint('sknetwork PROBABLY failed to import', color='yellow')


class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class SuppressTqdmOutput:

    def __enter__(self):
        # Redirect stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Save the original tqdm classes and clear instances
        self._original_tqdm = tqdm._instances.copy()
        tqdm._instances.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the open file descriptors and restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        # Restore the original tqdm instances
        tqdm._instances = self._original_tqdm


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SlurmJobDispatcher(object):

    def __init__(
        self,
        path_to_job_file_on_deigo,
        path_to_job_file_from_precision,
        job_file_name,
        output_folder_on_slurm,
        max_slurm_jobs,
    ):
        self.path_to_job_file_on_deigo = path_to_job_file_on_deigo
        self.path_to_job_file_from_precision = path_to_job_file_from_precision
        self.job_file_name = job_file_name
        self.output_folder_on_slurm = output_folder_on_slurm
        self.max_slurm_jobs = max_slurm_jobs
        self.numOfJobsSubmitted = 0

    def checkNumJobs(self):
        """ check the number of jobs running on slurm 
            takes about 4 s to execute
        """
        subproc = subprocess.Popen("ssh deigo 'squeue'", shell=True, stdout=subprocess.PIPE)
        ret = subproc.stdout.read().decode('UTF-8').split('\n')[1:-1]
        n_jobs_now = len(ret)
        msg = f'Number of jobs running: {n_jobs_now}'
        sys.stdout.write(msg)
        sys.stdout.flush()
        sys.stdout.write("\b" * len(msg))
        return n_jobs_now

    def MakeAJobFile(self, _dna, genID, cell_id):
        """ compile a job file to run each job in slurm """

        dna = copy.deepcopy(_dna)
        # dna['gen'] = genID
        # dna['cell_id'] = cell_id
        dna_str = " ".join(f"--{k}={v}" if k not in ['script', 'program'] else f"{v}" for k, v in dna.items()) + "\n"
        with open(f'{self.path_to_job_file_from_precision}/{self.job_file_name}', 'w') as f:
            f.writelines("#!/bin/bash\n")
            f.writelines(f"#SBATCH --job-name=JOBNAME1\n")
            f.writelines(f"#SBATCH --mail-user=roman.koshkin@oist.jp\n")
            f.writelines(f"#SBATCH --partition=short\n")
            f.writelines(f"#SBATCH --ntasks=1\n")
            f.writelines(f"#SBATCH --cpus-per-task=1\n")
            f.writelines(f"#SBATCH --mem-per-cpu=1g\n")
            f.writelines(f"#SBATCH --output=./{self.output_folder_on_slurm}/%j.out\n")
            f.writelines(f"#SBATCH --array=1-1\n")  # submit 4 jobs as an array, give them individual id from 1 to 4
            f.writelines(f"#SBATCH --time=0:15:00\n")
            f.writelines(dna_str)
            # f.writelines("python sim.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $1\n")

    def runJob(self):
        """ run a slurm job based on the file just compiled and written to deigo """

        if self.numOfJobsSubmitted > int(self.max_slurm_jobs * 0.9):
            num_unfinished_jobs = self.checkNumJobs()
            while num_unfinished_jobs > self.max_slurm_jobs:
                time.sleep(5)
                num_unfinished_jobs = self.checkNumJobs()
        commands = f'"cd {self.path_to_job_file_on_deigo} && sbatch {self.job_file_name}"'
        os.system(f"ssh deigo {commands}")
        self.numOfJobsSubmitted += 1
        time.sleep(0.3)  # NOTE: sleep not to constipate the slurm scheduler

    def next_job(self, dna):
        self.MakeAJobFile(dna, None, None)
        self.runJob()


class Stimulator(object):

    def __init__(
        self,
        m,
        stim_strength=1.0,
        nass=8,
        rotate_every_ms=20,
        cell_id=0,
        dump_stats=False,
        overlap=0,
        lif=False,
        dumpfolder='data',
    ):
        self.dumpfolder = dumpfolder
        cprint(f'Dumping data to {dumpfolder}', color='magenta')
        self.lif = lif
        self.NE = m.getState().NE
        self.N = m.getState().NE + m.getState().NI
        self.cell_id = cell_id
        # assert self.NE % nass == 0, 'check ass_size relative to NE'
        ass_size = self.NE // nass
        self.patterns = [[i
                          for i in range(j, np.clip(j + ass_size, a_min=0, a_max=self.NE))]
                         for j in [k for k in range(0, self.NE, ass_size - overlap)]]
        cprint(f'N patterns: {len(self.patterns)}', color='blue')
        # self.patterns = self.patterns[:nass]  # NOTE: drop the "pattern" whose id is greater than nass

        self.rotate_every_ms = rotate_every_ms

        self.pat_ID = 0
        self.m = m
        self.pat_st_t = m.getState().t
        self.stim_strength = stim_strength
        self.dump_stats = dump_stats
        self.FD = []
        self.EX = []
        self.sim = self.m.sim if not self.lif else self.m.sim_lif

    def get_next(self):
        self.pat_ID += 1
        if self.pat_ID >= len(self.patterns):
            self.pat_ID = 0
        return np.array(self.patterns[self.pat_ID])

    def check_if_rotate_stim(self):
        now = self.m.getState().t
        if now - self.pat_st_t > self.rotate_every_ms:
            self.pat_st_t = now
            x = np.zeros((self.NE,)).astype('int32')
            x1 = np.zeros((self.NE,)).astype(float)
            pat_ids = self.get_next()
            x[pat_ids] = 1  # the trailing excitatory neurons are input neurons
            x1[pat_ids] = self.stim_strength
            self.m.setStim(x)
            self.m.setStimIntensity(x1)

    def sham_stim(self):
        x = np.zeros((self.NE,)).astype('int32')
        x1 = np.zeros((self.NE,)).astype(float)
        self.m.setStim(x)
        self.m.setStimIntensity(x1)

    def set_pattern(self, pattern):
        x = np.zeros((self.NE,)).astype('int32')
        x1 = np.zeros((self.NE,)).astype(float)
        x[pattern] = 1  # the trailing excitatory neurons are input neurons
        x1[pattern] = self.stim_strength
        self.m.setStim(x)
        self.m.setStimIntensity(x1)

    def train(self,
              dur_ms,
              patternLenMs=10,
              clusterize_=False,
              callback=None,
              rwPatID=None,
              saveFD=True,
              random_order_of_patterns=False):

        h = self.m.getState().h
        msPerEpoch = 2
        stepsPerEpoch = int(msPerEpoch / h)

        fullPatternPresentationDurMs = patternLenMs * len(self.patterns)
        fullPatternPresentations = dur_ms // fullPatternPresentationDurMs
        epochsPerPattern = patternLenMs // msPerEpoch

        self.m.set_STDP(True)
        self.m.set_homeostatic(True)

        pbar = trange(fullPatternPresentations, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)
        for step in pbar:
            try:
                if clusterize_:
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)

                patIDs = np.array(range(len(self.patterns)))
                if random_order_of_patterns:
                    np.random.shuffle(patIDs)
                for patID in patIDs:
                    pattern = self.patterns[patID]
                    self.set_pattern(np.array(pattern))
                    if patID == rwPatID:
                        _stepsPerEpoch = stepsPerEpoch * 30
                    else:
                        _stepsPerEpoch = stepsPerEpoch
                    for ep in range(epochsPerPattern):
                        self.sim(_stepsPerEpoch)  # do one epoch
                        time_now = self.m.getState().t
                        if time_now > 200:
                            try:
                                Uexc = self.m.get_Uexc()
                                Uinh = self.m.get_Uinh()
                                FRs = self.m.get_FR()
                            except:
                                Uexc = np.zeros((self.N,))
                                Uinh = np.zeros((self.N,))
                                FRs = np.zeros((self.N,))
                            if saveFD:
                                self.EX.append({
                                    't': time_now,
                                    'totExcOntoExc': Uexc[:self.NE].mean(),
                                    'totInhOntoExc': Uinh[:self.NE].mean(),
                                    'totExcOntoInh': Uexc[self.NE:].mean(),
                                    'totInhOntoInh': Uinh[self.NE:].mean(),
                                    'f_rates': FRs,
                                    'F': self.m.getF().astype('float16'),
                                    'D': self.m.getD().astype('float16')
                                })
                        if callback is not None:
                            callback.record(self.m)
                        if clusterize_:
                            nass = np.unique(labels).shape[0]
                            pbar.set_description(
                                f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()} clus: {clusterize_}'
                            )
                            if np.round(self.m.getState().t) % self.dump_stats == 0:
                                self.dropstats(mod, nass, np.nan)
                        else:
                            pbar.set_description(
                                f't: {time_now/1000:.2f} s, STDP: {self.m.get_STDP()} clus: {clusterize_}')
            except KeyboardInterrupt:
                print("User interrupt")
                break

        # Z, Zsm, sp = getRecentSpikesAndCondDistOfCAs(
        #     self,
        #     last_ms=1000,
        #     window='10ms',
        #     assIDcolName='org_aid',
        # )
        # self.targets = calc_targets(Z)
        # print('targets calculated')

    def ping(self, patternLenMs=10, pattern_id=0, msPerEpoch=2, plasticity=False, callback=None, saveFD=True):

        h = self.m.getState().h
        stepsPerEpoch = int(msPerEpoch / h)
        epochsPerPattern = patternLenMs // msPerEpoch

        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)

        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
        w_, labels, counts, mod, newids = clusterize(W)

        pattern = self.patterns[pattern_id]
        self.set_pattern(np.array(pattern))
        pbar = trange(epochsPerPattern, bar_format=bar_format)
        for ep in pbar:
            self.sim(stepsPerEpoch)  # do one epoch
            time_now = self.m.getState().t
            w_, labels, counts, mod, newids = clusterize(W)
            nass = np.unique(labels).shape[0]
            pbar.set_description(
                f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()}')
            if np.round(self.m.getState().t) % self.dump_stats == 0:
                self.dropstats(mod, nass, np.nan)

            try:
                Uexc = self.m.get_Uexc()
                Uinh = self.m.get_Uinh()
                FRs = self.m.get_FR()
            except:
                Uexc = np.zeros((self.N,))
                Uinh = np.zeros((self.N,))
                FRs = np.zeros((self.N,))

            if saveFD:
                self.EX.append({
                    't': time_now,
                    'totExcOntoExc': Uexc[:self.NE].mean(),
                    'totInhOntoExc': Uinh[:self.NE].mean(),
                    'totExcOntoInh': Uexc[self.NE:].mean(),
                    'totInhOntoInh': Uinh[self.NE:].mean(),
                    'f_rates': FRs,
                    'F': self.m.getF().astype('float16'),
                    'D': self.m.getD().astype('float16')
                })
            if callback is not None:
                callback.record(self.m)

    def perturb(self, dur_ms, stID=0, enID=100, plasticity=True, clusterize_=False, callback=None, saveFD=True):

        h = self.m.getState().h
        msPerEpoch = 2
        stepsPerEpoch = int(msPerEpoch / h)

        epochs = int(dur_ms / msPerEpoch)

        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)

        x = np.zeros((self.NE,)).astype('int32')
        x1 = np.zeros((self.NE,)).astype(float)
        x[stID:enID] = 1
        x1[stID:enID] = 1.0

        self.m.setStim(x)
        self.m.setStimIntensity(x1)

        if clusterize_:
            W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
            w_, labels, counts, mod, newids = clusterize(W)

        pbar = trange(epochs, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)
        for ep in pbar:

            try:
                self.sim(stepsPerEpoch)  # do one epoch
                time_now = self.m.getState().t
                if time_now > 200:
                    try:
                        Uexc = self.m.get_Uexc()
                        Uinh = self.m.get_Uinh()
                        FRs = self.m.get_FR()
                    except:
                        Uexc = np.zeros((self.N,))
                        Uinh = np.zeros((self.N,))
                        FRs = np.zeros((self.N,))
                    if saveFD:
                        self.EX.append({
                            't': time_now,
                            'totExcOntoExc': Uexc[:self.NE].mean(),
                            'totInhOntoExc': Uinh[:self.NE].mean(),
                            'totExcOntoInh': Uexc[self.NE:].mean(),
                            'totInhOntoInh': Uinh[self.NE:].mean(),
                            'f_rates': FRs,
                            'F': self.m.getF().astype('float16'),
                            'D': self.m.getD().astype('float16')
                        })
                    if callback is not None:
                        callback.record(self.m)

                if clusterize_:
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()}'
                    )
                else:
                    pbar.set_description(f't: {time_now/1000:.2f} s, STDP: {self.m.get_STDP()}')
                if clusterize_:
                    if ep % 80 == 0:
                        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                        w_, labels, counts, mod, newids = clusterize(W)
                        nass = np.unique(labels).shape[0]
                    if np.round(self.m.getState().t) % self.dump_stats == 0:
                        # xent = calc_xent(self.targets, self)
                        # self.dropstats(mod, xent)
                        self.dropstats(mod, nass, np.nan)
            except KeyboardInterrupt:
                print("Interrupted by user")
                break

    def perturb_randM(self, dur_ms, M=None, plasticity=True, clusterize_=False, callback=None, saveFD=True):

        perturbed_ids = np.random.choice(self.NE, size=M, replace=False)
        h = self.m.getState().h
        msPerEpoch = 2
        stepsPerEpoch = int(msPerEpoch / h)

        epochs = int(dur_ms / msPerEpoch)

        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)

        x = np.zeros((self.NE,)).astype('int32')
        x1 = np.zeros((self.NE,)).astype(float)
        x[perturbed_ids] = 1
        x1[perturbed_ids] = 1.0

        self.m.setStim(x)
        self.m.setStimIntensity(x1)

        if clusterize_:
            W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
            w_, labels, counts, mod, newids = clusterize(W)

        pbar = trange(epochs, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)
        for ep in pbar:

            try:
                self.sim(stepsPerEpoch)  # do one epoch
                time_now = self.m.getState().t
                if time_now > 200:
                    try:
                        Uexc = self.m.get_Uexc()
                        Uinh = self.m.get_Uinh()
                        FRs = self.m.get_FR()
                    except:
                        Uexc = np.zeros((self.N,))
                        Uinh = np.zeros((self.N,))
                        FRs = np.zeros((self.N,))
                    if saveFD:
                        self.EX.append({
                            't': time_now,
                            'totExcOntoExc': Uexc[:self.NE].mean(),
                            'totInhOntoExc': Uinh[:self.NE].mean(),
                            'totExcOntoInh': Uexc[self.NE:].mean(),
                            'totInhOntoInh': Uinh[self.NE:].mean(),
                            'f_rates': FRs,
                            'F': self.m.getF().astype('float16'),
                            'D': self.m.getD().astype('float16')
                        })
                    if callback is not None:
                        callback.record(self.m)

                if clusterize_:
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()}'
                    )
                else:
                    pbar.set_description(f't: {time_now/1000:.2f} s, STDP: {self.m.get_STDP()}')
                if clusterize_:
                    if ep % 80 == 0:
                        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                        w_, labels, counts, mod, newids = clusterize(W)
                        nass = np.unique(labels).shape[0]
                    if np.round(self.m.getState().t) % self.dump_stats == 0:
                        # xent = calc_xent(self.targets, self)
                        # self.dropstats(mod, xent)
                        self.dropstats(mod, nass, np.nan)
            except KeyboardInterrupt:
                print("Interrupted by user")
                break

    def sham(self, dur_ms, plasticity=True, clusterize_=False, callback=None, saveFD=True):

        h = self.m.getState().h
        msPerEpoch = 2
        stepsPerEpoch = int(msPerEpoch / h)

        epochs = int(dur_ms / msPerEpoch)

        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)

        self.sham_stim()

        if clusterize_:
            W = self.m.getWeights()[:self.NE, :self.NE]
            w_, labels, counts, mod, newids = clusterize(W)

        pbar = trange(epochs, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)
        for ep in pbar:

            try:
                self.sim(stepsPerEpoch)  # do one epoch
                time_now = self.m.getState().t
                if time_now > 200:
                    try:
                        Uexc = self.m.get_Uexc()
                        Uinh = self.m.get_Uinh()
                        FRs = self.m.get_FR()
                    except:
                        Uexc = np.zeros((self.N,))
                        Uinh = np.zeros((self.N,))
                        FRs = np.zeros((self.N,))

                    if saveFD:
                        self.EX.append({
                            't': time_now,
                            'totExcOntoExc': Uexc[:self.NE].mean(),
                            'totInhOntoExc': Uinh[:self.NE].mean(),
                            'totExcOntoInh': Uexc[self.NE:].mean(),
                            'totInhOntoInh': Uinh[self.NE:].mean(),
                            'f_rates': FRs,
                            'F': self.m.getF().astype('float16'),
                            'D': self.m.getD().astype('float16')
                        })
                    if callback is not None:
                        callback.record(self.m)

                if clusterize_:
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()}'
                    )
                else:
                    pbar.set_description(f't: {time_now/1000:.2f} s, STDP: {self.m.get_STDP()}')
                if clusterize_:
                    if ep % 80 == 0:
                        W = self.m.getWeights()[:self.NE, :self.NE]
                        w_, labels, counts, mod, newids = clusterize(W)
                        nass = np.unique(labels).shape[0]
                    if np.round(self.m.getState().t) % self.dump_stats == 0:
                        # xent = calc_xent(self.targets, self)
                        # self.dropstats(mod, xent)
                        self.dropstats(mod, nass, np.nan)
            except KeyboardInterrupt:
                print("Interrupted by user")
                break

    def dropstats(self, mod, nass, xent):
        t = int(np.round(self.m.getState().t))
        self.FD.append({'t': t})
        # fd = self.m.getF() * self.m.getD()
        # inh = np.copy(self.m.getUinh())
        # exc = np.copy(self.m.getUexc())
        theta = self.m.getTheta()
        # d[f'inh'] = inh  #np.mean(inh[pat])
        # d[f'exc'] = exc  #np.mean(exc[pat])
        # d[f'fd'] = fd  # np.mean(fd[pat])
        # d[f'th'] = theta  #np.mean(theta[pat])
        try:
            inh = self.m.get_Uinh()
            exc = self.m.get_Uexc()
        except:
            inh = np.zeros((self.N,))
            exc = np.zeros((self.N,))
        d = {
            'spiketime': t,
            'w': self.m.getWeights()[:self.NE, :self.NE],
            'inh': inh,
            'exc': exc,
            'mod': mod,
            'xent': xent,
            'theta': theta,
            'nass': nass,
        }
        np.save(f'{self.dumpfolder}/dump_{self.cell_id}_{t:010}', d)


def calc_xent(targets, stimulator):
    Z, Zsm, sp = getRecentSpikesAndCondDistOfCAs(stimulator, last_ms=1000, window='10ms', assIDcolName='org_aid')
    return F.cross_entropy(torch.from_numpy(Z), targets).item()


class Timer:

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f'{self.interval:.4f} s.')


def calc_targets(Z: np.array) -> torch.tensor:
    return torch.from_numpy(Z.argmax(axis=1))


def tonp(tensor):
    return tensor.detach().cpu().numpy()


def getRecentSpikesAndCondDistOfCAs(
    stimulator,
    last_ms=1000,
    window='10ms',
    assIDcolName='org_aid',
):
    sp = pd.read_csv(f'data/spike_times_{stimulator.cell_id}', delimiter=' ', header=None)
    sp.columns = ['spiketime', 'neuronid']
    sp = sp[sp.spiketime > sp.spiketime.max() - last_ms]
    NEo = 0  # FIXME
    NE = stimulator.m.getState().NE
    NI = stimulator.m.getState().NI
    sp = processSpikes(stimulator.m, sp, NE, NEo, NI)
    # sp[assIDcolName] = sp.neuronid.apply(map2oid, args=(stimulator.patterns,))  # python mode
    sp[assIDcolName] = sp.neuronid.apply(map2oid, args=(List(stimulator.patterns),))  # njit
    Z = getCondDistOfCAs(sp, window=window, assIDcolName=assIDcolName)
    Zsm = F.softmax(torch.from_numpy(Z), dim=1).numpy()
    return Z, Zsm, sp


@njit
def map2oid(x, patterns):
    for pid, p in enumerate(patterns):
        if x in p:
            return pid
    else:
        return -1


def getCondDistOfCAs(sp, window='40ms', assIDcolName='aid'):
    # cprint(f"using assembly labels from {assIDcolName}", color='magenta')
    a = sp.copy()
    a.spiketime -= a.spiketime.min()
    org = pd.Timestamp(
        year=2022,
        month=1,
        day=1,
    )
    a.index = a.apply(lambda x: org + pd.Timedelta(microseconds=int(x.spiketime * 100) * 10), axis=1)
    b = a[a[assIDcolName] != -1][assIDcolName].rolling(window=window).apply(lambda x: x.mode()[0])

    y = [int(k) for k, g in groupby(b.to_numpy())]
    nass = len(set(y))
    Z = np.zeros((nass, nass))

    pairs = [(y[i], y[i - 1]) for i in range(len(y) - 1)]
    counts = Counter(pairs)

    for it in pairs:
        try:
            Z[it] = counts[it]
        except Exception as e:
            print(e)
            pprint(counts)
    return Z


def plotSTDP(params, ax, disp=0, weight_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    
    cprint(f"weight_values to be plotted: {weight_values}", color='red')

    if isinstance(params, dict):
        params = namedtuple("params", params.keys())(*params.values())

    def fd(x, alpha, JEE):
        return np.log(1 + x * alpha / JEE) / np.log(1 + alpha)

    # our neuron that we focus on is at t = 0
    current_spike_t = 0

    # history of other spikes in the recurrent network (up until t = 0)
    spts = np.linspace(-250, 0, 200)

    for i, w in enumerate(weight_values):
        if params.symmetric:
            # we treat the current neuron as presynaptic, and the previous history of other spikes (spts) as postsynaptic:
            # we depress if presyn is later than postsyn
            dwd = params.Cp * np.exp(-(current_spike_t - spts) / params.tpp) - fd(
                w, params.alpha, params.JEE) * params.Cd * np.exp(-(current_spike_t - spts) / params.tpd)
            # we potentiate if presyn is later than postsyn
            dwp = params.Cp * np.exp(-(current_spike_t - spts) / params.tpp) - fd(
                w, params.alpha, params.JEE) * params.Cd * np.exp(-(current_spike_t - spts) / params.tpd)

        else:
            # we treat the current neuron as presynaptic, and the previous history of other spikes (spts) as postsynaptic:
            # we depress if presyn is later than postsyn
            dwd = -params.Cd * np.exp(-(current_spike_t - spts) / params.tpd)
            # we potentiate if presyn is later than postsyn
            dwp = params.Cp * np.exp(-(current_spike_t - spts) / params.tpp)

        stdp_dict = {
            'dt': np.concatenate([disp + spts - current_spike_t, (disp + current_spike_t - spts)[::-1]]),
            'dw': np.concatenate([dwd, dwp[::-1]]),
        }
        # ax.plot(stdp_dict['dt'], stdp_dict['dw'], color=COLORS[i], label=f'w={w}')
        ax.plot(stdp_dict['dt'], stdp_dict['dw'], color='k', label=f'w={w}')
        auc = np.trapz(stdp_dict['dw'], stdp_dict['dt'])
        ax.set_title(f'AUC: {auc:.3f}')
    ax.legend(ncol=2, fontsize=12)
    return stdp_dict


def load_config(pathAndNameOfConfig):
    yaml_txt = open(pathAndNameOfConfig).read()
    return yaml.load(yaml_txt, Loader=yaml.FullLoader)


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


def calcAssWgts(t, m, labels):
    res = dict()
    res['t'] = t
    for i in np.unique(labels):
        for j in np.unique(labels):
            res[f'{i}-{j}'] = m.calcMeanW(
                np.where(labels == i)[0].astype('int32'),
                np.where(labels == j)[0].astype('int32'))
    return res


def calcF(t, m, labels):
    F = dict()
    F['t'] = t
    for i in np.unique(labels):
        F[f'{i}'] = m.calcMeanF(np.where(labels == i)[0].astype('int32'))
    return F


def calcD(t, m, labels):
    D = dict()
    D['t'] = t
    for i in np.unique(labels):
        D[f'{i}'] = m.calcMeanD(np.where(labels == i)[0].astype('int32'))
    return D


def plot_STDP(m):
    # see STDP_explained.ipynb for details
    def fd(x, alpha, JEE):
        return np.log(1 + x * alpha / JEE) / np.log(1 + alpha)

    states = m.getState()
    current_spike_t = 0
    spts = np.linspace(-250, 0, 200)
    dwd = states.Cp * np.exp(-(current_spike_t - spts) / states.tpp) - fd(
        0.1, states.alpha, states.JEE) * states.Cd * np.exp(-(current_spike_t - spts) / states.tpd)
    dwp = states.Cp * np.exp(-(current_spike_t - spts) / states.tpp) - fd(
        0.1, states.alpha, states.JEE) * states.Cd * np.exp(-(current_spike_t - spts) / states.tpd)
    plt.plot(spts - current_spike_t, dwd, label='LTD')
    plt.plot(current_spike_t - spts, dwp, label='LTP')
    plt.grid()
    plt.xlabel('$|t^i - t^j|$ [ms]', fontsize=18)


@numba.jit(fastmath=True)
def getMeanAssWgt(t):
    global m, labels, params, WgtEvo, FDevo, NE, NI
    d = dict()
    fd = dict()
    d['t'] = t
    fd['t'] = t
    W = m.getWeights()[:NE, :NE]
    F = np.copy(m.getF())
    D = np.copy(m.getD())
    for assID in np.unique(labels):
        wacc = 0
        c = 0
        assNids = np.where(labels == assID)[0]
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
    W = m.getWeights()[:NE, :NE]

    waccAB = 0
    cAB = 0
    waccBA = 0
    cBA = 0
    assANids = np.where(labels == assAid)[0]
    assBNids = np.where(labels == assBid)[0]
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
    bin_middle = i + step_ms / 2
    return sp_count, bin_middle


def DetermineUpstateLenghs(cell_ass_id, sub_sorted):

    global tmp, step_ms

    ref_vline_spacing_ms = 50  # for plotting
    plotting = False
    interval_ms_ = 160000

    step_ms = 5  # width of a time bin in which we count spikes (high spike count indicates an upstate)
    trigger_level = 10**0.9  # fine tune this parameter to determine the upstates (see plot below)
    sp_count, bin_middle = [], []  # list of spike counts for each bin, middle times for each bin

    s_ = np.where(labels[newids] == cell_ass_id)[0].min()
    e_ = np.where(labels[newids] == cell_ass_id)[0].max()

    tmp = sub_sorted[(sub_sorted.neuronid > s_) & (sub_sorted.neuronid < e_)]
    tmp = tmp[tmp.spiketime < tmp.spiketime.min() + interval_ms_].reset_index(drop=True)
    tmp.neuronid -= tmp.neuronid.min()

    if plotting:
        fig, ax = plt.subplots(4, 1, figsize=(18, 15), sharex=True)

        ax[0].plot(sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].spiketime,
                   sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].neuronid,
                   'bo',
                   ms=0.2)
        ax[0].axhspan(s_, e_, alpha=0.3, color='red')
        ax[0].set_ylabel(f'Neuron IDs')

        ax[1].plot(tmp.spiketime, tmp.neuronid, 'bo', ms=0.2)
        ax[1].set_title(f'Cell assembly id: {cell_ass_id}')
        for i in np.arange(st,
                           sub_sorted[sub_sorted.spiketime < sub_sorted.spiketime.min() + interval_ms_].spiketime.max(),
                           ref_vline_spacing_ms):
            ax[1].axvline(i, linewidth=0.5)
        ax[1].set_ylabel(f'Neuron IDs')

    pool_list = np.arange(tmp.spiketime.min(), tmp.spiketime.min() + interval_ms_, step_ms)
    #     with Pool(25) as p:
    #         res = list(tqdm(p.imap(par_loop, pool_list), total=len(pool_list))) # with a progress bar
    with Pool(25) as p:
        res = list(p.map(par_loop, pool_list))  # without a progress bar

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
    idx_at_which_to_split = np.where((np.diff(up_state_test) == 1) | (np.diff(up_state_test) == -1))[0] + 1
    # compute the lengths of each upstate in milliseconds
    consec = np.split(up_state_test,
                      idx_at_which_to_split)  # indicators that indicate if the timebin is up or downstate
    upstate_lengts_ms = [i.sum() * step_ms for i in consec if i.sum() > 0]

    #     _ = plt.hist(upstate_lengts_ms, bins=np.linspace(0, 300, 30), density=True)
    #     _ = plt.title(f'Distribution of upstate lenghts in cell assembly {cell_ass_id}\nMean: {np.mean(upstate_lengts_ms):.0f} ms.')
    #     _ = plt.xlabel(f'Upstate duration, ms')
    muFR, us_nogap = get_mean_FR_in_assembly(tmp)
    return np.mean(upstate_lengts_ms), e_ - s_, muFR, cell_ass_id, s_, e_


def get_mean_FR_in_assembly(tmp):

    window = 10  # in ms
    FR_min = 20  # firing rate within a window, below which we drop the data as belonging to a down state
    mingap = 50  # minimum length of downstate that you want to delete
    standard_gap = 0.1  # the length of gap you want to truncate gaps larger than mingap (must be smaller than mingap)

    ss, df_, us_nogap = delimit_upstates(tmp, FR_min, window, mingap, standard_gap)

    FR = []
    for i in range(us_nogap.neuronid.max()):
        FR.append(us_nogap[us_nogap.neuronid == i].shape[0] /
                  ((us_nogap.spiketime.max() - us_nogap.spiketime.min()) / 1000))
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
            ma1[i] = np.sum((x >= (spts[i] - window / 2)) & (x < (spts[i] + window / 2)))
        return ma1

    df_ = tmp.copy()
    spts = df_.spiketime.to_numpy()
    x = df_.spiketime.to_numpy()
    t = time.time()

    ma1 = fast_(spts, x, window)
    df_['drop'] = ma1 < FR_min  # mark down-states for dropping
    #     print(f'{(time.time()-t):2f} s.')

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
            ss_.append((start, stop, stop - start))
    ss = pd.DataFrame(ss_, columns=['start', 'stop', 'leng'])

    # drop the downstates:
    upstates = df_[df_['drop'] == False]
    x = upstates.spiketime.diff().values
    y = [i if i > mingap else 0 for i in x]
    y_ = [i - standard_gap if i > mingap else i for i in y]

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
    plt.figure(figsize=(16, 4))
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
    x = np.copy(w)
    G = nx.from_numpy_matrix(x)
    adj_mat = nx.to_numpy_array(G)
    louvain = Louvain()
    if SKNETWORK_OLD:
        labels = louvain.fit_transform(adj_mat)
    else:
        labels = louvain.fit_predict(adj_mat)
    mod = modularity(adj_mat, labels)

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup: tup[1], reverse=True)
    newids = [i[0] for i in tmp]

    W_ = x
    W_ = W_[newids, :]
    W_ = W_[:, newids]
    return W_, labels, counts, mod, newids


def processSpikes(m, sp, NE, NEo, NI):
    """
    1) cluster E neurons
    2) label neurons according to the clusters the are in
    3) return sorting dictionary to sort the raster
    """
    W_, labels, counts, mod, newids = clusterize(m.getWeights()[:NE, :NE])

    recurrent, inputs, inhibs = [], [], []  # lists of neuron ids

    inp_label = max(labels) + 1  # take the number of clusters and add 1 (this will be the label of inputs)
    inh_label = -1  # inhibitory neurons will be labeled -1
    for i, d in enumerate(labels):
        if i < NE - NEo:
            recurrent.append(d)
        else:
            inputs.append((i, inp_label))
    for i in range(NE, NE + NI):
        inhibs.append((i, inh_label))

    # sort recurrents, so that clusters become apparent
    recurrent = sorted([(i, d) for i, d in enumerate(recurrent)], key=lambda tup: tup[1], reverse=True)
    newidsAndLabels = recurrent + inputs + inhibs  # merge (neuron ids, cluster id) tuples

    newids = [it[0] for it in newidsAndLabels]
    newlabels = [it[1] for it in newidsAndLabels]
    nid2aid = {nid: assid for nid, assid in newidsAndLabels}  # maps neuron ids to cluster id
    sort_dict = {it[0]: i for i, it in enumerate(newidsAndLabels)}  # maps old neuron ids to sorted neuron ids
    sp['aid'] = sp.neuronid.map(nid2aid).astype('int')  # do map
    sp['sorted_neuronid'] = sp.neuronid.map(sort_dict)  # do map
    return sp


def bruteforce_sequences(candidate):
    global tmp
    S = List()
    for nid in candidate:
        S.append(tmp[tmp.neuronid == nid].spiketime.to_numpy())

    c, duplicates, not_duplicates = new_find_seq(S)
    return c, duplicates, not_duplicates, candidate


@njit
def new_find_seq(S: List) -> (list, set, set):
    A = []  # links
    for it, I in enumerate(S[1:]):
        if it == 0:
            J = S[0]
        a = np.zeros((len(I), len(J)))
        for iid, i in enumerate(I):
            for jid, j in enumerate(J):
                a[iid, jid] = i - j
        good_iids, good_jids = np.where((a > 0.0) & (a < 5.0))
        A.append([[j, i] for i, j in zip(I[good_iids], J[good_jids])])
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
    #     mod = modularity(adj_mat, labels)
    mod = 0

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup: tup[1], reverse=True)
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
                X[i, j] = 1.0 / (np.linalg.norm(np.array(SEQ[i]) - np.array(SEQ[j])))
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


def save_states():
    """" saves everything you need to resume a sim from checkpoint """
    global m, cell_id
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


def load_states_from_experiment(m, cell_id, experiment_id=0, stimDur=0, nneur=0, wide=0):
    """" load states from files in exeriment folders """

    # get files from their experiment folders
    width = 'wideU' if wide else 'fixedU'
    _path = f'/home/roman/flash/bmm_big_prtb_flsh_{experiment_id}/experimentGBIG_withFD_{width}_{stimDur}_{nneur}'
    for fname in [
            f'W_bak_{cell_id}.p', f'states_{cell_id}.json', f'SPTS_{cell_id}', f'DSPTS_{cell_id}', f'X_{cell_id}',
            f'UU_case3.npy'
    ]:
        shutil.copyfile(f'{_path}/{fname}', f'{os.getcwd()}/../data/{fname}')

    # load from checkpoints
    with open(f'../data/W_bak_{cell_id}.p', 'rb') as f:
        W_bak = pickle.load(f)
    m.setWeights(W_bak)
    m.loadSpikeStates('anystring')
    with open(f'../data/states_{cell_id}.json', 'rb') as f:
        t = json.loads(f.read())['t']
    m.set_t(t)

    case = 3
    m = set_UU(m, case, wide)
    return m


def set_UU(m, case, wide):
    case = 3
    with open(f'../data/UU_case{case}.npy', 'rb') as f:
        UU = np.load(f)

    if wide:
        m.setUU(np.ascontiguousarray(UU))
    else:
        m.setUU(np.ones_like(UU) * np.mean(UU))
    return m


def simple_make_UU(NE, case, wide):

    N_ = int(8e6)  # create an array with enough margin to delete values sampled above 1.0

    if case == 3:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, False
    elif case == 2:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13.3, 26.6, 1.0, 133, False
    elif case == 1:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, True
    elif case == 0:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13, 26, 1.0, 133, True
    else:
        raise

    if wide:
        x = np.random.gamma(shape_wide, scale_wide, N_) / divisor
    else:
        x = np.random.gamma(shape_narrow, scale_narrow, N_) / divisor
    x = np.delete(x, np.where(x > 1))

    return x[:NE]


def make_UU(m, params, NE, narrow=True, case=None):
    if case == 3:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, False
    elif case == 2:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13.3, 26.6, 1.0, 133, False
    elif case == 1:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 20, 38.9, 1.0, 133, True
    elif case == 0:
        shape_wide, scale_wide, shape_narrow, scale_narrow, divisor, flip = 2, 13, 26, 1.0, 133, True
    else:
        raise

    N_ = int(8e6)  # create an array with enough margin to delete values sampled above 1.0
    try:
        with open(f'/flash/FukaiU/roman/RelProbTestSims/RelProbTest_{case}_1_1/wts_wi_0.5.p', 'rb') as f:
            ww = pickle.load(f)[:NE, :NE]
        i, j = np.where(ww > 0.001)
    except:
        ww = np.copy(m.getWeights())[:NE, :NE]
        i, j = np.where(ww > params['Jepsilon'])

    if not narrow:
        x = np.random.gamma(shape_wide, scale_wide, N_) / divisor
        x = np.delete(x, np.where(x > 1))
        if not flip:
            x = (x[:NE * NE]).reshape(NE, NE)
        else:
            x = -(x[:NE * NE]).reshape(NE, NE) + 1
        UU_wide = np.zeros((NE, NE))
        UU_wide[i, j] = x[i, j]
        return UU_wide, i, j

    else:
        x = np.random.gamma(shape_narrow, scale_narrow, N_) / divisor
        x = np.delete(x, np.where(x > 1))
        if not flip:
            x = (x[:NE * NE]).reshape(NE, NE)
        else:
            x = -(x[:NE * NE]).reshape(NE, NE) + 1
        UU_narrow = np.zeros((NE, NE))
        UU_narrow[i, j] = x[i, j]
        return UU_narrow, i, j


def load_spike_data(W_bak, NE, NI, cell_id, NEo):
    """ this cell loads the spikes into a dataframe and adds adds 
        a column that labels each spike with its assembly id
        And clusters the input and recurrent layers separately
    """

    def Map(x):
        nid = int(x.neuronid)
        if nid < NE:
            return labels[nid]
        else:
            return -1

    # first, cluster recurrent neurons
    stNid, enNid = 0, NE - NEo
    w_, labels, counts, mod, newids = clusterize(W_bak, stNid, enNid)
    map_dict = {j: i for i, j in enumerate(newids)}

    sp = pd.read_csv(f'../data/spike_times_{cell_id}', delimiter=' ', header=None)
    sp.columns = ['spiketime', 'neuronid']

    # st, en = 43.80*1e3, 66.8*1e3
    st, en = sp.spiketime.min(), sp.spiketime.max()
    # st, en = sp.spiketime.max() - 38000, sp.spiketime.max() - 0

    sp = sp[(sp.spiketime > st) & (sp.spiketime < en)]

    sp['aid'] = sp.apply(Map, axis=1)
    sp_srt = sp.copy()
    sp_srt['neuronid'] = sp.neuronid.map(map_dict).fillna(sp.neuronid).astype('int')
    sp.spiketime.max() - sp.spiketime.min()

    sp_srt = sp[sp.neuronid < NE - NEo].copy()
    sp_srt['neuronid'] = sp.neuronid.map(map_dict).fillna(sp.neuronid).astype('int')

    # cluster recurrent
    sp_srt0 = sp[(sp.neuronid < NE - NEo)].copy()
    sp_srt0['neuronid'] -= 0
    sp_srt0['neuronid'] = sp_srt0.neuronid.map(map_dict).fillna(sp_srt0.neuronid).astype('int')
    sp_srt0['neuronid'] += 0

    # cluster input layer
    sp_srt1 = sp[(sp.neuronid >= NE - NEo) & (sp.neuronid < NE)].copy()
    sp_srt1['neuronid'] -= NE - NEo
    sp_srt1['neuronid'] += NE - NEo

    return sp, sp_srt0, sp_srt1


def readLastNms(period_ms, path):
    """"
    read spiketimes backward from the tail of the spike_times file
    """

    def get_line_bwd(f):
        while (f.read(1) != b'\n'):
            f.seek(-2, os.SEEK_CUR)
        line = f.readline().decode()
        f.seek(-len(line) - 2, os.SEEK_CUR)
        return line

    T, NID = [], []

    with open(path, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        line = get_line_bwd(f)
        a = [float(i) for i in line[:-1].split(' ')]
        t_last = a[0]
        t = t_last
        # print(t, t_last)
        while t_last - t < period_ms:
            # print(t, t_last)
            line = get_line_bwd(f)
            a = [float(i) for i in line[:-1].split(' ')]
            t = a[0]
            T.insert(0, a[0]), int(a[1])
            NID.insert(0, int(a[1]))
    return pd.DataFrame({'spiketime': T, 'neuronid': NID})


def removeFilesInFolder(pathToFolder):
    for filename in os.listdir(pathToFolder):
        file_path = os.path.join(pathToFolder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def getExpWghNeurAct(period_ms, path, tau):
    lst = readLastNms(period_ms, path)
    T_MAX = lst.spiketime.max()
    lst['dif'] = lst.spiketime - T_MAX
    lst['weight'] = lst.dif.apply(lambda x: np.exp(tau * x))
    lst
    d = dict()
    for nid in range(400):
        d[nid] = lst[lst.neuronid == nid].weight.sum()
    return d


@njit
def gridToLinear(A, i, j):
    for a in A:
        if ((a[1] == i) & (a[2] == j)):
            return a[0]
    return -1


@njit
def linearToGrid(A, i):
    return A[i][1], A[i][2]


@njit
def getVF(WW):
    """
    Args: WW - weight matrix (400, 400)
    for each neuron, find immediate spatial neighbors (8), get weights to these
    neurons as direction vectors, return the sum of those eight spatial vectors
    Assumes that the weight matrix is 400 x 400
    """
    A = np.arange(400).reshape(20, 20)  # maps neurons on a 2D plane to thier IDs in the network

    # displacements from the center (i, j), (row, column)
    disps = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    disps = [np.array(d) for d in disps]

    # angles to the neighbors
    phis = [np.radians(i) for i in np.arange(0, 360, 45)]

    z = np.zeros((3, 3))  # weights of neighbors
    J = np.zeros((3, 3))  # projecton on j of each weight vector from a neuron to its neighbors
    I = np.zeros((3, 3))  # projecton on i of each weight vector from a neuron to its neighbors
    J0 = np.zeros((3, 3))  # j-th coord of the origins of those projection vectors
    I0 = np.zeros((3, 3))  # i-th coord of the origins of those projection vectors

    X, Y, U, V = np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20))
    Z = np.zeros((20, 20, 3, 3))

    # magnitudes of outgoing weights (not the average weight yet)
    II = np.zeros((20, 20, 3, 3))
    JJ = np.zeros((20, 20, 3, 3))

    # origin coordinates outgoing weights (not the average weight yet)
    XX = np.zeros((20, 20, 3, 3))
    YY = np.zeros((20, 20, 3, 3))
    for i in range(20):
        for j in range(20):
            XX[i, j, :, :] = np.ones((3, 3)) * i  # for each neuron there are 8 neighbors
            YY[i, j, :, :] = np.ones((3, 3)) * j

    for i in range(20):
        for j in range(20):
            z *= 0
            J *= 0
            I *= 0
            J0 *= 0
            I0 *= 0

            fr = np.array([i, j])

            for d, phi in zip(disps, phis):
                a, b = d + 1  # coordinates in a 3x3 matrix (a central neuron and its neighbors (8))
                frd = fr + d  # coordinates in the 20x20 matrix or neurons
                I0[a, b] = i
                J0[a, b] = j
                if not ((np.min(frd) < 0) or (np.max(frd) >= 20)):  # skip if out of the 20x20 grid
                    w = WW[A[frd[0], frd[1]], A[fr[0], fr[1]]]
                    z[a, b] = w
                    I0[a, b] = i
                    J0[a, b] = j
                    I[a, b] = w * np.cos(phi)
                    J[a, b] = w * np.sin(phi)

            Z[i, j, :, :] = z
            II[i, j, :, :] = I
            JJ[i, j, :, :] = J
            X[i, j] = i
            Y[i, j] = j
            U[i, j] = I.sum()
            V[i, j] = J.sum()
    return X, Y, U, V, XX, YY, II, JJ, Z


class SnapshotLoader:

    def __init__(self, ROOT, PATH2EXPDATA, NE, verbose=True):
        self.PATH2EXPDATA = PATH2EXPDATA
        self.fullpath = f"{ROOT}/{PATH2EXPDATA}"
        if verbose:
            print(self.fullpath)
        self.NE = NE

    def load_dat(self, payload):
        idx, fn = payload
        global shared_array, sharr
        # sharr[idx] = idx # NOTE: not used
        ref = shared_array[:]
        d = np.load(f'{self.fullpath}/{fn}', allow_pickle=True).item()
        mWee = d['w'][d['w'] > 0.001].mean()
        trg = d['w'].flatten()
        t = d['spiketime']
        mod = d['mod']
        nass = d['nass']
        cov = np.cov(ref, trg)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        return t, corr, mod, nass, mWee

    def get(self, idInGrid, ref_t=0, num_cores=4, pbar=True):
        global shared_array, sharr

        # get the reference
        sharr = Array(c_int, self.NE)
        sharr[:] = np.zeros((self.NE,)).astype(int)
        shared_array = Array(c_double, int(self.NE**2))
        shared_array[:] = np.load(f'{self.fullpath}/dump_{idInGrid}_{ref_t:010}.npy',
                                  allow_pickle=True).item()['w'].flatten()

        flist = sorted([i for i in os.listdir(f'{self.fullpath}') if i.startswith(f'dump_{idInGrid}_')])  # NOTE the _

        if pbar:
            pbar = tqdm(enumerate(flist), total=len(flist), bar_format=bar_format)
        else:
            pbar = enumerate(flist)

        with Pool(num_cores) as p:
            RES = list(p.imap(self.load_dat, pbar))
        RES = np.vstack(RES)
        T, CORR, MOD, NASS, mWee = np.split(RES, RES.shape[1], axis=1)
        return T, CORR, MOD, NASS, mWee


def get_background(X):
    """ takes the CA1 data and permutes everything """
    X_ = np.zeros_like(X)

    # shuffle ISI, i.e. in the time dimension
    for i in range(X.shape[0]):
        spts = np.where(X[i, :] == 1)[0]
        spts = np.insert(spts, 0, 0)
        isi = np.diff(spts)
        np.random.shuffle(isi)
        spts_ = np.cumsum(isi)
        X_[i, spts_] = 1

    # shuffle in the neuronid dimension
    permidx = np.arange(X_.shape[0])
    np.random.shuffle(permidx)
    X_ = X_[permidx, :]
    return X_


class Stimulator2:

    def __init__(self, m, SEQ, patDurMs=10, nass=6, stim_strength=1.0):
        self.m = m
        self.patterns = dict()
        self.patDurMs = patDurMs
        self.NE = m.getState().NE
        self.ass_sz = self.NE // nass
        self.h = m.getState().h
        self.letters = [l for l in string.ascii_uppercase]
        self.SEQ = SEQ
        self.stim0 = np.zeros((self.NE,), dtype=np.int32)
        self.stimintensity0 = np.zeros((self.NE,), dtype=np.float)

        for i, name in zip(range(0, self.NE, self.ass_sz), self.letters):
            stim = np.copy(self.stim0)
            stimintensity = np.copy(self.stimintensity0)

            stim[i:i + self.ass_sz] = 1
            stimintensity[i:i + self.ass_sz] = stim_strength

            self.patterns[name] = {'stim': stim, 'stimintensity': stimintensity}

    def train(self, cycles=10, clusterize_=True, mexstim=0.01, mexisi=0.04, ISIms=100):
        self.m.set_STDP(True)
        self.m.set_homeostatic(True)

        pbar = trange(cycles, bar_format=bar_format)
        for i in pbar:
            try:
                seq = self.SEQ[i % len(self.SEQ)]
                self.m.set_mex(mexstim)

                for name in seq:
                    self.m.setStim(self.patterns[name]['stim'])
                    self.m.setStimIntensity(self.patterns[name]['stimintensity'])
                    self.m.sim(int(1 / self.h) * self.patDurMs)
                time_now = self.m.getState().t
                self.m.set_mex(mexisi)
                self.m.setStim(np.copy(self.stim0))
                self.m.setStimIntensity(np.copy(self.stimintensity0))
                self.m.sim(int(1 / self.h) * ISIms)  # ISI
                if clusterize_:
                    time_now = self.m.getState().t
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()} clus: {clusterize_}'
                    )
            except KeyboardInterrupt:
                print('User interrupt')
                break

    def sham(self, cycles=100, plasticity=True, clusterize_=True, mex=0.3):
        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)
        self.m.set_mex(mex)
        self.m.setStim(np.copy(self.stim0))
        self.m.setStimIntensity(np.copy(self.stimintensity0))
        pbar = trange(cycles, bar_format=bar_format)
        for i in pbar:
            try:
                for pat in self.patterns:
                    self.m.sim(int(1 / self.h) * self.patDurMs)

                if clusterize_:
                    time_now = self.m.getState().t
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()} clus: {clusterize_}'
                    )
            except KeyboardInterrupt:
                print('User interrupt')
                break


class Stimulator3:
    """
    ARGS:
        m - c++ model
        M - state transition kernel (2-D matrix) that defines P(S_t | S_(t-1)).
    """

    def __init__(self, m, M, patDurMs=10, stim_strength=1.0):
        self.m = m
        self.patterns = dict()
        self.patDurMs = patDurMs
        self.NE = m.getState().NE
        self.ass_sz = self.NE // M.shape[0]
        self.h = m.getState().h
        self.letters = [l for l in string.ascii_uppercase]
        self.M = M
        self.stim0 = np.zeros((self.NE,), dtype=np.int32)
        self.stimintensity0 = np.zeros((self.NE,), dtype=np.float)
        self.prev_letter = 'A'

        for i, name in zip(range(0, self.NE, self.ass_sz), self.letters):
            stim = np.copy(self.stim0)
            stimintensity = np.copy(self.stimintensity0)

            stim[i:i + self.ass_sz] = 1
            stimintensity[i:i + self.ass_sz] = stim_strength

            self.patterns[name] = {'stim': stim, 'stimintensity': stimintensity}

    def next_name(self):
        next_letter = np.random.choice(
            self.letters[:self.M.shape[0]],
            p=self.M[:, self.letters.index(self.prev_letter)],
        )
        self.prev_letter = next_letter
        return next_letter

    def train(self, cycles=10, clusterize_=True, mexstim=0.01, mexisi=0.04, ISIms=100):
        self.m.set_STDP(True)
        self.m.set_homeostatic(True)

        pbar = trange(cycles, bar_format=bar_format)
        for i in pbar:
            try:
                self.m.set_mex(mexstim)

                for j in range(self.M.shape[0]):
                    name = self.next_name()
                    self.m.setStim(self.patterns[name]['stim'])
                    self.m.setStimIntensity(self.patterns[name]['stimintensity'])
                    self.m.sim(int(1 / self.h) * self.patDurMs)
                time_now = self.m.getState().t
                self.m.set_mex(mexisi)
                self.m.setStim(np.copy(self.stim0))
                self.m.setStimIntensity(np.copy(self.stimintensity0))
                self.m.sim(int(1 / self.h) * ISIms)  # ISI
                if clusterize_:
                    time_now = self.m.getState().t
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()} clus: {clusterize_}'
                    )
            except KeyboardInterrupt:
                print('User interrupt')
                break

    def sham(self, cycles=100, plasticity=True, clusterize_=True, mex=0.3):
        self.m.set_STDP(plasticity)
        self.m.set_homeostatic(plasticity)
        self.m.set_mex(mex)
        self.m.setStim(np.copy(self.stim0))
        self.m.setStimIntensity(np.copy(self.stimintensity0))
        pbar = trange(cycles, bar_format=bar_format)
        for i in pbar:
            try:
                for pat in self.patterns:
                    self.m.sim(int(1 / self.h) * self.patDurMs)

                if clusterize_:
                    time_now = self.m.getState().t
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)
                    pbar.set_description(
                        f't: {time_now/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, STDP: {self.m.get_STDP()} clus: {clusterize_}'
                    )
            except KeyboardInterrupt:
                print('User interrupt')
                break
