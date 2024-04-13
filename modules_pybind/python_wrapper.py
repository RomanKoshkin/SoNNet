import sys
from typing import Dict
import numpy as np
from termcolor import cprint

sys.path.append('../')
from .cpp_modules import Model as cpp_core
from modules.utils import AttributeDict


class Model:
    """"
    A convenience wrapper around the cpp shared module (with a pybind11 interface)
    Of course you can access the methods and properties in the c++ code directly 
    (just expose them in main.cpp), but for backwards compatibility with cClasses_10b.py
    and the many utilities already created, it's better to use this wrapper.
    NOTE: Before I used c++ code with assessed through a C interface (extern C), which 
    required the definition of c-functions, c-types, pointers, input and return structs,
    maintaining the python wrapper (with its own type definitions and wrappers).
    """

    def __init__(self, NE, NI, NEo, cell_id):
        self.m = cpp_core(NE, NI, NEo, cell_id)

    def saveSpikes(self, flag: bool):
        self.m.saveflag = flag

    def setWeights(self, w):
        self.m.Jo = list(w)

    def getWeights(self):
        return np.stack(self.m.Jo)

    def setFrozens(self, frozens):
        self.m.frozens = list(frozens)

    def getFrozens(self):
        return np.stack(self.m.frozens)

    def getState(self):
        return AttributeDict(self.m.getState())

    def setParams(self, params: dict, reset_connection_probs=False):
        self.m.setParams(params, reset_connection_probs)

    def setStim(self, stim):
        self.m.hStim = list(stim)

    def setStimIntensity(self, intensities):
        self.m.stimIntensity = list(intensities)

    def setUU(self, UU):
        self.m.UU = list(np.ascontiguousarray(UU))

    def getUU(self):
        return np.ascontiguousarray(self.m.UU)

    def setTheta(self, theta):
        self.m.theta = list(np.ascontiguousarray(theta))

    def getTheta(self):
        return np.ascontiguousarray(self.m.theta)

    def setF(self, F: np.array):
        self.m.F = list(F)

    def getF(self):
        return np.array(self.m.F)

    def setD(self, D: np.array):
        self.m.D = list(D)

    def getD(self):
        return np.array(self.m.D)

    def sim(self, steps: int):
        self.m.sim(steps)

    def sim_lif(self, steps: int):
        self.m.sim_lif(steps)

    def set_useThetas(self, flag: bool):
        self.m.use_thetas = flag

    def get_useThetas(self):
        return self.m.use_thetas

    def dumpSpikeStates(self):
        self.m.saveDSPTS()
        self.m.saveX()
        self.m.saveSpts()

    def loadSpikeStates(self, fname):
        self.m.loadDSPTS(fname)
        self.m.loadX(fname)
        self.m.loadSpts(fname)

    def set_mex(self, mex: float):
        self.m.mex = mex

    def get_mex(self):
        return self.m.mex

    def set_Jmin(self, Jmin: float):
        self.m.Jmin = Jmin

    def get_Jmin(self):
        return self.m.Jmin

    def set_Jmax(self, Jmax: float):
        self.m.Jmax = Jmax

    def get_Jmax(self):
        return self.m.Jmax

    def set_STDP(self, flag: bool):
        self.m.STDPon = flag

    def get_STDP(self):
        return self.m.STDPon

    def get_Uinh(self):
        return np.stack(self.m.Uinh).mean(axis=1)

    def get_Uexc(self):
        return np.stack(self.m.Uexc).mean(axis=1)

    def set_homeostatic(self, flag: bool):
        self.m.homeostatic = flag

    def set_symmetric(self, flag: bool):
        self.m.symmetric = flag

    def get_FR(self):
        return np.array(self.m.getFR())

    def save_state(self, name, path="data/states"):
        state_dict = dict()
        state_dict['x'] = self.m.x
        state_dict['dspts'] = self.m.dspts
        state_dict['spts'] = self.m.spts
        state_dict['Jo'] = self.m.Jo
        state_dict['UU'] = self.getUU()
        state_dict['F'] = self.m.F
        state_dict['D'] = self.m.D
        state_dict['t'] = self.m.getState()['t']
        state_dict['stp_on_I'] = self.m.stp_on_I
        print(state_dict['t'])
        state_dict['params'] = dict(self.m.getState())
        state_dict['frozens'] = self.getFrozens()
        np.save(f'{path}/{name}', state_dict)

    @classmethod
    def from_state(cls, name, path="data/states"):
        state_dict = np.load(f'{path}/{name}.npy', allow_pickle=True).item()

        NE = state_dict['params']['NE']
        NI = state_dict['params']['NI']
        NEo = 0  # FIXME: state_dict['params']['NEo']

        m = cls(NE, NI, NEo, 0)
        m.m.stp_on_I = state_dict['stp_on_I']
        m.m.t = state_dict['t']
        m.setParams(state_dict['params'])
        # NOTE: remove this in the future. I fixed run_cell_Fig5a.py. Just do m.setUU(state_dict['params'])
        UU = np.ones((NE + NI,))  # NOTE: remove this
        UU[:NE] = state_dict['UU']  # NOTE: remove this
        cprint("Using a hotfix on UU. REMOVE IT WHEN YOU RE-RUN FIG5A", color='red', attrs=['bold'])
        m.setUU(UU)
        m.setFrozens(state_dict['frozens'])
        m.m.x = state_dict['x']
        m.m.dspts = state_dict['dspts']
        m.m.spts = state_dict['spts']
        m.m.F = state_dict['F']
        m.m.D = state_dict['D']
        m.m.Jo = state_dict['Jo']
        return m
