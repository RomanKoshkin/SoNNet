import os
from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
from termcolor import cprint

cprint(os.getcwd(), color='yellow')

# here we CREATE A CUSTOM C TYPE
# StimMat = c_double * 12 # here you must specify the size
# with a POINTER, you can pass arrays to C, without specifying the size of the array
StimMat = POINTER(c_double)  # https://stackoverflow.com/a/23248168/5623100


class Params(Structure):
    _fields_ = [("alpha", c_double), ("JEI", c_double), ("T", c_double), ("h", c_double), ("cEE", c_double),
                ("cIE", c_double), ("cEI", c_double), ("cII", c_double), ("JEE", c_double), ("JEEinit", c_double),
                ("JIE", c_double), ("JII", c_double), ("JEEh", c_double), ("sigJ", c_double), ("Jtmax", c_double),
                ("Jtmin", c_double), ("hE", c_double), ("hI", c_double), ("IEex", c_double), ("IIex", c_double),
                ("mex", c_double), ("sigex", c_double), ("tmE", c_double), ("tmI", c_double), ("trec", c_double),
                ("Jepsilon", c_double), ("tpp", c_double), ("tpd", c_double), ("twnd", c_double), ("g", c_double),
                ("itauh", c_int), ("hsd", c_double), ("hh", c_double), ("Ip", c_double), ("a", c_double),
                ("xEinit", c_double), ("xIinit", c_double), ("tinit", c_double), ("U", c_double), ("taustf", c_double),
                ("taustd", c_double), ("Cp", c_double), ("Cd", c_double), ("HAGA", c_bool),
                ("symmetric", c_bool)]  # https://stackoverflow.com/a/23248168/5623100]


class retParams(Structure):
    _fields_ = [("alpha", c_double), ("JEI", c_double), ("T", c_double), ("h", c_double), ("NE", c_int), ("NI", c_int),
                ("cEE", c_double), ("cIE", c_double), ("cEI", c_double), ("cII", c_double), ("JEE", c_double),
                ("JEEinit", c_double), ("JIE", c_double), ("JII", c_double), ("JEEh", c_double), ("sigJ", c_double),
                ("Jtmax", c_double), ("Jtmin", c_double), ("hE", c_double), ("hI", c_double), ("IEex", c_double),
                ("IIex", c_double), ("mex", c_double), ("sigex", c_double), ("tmE", c_double), ("tmI", c_double),
                ("trec", c_double), ("Jepsilon", c_double), ("tpp", c_double), ("tpd", c_double), ("twnd", c_double),
                ("g", c_double), ("itauh", c_int), ("hsd", c_double), ("hh", c_double), ("Ip", c_double),
                ("a", c_double), ("xEinit", c_double), ("xIinit", c_double), ("tinit", c_double), ("Jmin", c_double),
                ("Jmax", c_double), ("Cp", c_double), ("Cd", c_double), ("SNE", c_int), ("SNI", c_int), ("NEa", c_int),
                ("t", c_double), ("U", c_double), ("taustf", c_double), ("taustd", c_double), ("HAGA", c_bool),
                ("symmetric", c_bool)]


class cClassOne(object):

    # we have to specify the types of arguments and outputs of each function in the c++ class imported
    # the C types must match.

    def __init__(self, NE=-1, NI=-1, NEo=0, cell_id=-1, home=None):

        N = NE + NI
        self.NE = NE
        self.NEo = NEo
        self.params_c_obj = Params()
        self.ret_params_c_obj = retParams()
        if home is None:
            self.lib = cdll.LoadLibrary('modules/bmm.dylib')
        else:
            self.lib = cdll.LoadLibrary(os.path.join(home, 'bmm.dylib'))

        self.lib.createModel.argtypes = [c_int, c_int, c_int, c_int]  # if the function gets no arguments, use None
        self.lib.createModel.restype = c_void_p  # returns a pointer of type void

        self.lib.sim.argtypes = [c_void_p, c_int]  # takes no args
        self.lib.sim.restype = c_void_p  # returns a void pointer

        self.lib.sim_lif.argtypes = [c_void_p, c_int]  # takes no args
        self.lib.sim_lif.restype = c_void_p  # returns a void pointer

        self.lib.setParams.argtypes = [c_void_p, Structure]  # takes no args
        self.lib.setParams.restype = c_void_p  # returns a void pointer

        self.lib.getState.argtypes = [c_void_p]  # takes no args
        self.lib.getState.restype = retParams  # returns a void pointer

        self.lib.getWeights.argtypes = [c_void_p]  # takes no args
        self.lib.getWeights.restype = ndpointer(dtype=c_double, ndim=2, shape=(N, N))

        self.lib.setWeights.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(N, N))]
        self.lib.setWeights.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.getF.argtypes = [c_void_p]  # takes no args
        self.lib.getF.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        self.lib.getUexc.argtypes = [c_void_p]  # takes no args
        self.lib.getUexc.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE + NI,))

        self.lib.getUinh.argtypes = [c_void_p]  # takes no args
        self.lib.getUinh.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE + NI,))

        self.lib.getD.argtypes = [c_void_p]  # takes no args
        self.lib.getD.restype = ndpointer(dtype=c_double, ndim=1, shape=(NE,))

        self.lib.setF.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        self.lib.setF.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.setD.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        self.lib.setD.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.setStim.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,))]
        self.lib.setStim.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.getState.argtypes = [c_void_p]  # takes no args
        self.lib.getState.restype = retParams  # returns a void pointer

        self.lib.dumpSpikeStates.argtypes = [c_void_p]  # takes no args
        self.lib.dumpSpikeStates.restype = c_void_p  # returns a void pointer

        self.lib.loadSpikeStates.argtypes = [c_void_p, c_char_p
                                            ]  # c_char_p is a zero-terminated pointer to a string of characters
        self.lib.loadSpikeStates.restype = c_void_p  # returns a void pointer

        self.lib.set_t.argtypes = [c_void_p, c_double]  # takes no args
        self.lib.set_t.restype = c_void_p  # returns a void pointer

        self.lib.set_z.argtypes = [c_void_p, c_double]  # takes no args
        self.lib.set_z.restype = c_void_p  # returns a void pointer

        self.lib.set_Ip.argtypes = [c_void_p, c_double]  # takes no args
        self.lib.set_Ip.restype = c_void_p  # returns a void pointer

        self.lib.set_totalInhibW.argtypes = [c_void_p, c_double]  # takes c_double
        self.lib.set_totalInhibW.restype = c_void_p  # returns a void pointer

        self.lib.get_totalInhibW.argtypes = [c_void_p]  # takes no args
        self.lib.get_totalInhibW.restype = c_double  # returns a double pointer

        self.lib.set_inhibition_mode.argtypes = [c_void_p, c_int]  # takes c_double
        self.lib.set_inhibition_mode.restype = c_void_p  # returns a void pointer

        self.lib.get_inhibition_mode.argtypes = [c_void_p]  # takes no args
        self.lib.get_inhibition_mode.restype = c_int  # returns a double pointer

        self.lib.set_STDP.argtypes = [c_void_p, c_bool]  # takes no args
        self.lib.set_STDP.restype = c_void_p  # returns a void pointer

        self.lib.get_STDP.argtypes = [c_void_p]  # takes no args
        self.lib.get_STDP.restype = c_bool  # returns a void pointer

        self.lib.set_useThetas.argtypes = [c_void_p, c_bool]  # takes no args
        self.lib.set_useThetas.restype = c_void_p  # returns a void pointer

        self.lib.get_useThetas.argtypes = [c_void_p]  # takes no args
        self.lib.get_useThetas.restype = c_bool  # returns a void pointer

        self.lib.set_symmetric.argtypes = [c_void_p, c_bool]  # takes no args
        self.lib.set_symmetric.restype = c_void_p  # returns a void pointer

        self.lib.get_symmetric.argtypes = [c_void_p]  # takes no args
        self.lib.get_symmetric.restype = c_bool  # returns a void pointer

        self.lib.set_homeostatic.argtypes = [c_void_p, c_bool]  # takes no args
        self.lib.set_homeostatic.restype = c_void_p  # returns a void pointer

        self.lib.get_homeostatic.argtypes = [c_void_p]  # takes no args
        self.lib.get_homeostatic.restype = c_bool  # returns a void pointer

        self.lib.set_HAGA.argtypes = [c_void_p, c_bool]  # takes no args
        self.lib.set_HAGA.restype = c_void_p  # returns a void pointer

        self.lib.saveSpikes.argtypes = [c_void_p, c_int]  # takes no args
        self.lib.saveSpikes.restype = c_void_p  # returns a void pointer

        self.lib.perturbU.argtypes = [c_void_p]  # takes no args
        self.lib.perturbU.restype = c_void_p  # returns a void pointer

        self.lib.getUU.argtypes = [c_void_p]  # takes no args
        self.lib.getUU.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        self.lib.setUU.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        self.lib.setUU.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.getTheta.argtypes = [c_void_p]  # takes no args
        self.lib.getTheta.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        # self.lib.getExc.argtypes = [c_void_p]  # takes no args
        # self.lib.getExc.restype = ndpointer(dtype=np.float64, ndim=1, shape=(N,))

        self.lib.setStimIntensity.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        self.lib.setStimIntensity.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        self.lib.calcMeanW.argtypes = [
            c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int,
            ndpointer(dtype=c_int, shape=(NE,)), c_int
        ]
        self.lib.calcMeanW.restype = c_double  # returns a double

        self.lib.calcMeanF.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int]
        self.lib.calcMeanF.restype = c_double  # returns a double

        self.lib.calcMeanD.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int]
        self.lib.calcMeanD.restype = c_double  # returns a double

        self.lib.set_mex.argtypes = [c_void_p, c_double]  # takes no args
        self.lib.set_mex.restype = c_void_p  # returns a void pointer

        self.lib.set_hEhI.argtypes = [c_void_p, c_double, c_double]  # takes no args
        self.lib.set_hEhI.restype = c_void_p  # returns a void pointer

        # NOTE: in progress
        self.lib.getRecents.argtypes = [c_void_p]  # takes no args
        self.lib.getRecents.restype = ndpointer(dtype=c_double, shape=(NE,))

        # we call the constructor from the imported libpkg.so module
        self.obj = self.lib.createModel(NE, NI, NEo, cell_id)  # look in teh cpp code. CreateNet returns a pointer

    def setWeights(self, W):
        self.lib.setWeights(self.obj, W)

    # in the Python wrapper, you can name these methods anything you want. Just make sure
    # you call the right C methods (that in turn call the right C++ methods)

    # the order of keys defined in cluster.py IS IMPORTANT for the cClasses not to break down
    def setParams(self, params):
        for key, typ in dict(self.params_c_obj._fields_).items():
            typename = typ.__name__
            # if the current field must be c_int
            if typename == 'c_int':
                setattr(self.params_c_obj, key, c_int(params[key]))
            # if the current field must be c_double
            if typename == 'c_double':
                setattr(self.params_c_obj, key, c_double(params[key]))
            # if the current field must be c_bool
            if typename == 'c_bool':
                setattr(self.params_c_obj, key, c_bool(params[key]))
        self.lib.setParams(self.obj, self.params_c_obj)

    def loadSpikeStates(self, string):
        bstring = bytes(string, 'utf-8')  # you must convert a python string to bytes
        self.lib.loadSpikeStates(self.obj, c_char_p(bstring))

    def getState(self):
        resp = self.lib.getState(self.obj)
        return resp

    def getWeights(self):
        resp = self.lib.getWeights(self.obj)
        return resp

    def getF(self):
        resp = self.lib.getF(self.obj)
        return resp

    def setF(self, F):
        self.lib.setF(self.obj, F)

    def getD(self):
        resp = self.lib.getD(self.obj)
        return resp

    def setD(self, D):
        self.lib.setD(self.obj, D)

    def getUexc(self):
        """ get average exitation over the last 1000 of updates """
        resp = self.lib.getUexc(self.obj)
        return resp

    def getUinh(self):
        """ get average exitation over the last 1000 of updates """
        resp = self.lib.getUinh(self.obj)
        return resp

    def setStim(self, stimVec):
        self.lib.setStim(self.obj, stimVec)

    def sim(self, interval):
        self.lib.sim(self.obj, interval)

    def sim_lif(self, interval):
        self.lib.sim_lif(self.obj, interval)

    def set_z(self, z):
        self.lib.set_z(self.obj, z)

    def dumpSpikeStates(self):
        self.lib.dumpSpikeStates(self.obj)

    def set_t(self, t):
        self.lib.set_t(self.obj, t)

    def set_Ip(self, Ip):
        self.lib.set_Ip(self.obj, Ip)

    def set_totalInhibW(self, totalInhibW):
        self.lib.set_totalInhibW(self.obj, totalInhibW)

    def get_totalInhibW(self):
        return self.lib.get_totalInhibW(self.obj)

    def set_inhibition_mode(self, inhibition_mode):
        self.lib.set_inhibition_mode(self.obj, inhibition_mode)

    def get_inhibition_mode(self):
        return self.lib.get_inhibition_mode(self.obj)

    def set_STDP(self, _STDP):
        self.lib.set_STDP(self.obj, _STDP)

    def get_STDP(self):
        return self.lib.get_STDP(self.obj)

    def set_useThetas(self, _useThetas):
        self.lib.set_useThetas(self.obj, _useThetas)

    def get_useThetas(self):
        return self.lib.get_useThetas(self.obj)

    def set_symmetric(self, _symmetric):
        self.lib.set_symmetric(self.obj, _symmetric)

    def get_symmetric(self):
        return self.lib.get_symmetric(self.obj)

    def set_homeostatic(self, _homeostatic):
        self.lib.set_homeostatic(self.obj, _homeostatic)

    def get_homeostatic(self):
        return self.lib.get_homeostatic(self.obj)

    def saveSpikes(self, saveflag):
        self.lib.saveSpikes(self.obj, saveflag)

    def perturbU(self):
        self.lib.perturbU(self.obj)

    def getUU(self):
        resp = self.lib.getUU(self.obj)
        return resp

    def getTheta(self):
        return self.lib.getTheta(self.obj)

    # def getExc(self):
    #     return self.lib.getTheta(self.obj)

    def setUU(self, UU):
        self.lib.setUU(self.obj, UU)

    def setStimIntensity(self, _StimIntensity):
        self.lib.setStimIntensity(self.obj, _StimIntensity)

    def calcMeanW(self, _iIDs, _jIDs):
        leniIDs, lenjIDs = len(_iIDs), len(_jIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))
        jIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]
        for j in range(lenjIDs):
            jIDs[j] = _jIDs[j]

        return self.lib.calcMeanW(self.obj, iIDs, leniIDs, jIDs, lenjIDs)

    def calcMeanF(self, _iIDs):
        leniIDs = len(_iIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]

        return self.lib.calcMeanF(self.obj, iIDs, leniIDs)

    def calcMeanD(self, _iIDs):
        leniIDs = len(_iIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]

        return self.lib.calcMeanD(self.obj, iIDs, leniIDs)

    def set_mex(self, mex):
        self.lib.set_mex(self.obj, mex)

    def set_hEhI(self, hE, hI):
        self.lib.set_hEhI(self.obj, hE, hI)

    def set_HAGA(self, _HAGA):
        self.lib.set_HAGA(self.obj, _HAGA)

    # NOTE: in progress
    def getRecents(self):
        resp = self.lib.getRecents(self.obj)
        return resp

    # def __setattr__(self, name, value):
    #     typ = [t for n, t in self.params_c_obj._fields_ if n == name][0]
    #     typstr = typ.__name__

    #     if typstr == 'c_int':
    #         self.params_c_obj.__setattr__(name, typ, c_int(value))
    #     # if the current field must be c_double
    #     elif typstr == 'c_double':
    #         self.params_c_obj.__setattr__(name, c_double(value))
    #     # if the current field must be c_bool
    #     elif typstr == 'c_bool':
    #         self.params_c_obj.__setattr__(name, c_bool(value))
    #     else:
    #         raise TypeError(f"Can't cast {name} to {typstr}.")

    #     self.lib.setParams(self.obj, self.params_c_obj)

    # def __getattr__(self, name):
    #     resp = self.lib.getState(self.obj)
    #     return getattr(resp, name)


if __name__ == "__main__":
    m = cClassOne(460, 92, 0, 6347)