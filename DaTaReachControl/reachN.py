import numpy as np

from intervalN import *
from overapprox_functionsN import *

from numba.experimental import jitclass
from numba import types

from numpy import float64 as realN
from numba import float64 as real
from numba import int64 as indType

depTypebf = Dict.empty(key_type=indType, value_type=types.UniTuple(real,2))
depTypebG = Dict.empty(key_type=types.UniTuple(indType,2),
                            value_type=types.UniTuple(real,2))

spec = [
    ('nS', indType),
    ('nC', indType),
    ('useGronwall', types.boolean),
    ('verbose', types.boolean),

    ('Lf', real[:]),
    ('Lfknown', real[:]),
    ('nvDepF', typeof(depTypeF)),
    ('vDepF', typeof(depTypeF)),

    ('bf', typeof(depTypebf)),
    ('bGf', typeof(depTypeGradF)),

    ('LG', real[:,:]),
    ('LGknown', real[:,:]),
    ('nvDepG', typeof(depTypeG)),
    ('vDepG', typeof(depTypeG)),

    ('bG', typeof(depTypebG)),
    ('bGG', typeof(depTypeGradG)),

    ('Jf_lb', real[:,:]),
    ('Jf_ub', real[:,:]),
    ('JG_lb', real[:,:,:]),
    ('JG_ub', real[:,:,:]),

    ('xTraj', real[:,:]),
    ('fOverTraj_lb', real[:,:]),
    ('fOverTraj_ub', real[:,:]),
    ('GOverTraj_lb', real[:,:,:]),
    ('GOverTraj_ub', real[:,:,:]),
]

@jitclass(spec)
class ReachDyn(object):
    def __init__(self, Lf, LG, Lfknown=None, LGknown=None, nvDepF=depTypeF,
                 nvDepG=depTypeG, bf=depTypebf , bG =depTypebG , bGf = depTypeGradF,
                 bGG=depTypeGradG, xTraj=None, xDotTraj = None, uTraj = None,
                 useGronwall=False, verbose=False, fknown=None, Gknown=None):
        # Save the number of state and control
        self.nS = Lf.shape[0]
        self.nC = LG.shape[1]
        # Save if gronwall needs to be used or not
        self.useGronwall = useGronwall
        # Verbose --> probably won't work with jitclass mode
        self.verbose = verbose
        # Update the Lipschitz constant of the Known function
        self.updateKnownLip(Lfknown, LGknown)
        # Update the variable dependencies and non dependencies
        self.updateDecoupling(nvDepF, nvDepG, updateJac=False)
        self.bGf = bGf
        self.bf = bf
        self.bGG = bGG
        self.bG = bG
        # Update the Lipschitz constants and f and G Jacobian
        self.updateLip(Lf, LG)
        # Update the trajectory data
        self.xTraj = np.empty((self.nS, 0), dtype=realN)
        self.fOverTraj_lb = np.empty((self.nS, 0), dtype=realN)
        self.fOverTraj_ub = np.empty((self.nS, 0), dtype=realN)
        self.GOverTraj_lb = np.empty((self.nS, self.nC,0), dtype=realN)
        self.GOverTraj_ub = np.empty((self.nS, self.nC,0), dtype=realN)
        if uTraj is None:
            return
        for i in range(uTraj.shape[1]):
            self.update(xTraj[:,i][:], xDotTraj[:,i][:],uTraj[:,i][:], fknown, Gknown)

    def updateKnownLip(self, Lfknown=None, LGknown=None):
        """ Update the Lipschzt constants of the known functions"""
        if Lfknown is None:
            self.Lfknown = np.zeros(self.nS, dtype=realN)
        else:
            self.Lfknown = Lfknown
        if LGknown is None:
            self.LGknown = np.zeros((self.nS, self.nC), dtype=realN)
        else:
            self.LGknown = LGknown

    def updateDecoupling(self, nvDepF=depTypeF, nvDepG=depTypeG, updateJac=True):
        """ COmpute and store the variable for which f and G depends on.
            TODO: Inefficient approach -> But not important since computed only once
        """
        self.nvDepF = nvDepF
        self.nvDepG = nvDepG
        self.vDepF = {0 : np.empty(1, dtype=indType)}
        self.vDepG = {(0,0) : np.empty(1, dtype= indType)}
        for i in range(self.nS):
            self.vDepF[i] = np.array([k for k in range(self.nS)], dtype=indType)
            if i in nvDepF:
                arrayIndex = nvDepF[i]
                for k in range(arrayIndex.shape[0]):
                    self.vDepF[i] = self.vDepF[i][self.vDepF[i]-arrayIndex[k] != 0]
            for j in range(self.nC):
                self.vDepG[(i,j)] = np.array([k for k in range(self.nS)], dtype=indType)
                if (i,j) in nvDepG:
                    arrayIndex = nvDepG[(i,j)]
                    for k in range(arrayIndex.shape[0]):
                        self.vDepG[(i,j)] = self.vDepG[(i,j)][self.vDepG[(i,j)]-arrayIndex[k]!=0]
        if updateJac:
            self.updateLip(self.Lf, self.LG)

    def updateLip(self, Lf, LG):
        """ Store the Lipschitz constant of the unknown Lf and comûte the
            Jacobian.
        """
        self.Lf = Lf
        self.LG = LG
        self.Jf_lb, self.Jf_ub = buildJacF(self.Lf, self.nvDepF, self.bGf)
        self.JG_lb, self.JG_ub = buildJacG(self.LG, self.nvDepG, self.bGG)
        if self.verbose:
            print('[f] Unknown fun Lipschitz: ', self.Lf)
            print('[f] Known fun Lipschitz:', self.Lfknown)

            print('[G] Unknown fun Lipschitz: \n', self.LG)
            print('[G] Known fun Lipschitz: \n', self.LGknown)

            print('[Jf] Jacobian unknown f: \n', self.Jf_lb, self.Jf_ub)
            print('[JG] Jacobian unknown G: \n', self.JG_lb, self.JG_ub)

    def fover(self, x_lb, x_ub, knownf=None):
        """ COmpute an over approximation of f over the interval [x_lb, x_ub]
            knownf provides the Known part of the unknown function f
        """
        if self.fOverTraj_lb.shape[1] == 0:
            res_lb = np.full(self.nS, -np.inf, dtype=realN)
            res_ub = np.full(self.nS, np.inf, dtype=realN)
        else:
            fknowx_lb, fknowx_ub = None, None if knownf is None else knownf(x_lb,x_ub)
            res_lb, res_ub = foverapprox(x_lb, x_ub, self.Lf, self.vDepF,
                                self.xTraj, self.fOverTraj_lb, self.fOverTraj_ub,
                                fknowx_lb, fknowx_ub)
        for i, (vlb, vub) in self.bf.items():
            res_lb[i],res_ub[i] = and_i(res_lb[i], res_ub[i], vlb, vub)
        return res_lb, res_ub

    def Gover(self, x_lb, x_ub, knownG=None):
        """ COmpute an over approximation of G over the interval [x_lb, x_ub]
            knownf provides the Known part of the unknown function G
        """
        if self.GOverTraj_lb.shape[2] == 0:
            res_lb = np.full((self.nS,self.nC), -np.inf, dtype=realN)
            res_ub = np.full((self.nS,self.nC), np.inf, dtype=realN)
        else:
            Gknowx_lb, Gknowx_ub = None, None if knownG is None else knownf(x_lb,x_ub)
            res_lb, res_ub = Goverapprox(x_lb, x_ub, self.LG, self.vDepG,
                                self.xTraj, self.GOverTraj_lb, self.GOverTraj_ub,
                                Gknowx_lb, Gknowx_ub)
        for (i,j), (vlb, vub) in self.bG.items():
            res_lb[i,j],res_ub[i,j] = and_i(res_lb[i,j], res_ub[i,j], vlb, vub)
        return res_lb, res_ub

    def update(self, xVal, xDot, uVal, knownf=None, knownG=None):
        if knownf is not None:
            xDot = xDot - knownf(xVal, xVal)[0]
        if knownG is not None:
            xDot = xDot - np.matmul(knownG(xVal, xVal)[0], uVal)
        foverx = self.fover(xVal, xVal)
        Goverx = self.Gover(xVal, xVal)
        if self.verbose:
            print('xVal : ', xVal)
            print('xDot-Known : ', xDot)
            print('foverx : \n', foverx)
            print('Goverx : \n', Goverx)
        updateTraj(xVal, xDot, uVal, *foverx, *Goverx)
        if self.verbose:
            print('foverx-tight : \n', foverx)
            print('Goverx-tight : \n', Goverx)
