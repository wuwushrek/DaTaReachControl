import numpy as np

from intervalN import *
from overapprox_functionsN import *

from numba import jit
from numba.experimental import jitclass
from numba import types

from numpy import float64 as realN
from numba import float64 as real
from numba import int64 as indType
from numpy import int64 as indTypeN

depTypebf = Dict.empty(key_type=indType, value_type=types.UniTuple(real,2))
depTypebG = Dict.empty(key_type=types.UniTuple(indType,2),
                            value_type=types.UniTuple(real,2))

# Data type we use to save all the information about the over-approximation
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

    ('fixpointWidenCoeff', real),
    ('zeroDiameter', real),
    ('widenZeroInterval', real),
]

@jitclass(spec)
class ReachDyn(object):
    """ Structure class that stores all information that must be used to
        compute the over-approximations
        Note that, a value of 0 for an Unknown Lipschitz constant means
        that the function is fully known while a positive value specifies
        that the component is partially known where side information gives
        the parts that is known
    """
    def __init__(self, Lf, LG, Lfknown=None, LGknown=None, nvDepF=depTypeF,
        nvDepG=depTypeG, bf=depTypebf , bG =depTypebG ,
        bGf = depTypeGradF, bGG=depTypeGradG, xTraj=None,
        xDotTraj = None, uTraj = None, useGronwall=False, verbose=False,
        fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
        widenZeroInterval=1e-3):
        # Save the number of state and control
        self.nS = Lf.shape[0]
        self.nC = LG.shape[1]

        # Save unknown Lipschitz constants
        self.Lf = Lf
        self.LG = LG

        # Save Known Lipschitz constants
        self.Lfknown = np.zeros(Lf.shape)
        self.LGknown = np.zeros(LG.shape)

        # Independent state when calling the function
        self.nvDepF = nvDepF
        self.nvDepG = nvDepG

        # Dependent state when calling the function
        self.vDepF = nvDepF
        self.vDepG = nvDepG

        # Known bounds on f and G
        self.bf = bf
        self.bG = bG

        # Bounds on the gradient of f and G
        self.bGf = bGf
        self.bGG = bGG

        # Lower and Upper bounds on the Jacobian of f and G
        self.Jf_lb = np.zeros((Lf.shape[0],Lf.shape[0]))
        self.Jf_ub = np.zeros((Lf.shape[0],Lf.shape[0]))
        self.JG_lb = np.zeros((LG.shape[0], LG.shape[1], LG.shape[0]))
        self.JG_ub = np.zeros((LG.shape[0], LG.shape[1], LG.shape[0]))


        # Save if gronwall needs to be used or not
        self.useGronwall = useGronwall
        # Verbose --> probably won't work with jitclass mode
        self.verbose = verbose

        # Update the trajectory data
        self.xTraj = np.empty((self.nS, 0), dtype=realN)
        self.fOverTraj_lb = np.empty((self.nS, 0), dtype=realN)
        self.fOverTraj_ub = np.empty((self.nS, 0), dtype=realN)
        self.GOverTraj_lb = np.empty((self.nS, self.nC,0), dtype=realN)
        self.GOverTraj_ub = np.empty((self.nS, self.nC,0), dtype=realN)

        # Coefficient parameter for computing the apriori enclosure
        self.fixpointWidenCoeff = fixpointWidenCoeff
        self.zeroDiameter = zeroDiameter
        self.widenZeroInterval = widenZeroInterval


@jit(nopython=True, parallel=False, fastmath=True)
def initOverApprox(Lf, LG, Lfknown=None, LGknown=None, nvDepF=depTypeF,
        nvDepG=depTypeG, bf=depTypebf , bG =depTypebG , bGf = depTypeGradF,
        bGG=depTypeGradG, xTraj=None, xDotTraj = None, uTraj = None,
        useGronwall=False, verbose=False, fknown=None, Gknown=None,
        fixpointWidenCoeff=0.2, zeroDiameter=1e-5, widenZeroInterval=1e-3):
    """ Initialize an object containing all the side information and Trajectory
    needed to compute the overapproximation of f and G
    """
    overApprox = ReachDyn(Lf, LG, Lfknown, LGknown, nvDepF,
                 nvDepG, bf, bG, bGf, bGG, xTraj, xDotTraj, uTraj,
                 useGronwall, verbose, fixpointWidenCoeff, zeroDiameter,
                 widenZeroInterval)

    # Update the Lipschitz constant of the Known function
    updateKnownLip(overApprox, Lfknown, LGknown)

    # Update the variable dependencies and non dependencies
    updateDecoupling(overApprox, nvDepF, nvDepG, updateJac=False)

    # Update the Lipschitz constants and f and G Jacobian
    updateLip(overApprox, Lf, LG)

    # If the trajectory data is empty -> Nothing to do
    if uTraj is None:
        return

    # If not update the trajectory list based on HC4-Revise
    for i in range(uTraj.shape[1]):
        update(overApprox, xTraj[:,i][:], xDotTraj[:,i][:],uTraj[:,i][:],
                fknown, Gknown)

    return overApprox


@jit(nopython=True, parallel=False, fastmath=True)
def updateKnownLip(overApprox, Lfknown=None, LGknown=None):
    """ Update the Lipschzt constants of the known functions"""
    if Lfknown is None:
        overApprox.Lfknown = np.zeros(overApprox.nS, dtype=realN)
    else:
        overApprox.Lfknown = Lfknown
    if LGknown is None:
        overApprox.LGknown = np.zeros((overApprox.nS, overApprox.nC), dtype=realN)
    else:
        overApprox.LGknown = LGknown


@jit(nopython=True, parallel=False, fastmath=True)
def updateDecoupling(overApprox, nvDepF=depTypeF, nvDepG=depTypeG, updateJac=True):
    """ Compute and store the variable for which f and G depends on.
        TODO: Inefficient approach -> But not important since computed only once
    """
    overApprox.nvDepF = nvDepF
    overApprox.nvDepG = nvDepG
    overApprox.vDepF = {0 : np.empty(1, dtype=indType)}
    overApprox.vDepG = {(0,0) : np.empty(1, dtype= indType)}
    for i in range(overApprox.nS):
        overApprox.vDepF[i] = np.array([k for k in range(overApprox.nS)],
                                        dtype=indType)
        if i in nvDepF:
            arrayIndex = nvDepF[i]
            for k in range(arrayIndex.shape[0]):
                overApprox.vDepF[i] = \
                    overApprox.vDepF[i][overApprox.vDepF[i]-arrayIndex[k] != 0]
        for j in range(overApprox.nC):
            overApprox.vDepG[(i,j)] = \
                np.array([k for k in range(overApprox.nS)], dtype=indType)
            if (i,j) in nvDepG:
                arrayIndex = nvDepG[(i,j)]
                for k in range(arrayIndex.shape[0]):
                    overApprox.vDepG[(i,j)] = \
                        overApprox.vDepG[(i,j)][overApprox.vDepG[(i,j)]-arrayIndex[k]!=0]
    if updateJac:
        updateLip(overApprox, overApprox.Lf, overApprox.LG)

@jit(nopython=True, parallel=False, fastmath=True)
def updateLip(overApprox, Lf, LG):
    """ Store the Lipschitz constant of the unknown Lf and comûte the
        Jacobian.
    """
    overApprox.Lf = Lf
    overApprox.LG = LG
    overApprox.Jf_lb, overApprox.Jf_ub = buildJacF(overApprox.Lf,
                                            overApprox.nvDepF, overApprox.bGf)
    overApprox.JG_lb, overApprox.JG_ub = buildJacG(overApprox.LG,
                                            overApprox.nvDepG, overApprox.bGG)
    if overApprox.verbose:
        print('[f] Unknown fun Lipschitz: ', overApprox.Lf)
        print('[f] Known fun Lipschitz:', overApprox.Lfknown)

        print('[G] Unknown fun Lipschitz: \n', overApprox.LG)
        print('[G] Known fun Lipschitz: \n', overApprox.LGknown)

        print('[Jf] Jacobian unknown f: \n', overApprox.Jf_lb, overApprox.Jf_ub)
        print('[JG] Jacobian unknown G: \n', overApprox.JG_lb, overApprox.JG_ub)

@jit(nopython=True, parallel=False, fastmath=True, nogil=True)
def fover(overApprox, x_lb, x_ub, knownf=None):
    """ COmpute an over approximation of f over the interval [x_lb, x_ub]
        knownf provides the Known part of the unknown function f
    """
    if overApprox.fOverTraj_lb.shape[1] == 0:
        res_lb = np.full(overApprox.nS, -np.inf, dtype=realN)
        res_ub = np.full(overApprox.nS, np.inf, dtype=realN)
        for i in range(overApprox.nS):
            if overApprox.Lf[i] <= 0:
                res_lb[i] = 0
                res_ub[i] = 0
        if knownf is not None:
            fknowx_lb, fknowx_ub = knownf(x_lb, x_ub)
            res_lb += fknowx_lb
            res_ub += fknowx_ub
    else:
        res_lb, res_ub = foverapprox(x_lb, x_ub, overApprox.Lf, overApprox.vDepF,
                            overApprox.xTraj, overApprox.fOverTraj_lb,
                            overApprox.fOverTraj_ub)
        if knownf is not None:
            fknowx_lb, fknowx_ub = knownf(x_lb,x_ub)
            res_lb += fknowx_lb
            res_ub += fknowx_ub
    for i, (vlb, vub) in overApprox.bf.items():
        res_lb[i],res_ub[i] = and_i(res_lb[i], res_ub[i], vlb, vub)
    return res_lb, res_ub

@jit(nopython=True, parallel=False, fastmath=True, nogil=True)
def Gover(overApprox, x_lb, x_ub, knownG=None):
    """ COmpute an over approximation of G over the interval [x_lb, x_ub]
        knownf provides the Known part of the unknown function G
    """
    if overApprox.GOverTraj_lb.shape[2] == 0:
        res_lb = np.full((overApprox.nS,overApprox.nC), -np.inf, dtype=realN)
        res_ub = np.full((overApprox.nS,overApprox.nC), np.inf, dtype=realN)
        for i in range(overApprox.nS):
            for j in range(overApprox.nC):
                if overApprox.LG[i,j] <= 0:
                    res_lb[i,j] = 0
                    res_ub[i,j] = 0
        if knownG is not None:
            Gknowx_lb, Gknowx_ub = knownG(x_lb,x_ub)
            res_lb += Gknowx_lb
            res_ub += Gknowx_ub
    else:
        res_lb, res_ub = Goverapprox(x_lb, x_ub, overApprox.LG, overApprox.vDepG,
                            overApprox.xTraj, overApprox.GOverTraj_lb,
                            overApprox.GOverTraj_ub)
        if knownG is not None:
            Gknowx_lb, Gknowx_ub = knownG(x_lb,x_ub)
            res_lb += Gknowx_lb
            res_ub += Gknowx_ub
    for (i,j), (vlb, vub) in overApprox.bG.items():
        res_lb[i,j],res_ub[i,j] = and_i(res_lb[i,j], res_ub[i,j], vlb, vub)
    return res_lb, res_ub

@jit(nopython=True, parallel=False, fastmath=True)
def update(overApprox, xVal, xDot, uVal, knownf=None, knownG=None):
    """ Update the over-approximation based on the new xVal, xDot, and
        the control u. The update is based on the HC4revise algorithm.
    """
    if knownf is not None:
        xDot = xDot - knownf(xVal, xVal)[0]
    if knownG is not None:
        xDot = xDot - np.dot(knownG(xVal, xVal)[0], uVal)
    foverx = fover(overApprox, xVal, xVal)
    Goverx = Gover(overApprox, xVal, xVal)
    if overApprox.verbose:
        print('uVal :  ', uVal)
        print('xVal : ', xVal)
        print('xDot-Known : ', xDot)
        print('foverx : \n', foverx)
        print('Goverx : \n', Goverx)
    updateTraj(xVal, xDot, uVal, *foverx, *Goverx)
    if overApprox.verbose:
        print('foverx-tight : \n', foverx)
        print('Goverx-tight : \n', Goverx)
    overApprox.xTraj = \
        np.concatenate((overApprox.xTraj,xVal.reshape(-1,1)),axis=1)
    overApprox.fOverTraj_lb = np.concatenate((overApprox.fOverTraj_lb,
                foverx[0].reshape(-1,1)),axis=1)
    overApprox.fOverTraj_ub = np.concatenate((overApprox.fOverTraj_ub,
                foverx[1].reshape(-1,1)),axis=1)
    overApprox.GOverTraj_lb = np.concatenate((overApprox.GOverTraj_lb,
                Goverx[0].reshape(overApprox.nS,overApprox.nC,1)),axis=2)
    overApprox.GOverTraj_ub = np.concatenate((overApprox.GOverTraj_ub,
                Goverx[1].reshape(overApprox.nS,overApprox.nC,1)),axis=2)

@jit(nopython=True, parallel=False, fastmath=True)
def getCoeffGronwall(overApprox, dt, uRange_lb, uRange_ub):
    """ Compute the gronwall coefficient used in the computation of
        the fixpoint enclosure. Si = Ri + getCOeff() abs(h)
    """
    abs_u = abs_i(uRange_lb, uRange_ub)
    vecL = overApprox.Lf + overApprox.Lfknown + \
            np.dot(overApprox.LG+overApprox.LGknown, abs_u)
    betaVal = np.sqrt(np.dot(vecL, vecL))
    return dt / (1-np.sqrt(overApprox.nS)*dt*betaVal)


@jit(nopython=True, parallel=False, fastmath=True)
def fixpoint(overApprox, x_lb, x_ub, dt, uOver_lb, uOver_ub,
             knownf, knownG, hOver=None):
    """ Compute an a priori enclosure, i.e. a loose over-approximation of
        the state for all time between t and  t+dt, that ensures the existence
        of a solution to the unknown dynamical system. The fixpoint
        (solution of the Picard Linderloof operator) using either the recursive
        approach or the Gronwall method developed in the paper
    """
    if hOver is None:
        hVal = add_i(*fover(overApprox, x_lb, x_ub, knownf),
                        *mul_iMv(*Gover(overApprox, x_lb, x_ub, knownG),
                                uOver_lb, uOver_ub)
                    )
    else:
        hVal = hOver

    if overApprox.useGronwall:
        betaVal = getCoeffGronwall(overApprox, dt, uOver_lb, uOver_ub)
        hValAbs = abs_i(*hVal)
        maxVal = np.max(hValAbs) * betaVal
        r_lb, r_ub = x_lb - maxVal, x_ub + maxVal
    else:
        r_lb, r_ub = add_i(x_lb, x_ub, *mul_iv_0c(*hVal, dt))
    while True:
        hVal = add_i(*fover(overApprox, r_lb, r_ub, knownf),
                        *mul_iMv(*Gover(overApprox, r_lb, r_ub, knownG),
                                uOver_lb, uOver_ub)
                    )
        newX_lb, newX_ub = add_i(x_lb, x_ub, *mul_iv_0c(*hVal, dt))
        if overApprox.useGronwall:
            r_lb, r_ub = newX_lb, newX_ub
            break
        isIn = True
        for i in prange(overApprox.nS):
            if not contains_i(newX_lb[i], newX_ub[i], r_lb[i], r_ub[i]):
                widR = r_ub[i] - r_lb[i]
                if widR < overApprox.zeroDiameter:
                    maxR = np.maximum(np.abs(r_lb[i]), np.abs(r_ub[i]))
                    radAdd = overApprox.widenZeroInterval \
                                if maxR <= overApprox.zeroDiameter \
                                else maxR * overApprox.fixpointWidenCoeff
                else:
                    radAdd = widR * overApprox.fixpointWidenCoeff
                r_lb[i] -= radAdd
                r_ub[i] += radAdd
                isIn = False
        if isIn:
            r_lb, r_ub = newX_lb, newX_ub
            break
    return r_lb, r_ub

@jit(nopython=True, parallel=False, fastmath=True)
def DaTaReachN(overApprox, x0_lb, x0_ub, t0, nPoint, dt, uOver, uDer,
                knownf=None, knownG=None, fGradKnown=None, GGradKnown=None):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ----------
    :param x0 : Intial state
    :param t0 : Initial time
    :param nPoint : Number of point
    :param dt : Integration time
    :param uOver : interval extension of the control signal u
    :param uDer : Interval extension of the derivative of the control signal u

    Returns
    -------
    list
        a list of different time at which the over-approximation is computed
    list
        a list of the over-approximation of the state at the above time
    """
    # Save the integration time
    integTime = np.array([t0 + i*dt for i in range(nPoint+1)])
    # Save the tube of over-approximation of the reachable set
    x_lb = np.zeros((x0_lb.shape[0], nPoint+1), realN)
    x_ub = np.zeros((x0_lb.shape[0], nPoint+1), realN)
    # Store the initial point in the trajectory
    x_lb[:,0] = x0_lb[:]
    x_ub[:,0] = x0_ub[:]
    # Constant to not compute everytime
    dt_2 = (0.5*dt**2)
    for i in range(1, nPoint+1):
        # Fetch the previous over-approximation as the current uncertain state
        lastX_lb = x_lb[:, i-1]
        lastX_ub = x_ub[:, i-1]
        # COmpute the control to apply at time t and the control range
        # between t and t + dt --> Ut_lb = Ut_ub in this case
        Ut_lb, Ut_ub = uOver(integTime[i-1], integTime[i-1])
        Ur_lb, Ur_ub = uOver(integTime[i-1], integTime[i])
        # Compute the a priori enclosure
        rEncl_lb, rEncl_ub = fixpoint(overApprox, lastX_lb, lastX_ub, 
                                dt, Ur_lb, Ur_ub, knownf, knownG)
        # Compute the function f(x_t) + G(x_t) u_t
        hx_lb, hx_ub = add_i(
                *fover(overApprox, lastX_lb, lastX_ub, knownf),
                *mul_Ms_i(*Gover(overApprox, lastX_lb, lastX_ub, knownG), Ut_lb)
                            )
        # Compute G at the a priori enclosure rEncl for efficiency
        GEncl = Gover(overApprox, rEncl_lb, rEncl_ub, knownG)
        # COmpite f(rEncl) + G(rEncl)* u_{t, t+dt}
        hr = add_i(
                *fover(overApprox,rEncl_lb, rEncl_ub, knownf),
                *mul_iMv(*GEncl, Ur_lb, Ur_ub)
                )
        # Obtain the approximation of the Jacobian of f
        Jf_lb, Jf_ub = overApprox.Jf_lb, overApprox.Jf_ub
        # Add the Jacobian of knownf if given
        if fGradKnown is not None:
            Jfx_lb, Jfx_ub = fGradKnown(lastX_lb, lastX_ub)
            Jf_lb += Jfx_lb
            Jf_ub += Jfx_ub
        # Obtain the approximation of the Jacobian of G
        JG_lb, JG_ub = overApprox.JG_lb, overApprox.JG_ub
        # Add the Jacobian of knownG if given
        if GGradKnown is not None:
            JGx_lb, JGx_ub = GGradKnown(lastX_lb, lastX_ub)
            JG_lb += JGx_lb
            JG_ub += JGx_ub
        # Compute the second order term (Jf + Jg Ur) * (hr)
        s_lb, s_ub = mul_iMv(
                        *add_i( Jf_lb, Jf_ub ,
                                *mul_iTv(JG_lb, JG_ub, Ur_lb, Ur_ub)
                                ),
                        *hr
                    )
        # Compute the second order term G(rEncl) * \dot{Ur}
        sx_lb, sx_ub = mul_iMv(*GEncl,
                                *uDer(integTime[i-1], integTime[i])
                                )
        x_lb[:,i] = x_lb[:,i-1] + hx_lb * dt + (s_lb + sx_lb) * dt_2
        x_ub[:,i] = x_ub[:,i-1] + hx_ub * dt + (s_ub + sx_ub) * dt_2
    return integTime, x_lb, x_ub
