import numpy as np

from numba import jit, types, prange, typeof
from numba.typed import Dict

from intervalN import *

from numpy import float64 as realN
from numba import float64 as real
from numba import int64 as indType


# Predefined types for function signature
depTypeF = Dict.empty(key_type=indType, value_type=indType[:])
depTypeG = Dict.empty(key_type=types.UniTuple(indType,2), value_type=indType[:])
depTypeGradF = Dict.empty(key_type=types.UniTuple(indType,2),
                          value_type=types.UniTuple(real,2))
depTypeGradG = Dict.empty(key_type=types.UniTuple(indType,3),
                          value_type=types.UniTuple(real,2))

@jit([types.UniTuple(real,2)(real, real, real, real[:], real[:], real[:])],
    nopython=True, parallel=False, fastmath=True)
def hc4Revise(xdot_i, fx_i_lb, fx_i_ub, Gx_i_lb, Gx_i_ub, u):
    """ Given an equation constraint of the type
        xdot_i = fx_i + sum_k (Gx_i)_k u_k where fx_i, (Gx_i)_k are unknown for
        all k, this function returns set of possible values of fx_i an (Gx_i)_k
        satisfying the constraints above. The returned sets are contraction of
        the original sets (fx_i_lb,fx_i_ub) and (Gx_i_lb, Gx_i_ub) that
        over-approximates the unknonw variables fx_i, (Gx_i)_k.
        Assumption: u is nonzero
    """
    # Find the control that are non-zero --> Reduce the complexity
    (indNZu,) = np.nonzero(u)
    # If the size of indNZu is zero then only f can be update
    if indNZu.shape[0] == 0:
        return xdot_i, xdot_i
    # Compute the elementwise product Gx_i u and use it as the nodes of the
    # tree for the HC4revise algorithm
    nGu_lb = np.empty(indNZu.shape[0]+1, dtype=realN)
    nGu_ub = np.empty(indNZu.shape[0]+1, dtype=realN)
    u_red = u[indNZu]
    nGu_lb[:-1], nGu_ub[:-1] = mul_iv_sv(Gx_i_lb[indNZu], Gx_i_ub[indNZu], u_red)
    nGu_lb[-1] = fx_i_lb
    nGu_ub[-1] = fx_i_ub
    # Store the forward and backward interval result of the nodes representing
    # the 2-ary operation (Here addition) -> There's len(indZu) additions
    plusArray_lb = np.empty(indNZu.shape[0], dtype=realN)
    plusArray_ub = np.empty(indNZu.shape[0], dtype=realN)
    # Forward Evaluation of the tree to update the addition node
    plusArray_lb[0], plusArray_ub[0] = add_i(nGu_lb[0], nGu_ub[0], nGu_lb[1], nGu_ub[1])
    for i in range(1, indNZu.shape[0]-1): # The last evaluation doesn't matter -> x_dot
        plusArray_lb[i], plusArray_ub[i] = add_i(plusArray_lb[i-1], plusArray_ub[i-1],\
                                                 nGu_lb[i+1], nGu_ub[i+1])
    # Backward evaluation of the tree to tighten the addition values
    plusArray_lb[-1], plusArray_ub[-1] = xdot_i, xdot_i
    for i in range(indNZu.shape[0]-1, 0, -1):
        lTerm_lb, lTerm_ub = nGu_lb[i+1], nGu_ub[i+1]
        rTerm_lb, rTerm_ub = plusArray_lb[i-1], plusArray_ub[i-1]
        nlTerm_lb, nlTerm_ub = \
            and_i(*sub_i(plusArray_lb[i], plusArray_ub[i], rTerm_lb, rTerm_ub),\
                    lTerm_lb, lTerm_ub)
        nrTerm_lb, nrTerm_ub = \
            and_i(*sub_i(plusArray_lb[i], plusArray_ub[i], nlTerm_lb, nlTerm_ub),\
                    rTerm_lb, rTerm_ub)
        plusArray_lb[i-1], plusArray_ub[i-1] = nrTerm_lb, nrTerm_ub
        nGu_lb[i+1], nGu_ub[i+1] = nlTerm_lb, nlTerm_ub
    # We have to update the last two nodes correctly
    lTerm_lb, lTerm_ub = nGu_lb[1], nGu_ub[1]
    rTerm_lb, rTerm_ub = nGu_lb[0], nGu_ub[0]
    nlTerm_lb, nlTerm_ub = \
        and_i(*sub_i(plusArray_lb[0], plusArray_ub[0], rTerm_lb, rTerm_ub),\
            lTerm_lb, lTerm_ub)
    nrTerm_lb, nrTerm_ub = \
        and_i(*sub_i(plusArray_lb[0], plusArray_ub[0], nlTerm_lb, nlTerm_ub),\
            rTerm_lb, rTerm_ub)
    nGu_lb[0], nGu_ub[0] = nrTerm_lb, nrTerm_ub
    nGu_lb[1], nGu_ub[1] = nlTerm_lb, nlTerm_ub
    # POst processing and return correct value format
    Gx_i_lb[indNZu], Gx_i_ub[indNZu] = mul_iv_sv(nGu_lb[:-1], nGu_ub[:-1], 1/u_red)
    return nGu_lb[-1], nGu_ub[-1]


@jit(types.UniTuple(real,2)(real[:], real[:], real, indType[:], real[:,:],real[:], real[:]),\
     nopython=True, parallel=True, fastmath=True)
def lipOverApprox(x_lb, x_ub, L, varDep, dataState, dataFun_lb, dataFun_ub):
    """ Function to over-approximate a real-valued function at the given
        input x based on the Lipschitz constants of the function, and
        interval-based samples of the values of such function

        Parameters
        ----------
        :param x=(x_lb,x_ub): The point to evaluate the function based on data
        :param L: An upper bound on the LIpschitz constant of that function
        :param varDep: An array specifying the indexes of the variables that the
                       function depends on.
        :param dataState: 2d array providing the historic of the state
        :param dataFun=(dataFun_lb, dataFun_ub): arrays providing an
                        over-approximation of the function at every point of dataState

        Returns
        -------
        An overapproximation of the unknown function at given x
    """
    if L == 0:
        return dataFun_lb[-1], dataFun_ub[-1]
    normValLip_lb = np.empty(dataState.shape[1], dtype=realN)
    normValLip_ub = np.empty(dataState.shape[1], dtype=realN)
    for i in prange(normValLip_lb.shape[0]):
        normValLip_lb[i], normValLip_ub[i] = \
            mul_i_lip(*norm_i(*sub_i(x_lb[varDep], x_ub[varDep], dataState[varDep,i])),
                       L)
    res_lb, res_ub = and_iv(*add_i(dataFun_lb, dataFun_ub, normValLip_lb, normValLip_ub))
    return res_lb, res_ub

@jit([types.UniTuple(real[:,:],2)(real[:], typeof(depTypeF), typeof(depTypeGradF))],
    nopython=True, parallel=True, fastmath=True)
def buildJacF(LipF, nDep, gradBounds):
    """ Compute the enclosure of the Jacobian matrice Jf based only on the
        LIpschitz constant LipF, the non-dependent variables of f, and
        side information such as gradient bounds
    """
    # Upper bound given by the Lipschitz constants
    Jf_init_lb = np.empty((LipF.shape[0], LipF.shape[0]), dtype=realN)
    Jf_init_ub = np.empty((LipF.shape[0], LipF.shape[0]), dtype=realN)
    for i in prange(LipF.shape[0]):
        for j in prange(LipF.shape[0]):
            Jf_init_lb[i,j] = -LipF[i]
            Jf_init_ub[i,j] = LipF[i]
    # Independent variables
    for i, value in nDep.items():
        Jf_init_lb[i, :][value] = 0
        Jf_init_ub[i, :][value] = 0
    # Tighter Jacobian bounds given by side information
    for (i,j), (lb,ub) in gradBounds.items():
        Jf_init_lb[i,j], Jf_init_ub[i,j] = and_i(Jf_init_lb[i,j],
                                                 Jf_init_ub[i,j], lb, ub)
    return Jf_init_lb, Jf_init_ub

@jit([types.UniTuple(real[:,:,:],2)(real[:,:], typeof(depTypeG), typeof(depTypeGradG))],
    nopython=True, parallel=True, fastmath=True)
def buildJacG(LipG, nDep, gradBounds):
    """ Compute the enclosure of the Jacobian matrice JG based only on the
        LIpschitz constant LipG, the non-dependent variables of G, and
        side information such as gradient bounds
    """
    JG_init_lb = np.empty((LipG.shape[0], LipG.shape[1], LipG.shape[0]), dtype=realN)
    JG_init_ub = np.empty((LipG.shape[0], LipG.shape[1], LipG.shape[0]), dtype=realN)
    # Upper bound given by the Lipschitz constants
    for k in prange(LipG.shape[0]):
        for l in prange(LipG.shape[1]):
            for i in prange(LipG.shape[0]):
                JG_init_lb[k,l,i] = -LipG[k,l]
                JG_init_ub[k,l,i] = LipG[k,l]
    # Independent variables
    for (k,l), value in nDep.items():
        JG_init_lb[k,l, :][value] = 0
        JG_init_ub[k,l, :][value] = 0
    # Tighter Jacobian bounds given by side information
    for (k,l,i), (lb,ub) in gradBounds.items():
        JG_init_lb[k,l,i], JG_init_ub[k,l,i] = and_i(JG_init_lb[k,l,i],
                                                     JG_init_ub[k,l,i], lb, ub)
    return JG_init_lb, JG_init_ub


@jit([types.UniTuple(real[:,:],2)(real[:,:], real[:,:], types.misc.Omitted(None)),
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real[:,:]),
      types.UniTuple(real[:,:,:],2)(real[:,:,:], real[:,:,:], types.misc.Omitted(None)),
      types.UniTuple(real[:,:,:],2)(real[:,:,:], real[:,:,:], real[:,:,:])],
    nopython=True, parallel=False, fastmath=True)
def updateJac(encJac_lb, encJac_ub, knownJacx=None):
    """ Given Assuming the unknown function f = f_known + f_unknown,
    encJac = (encJac_lb,encJac_ub) provides an over-approximation of the
    Jacobian of f_unknown while knownJac provides the exact Jacobian of
    f_known. The resulting Jacobian of f is the adddition of the two
    """
    if knownJacx is None:
        return encJac_lb, encJac_ub
    # Compute the known jac
    return encJac_lb+knownJacx, encJac_ub+knownJacx

@jit([types.UniTuple(real[:],2)(real[:], real[:], real[:], typeof(depTypeF),
                     real[:,:], real[:,:], real[:,:], types.misc.Omitted(None),
                     types.misc.Omitted(None)),
      types.UniTuple(real[:],2)(real[:], real[:], real[:], typeof(depTypeF),
                     real[:,:], real[:,:], real[:,:], real[:], real[:])],
    nopython=True, parallel=True, fastmath=True)
def foverapprox(x_lb, x_ub, LipF, varDep, dataState, dataFun_lb, dataFun_ub,
                knownFx_lb=None, knownFx_ub=None):
    """Compute the over-approximation of f at the given interval x=(x_lb, x_ub)
       given a trajectory x_traj = dataState and dataFun=(dataFun_lb, dataFun_ub)
       over-approximate f at each point x inside dataState
    """
    res_lb = np.empty(LipF.shape[0], dtype=realN)
    res_ub = np.empty(LipF.shape[0], dtype=realN)
    for k in prange(x_lb.shape[0]):
        res_lb[k], res_ub[k] = lipOverApprox(x_lb, x_ub, LipF[k], varDep[k],
                                dataState, dataFun_lb[k,:], dataFun_ub[k,:])
    if knownFx_lb is None or knownFx_ub is None:
        return res_lb, res_ub
    else:
        return  res_lb+knownFx_lb, res_ub+knownFx_ub

@jit([types.UniTuple(real[:,:],2)(real[:], real[:], real[:,:], typeof(depTypeG),
                     real[:,:], real[:,:,:], real[:,:,:], types.misc.Omitted(None),
                     types.misc.Omitted(None)),
      types.UniTuple(real[:,:],2)(real[:], real[:], real[:,:], typeof(depTypeG),
                     real[:,:], real[:,:,:], real[:,:,:], real[:,:], real[:,:])],
    nopython=True, parallel=True, fastmath=True)
def Goverapprox(x_lb, x_ub, LipG, varDep, dataState, dataFun_lb, dataFun_ub,
                knownGx_lb=None, knownGx_ub=None):
    """Compute the over-approximation of G at the given interval x=(x_lb, x_ub)
       given a trajectory x_traj = dataState and dataFun=(dataFun_lb, dataFun_ub)
       over-approximate G at each point x inside dataState
    """
    res_lb = np.empty(LipG.shape, dtype=realN)
    res_ub = np.empty(LipG.shape, dtype=realN)
    for k in prange(res_ub.shape[0]):
        for l in prange(res_ub.shape[1]):
            res_lb[k,l], res_ub[k,l] = lipOverApprox(x_lb, x_ub, LipG[k,l],
                                        varDep[(k,l)], dataState,
                                        dataFun_lb[k,l,:], dataFun_ub[k,l,:])
    if knownGx_lb is None or knownGx_ub is None:
        return res_lb, res_ub
    else:
        return res_lb+knownGx_lb, res_ub+knownGx_ub

@jit([types.void(real[:], real[:], real[:], real[:], real[:], real[:,:], real[:,:])],
    nopython=True, parallel=True, fastmath=True)
def updateTraj(x, xdot, u, fxR_lb, fxR_ub, GxR_lb, GxR_ub):
    """ Given new data point, update your knowledge of f and G """
    for i in prange(x.shape[0]):
        f_lb, f_ub = hc4Revise(xdot[i], fxR_lb[i], fxR_ub[i],
                                    GxR_lb[i,:], GxR_ub[i,:], u)
        fxR_lb[i] = f_lb
        fxR_ub[i] = f_ub
