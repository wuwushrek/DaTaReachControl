import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as spo

def synthTraj(fFun, GFun, uFun, xInit, evalTime, atol=1e-10, rtol=1e-10):
    """Compute a solution of the ODE fFun(x) + Gfun(x) u(t) with the initial
    state given by xInit, and evalTime contains the different time at which the
    state and its derivative (xDotVal) should be returned.
    For this function xInit, fFun, and GFun takes as input (n,) array
    and returns (n,) array. uFun return a (n,1) array.
    """
    nState = xInit.shape[0]
    t0 = evalTime[0] # Init time
    tend = evalTime[-1] # End time
    def dynFun (t , x):
        return fFun(x) + np.matmul(GFun(x), uFun(t))[:,0]
    # Numerical solution of the ODE
    solODE = solve_ivp(dynFun, t_span=(t0,tend), y0=xInit,
                        t_eval=evalTime, atol=atol, rtol=rtol)
    xDotVal = np.zeros((nState, len(evalTime)))
    # Compute xDot
    for i in range(len(evalTime)):
        xDotVal[:,i] = dynFun(solODE.t[i], solODE.y[:,i])
    return solODE.t, solODE.y, xDotVal


def synthNextState(fFun, GFun, samplingTime=0.1, atol=1e-10, rtol=1e-10):
    """Compute the value of the state at time t+samplingTime of the dynamics
    fFun(x) + Gfun(x) uVal given the current state xVal and the constant
    control uVal. For this function x, currX , fFun, and GFun takes as input (n,)
    array and returns (n,) array. uFun return a (n,1) array.

    Returns
    -------
    a function taking as input the current state and returning the state
    at time t+samplingTime and the derivatives between t and t+samplingTime
    """
    def nextState(currX, currU, dt=samplingTime):
        def dynFun (t, x):
            return fFun(x) + np.matmul(GFun(x), currU)[:,0]
        solODE = solve_ivp(dynFun, t_span=(0,dt), y0=currX, \
                                atol=atol, rtol=rtol)
        return solODE.y[:,-1], dynFun(solODE.t[0],currX)
    return nextState

def generateTraj(fFun, GFun, xInit, uSeq, dt, nbPoint=1):
    """
    Generate a trajectory/measurements based on the control sequences
    uSeq. xInit is the initial state and dt is the sampling time when
    nbPoint is 1 and dt/nbPoint gives the general sampling time.
    uSeq[:,i] is applied every dt.
    The last nbPoint points are obtained by applying control value
    uSeq[:,-1]
    """
    newDt = dt / nbPoint
    nextState = synthNextState(fFun, GFun, newDt)
    currX = np.zeros((xInit.shape[0], nbPoint*uSeq.shape[1]+1))
    currX[:,0] = xInit
    currXdot = np.zeros((xInit.shape[0], nbPoint* uSeq.shape[1]))
    currInd = 0
    for i in range(uSeq.shape[1]):
        for j in range(nbPoint):
            nX , cXdot = nextState(currX[:,currInd], uSeq[:,i:(i+1)])
            currXdot[:,currInd] = cXdot
            currX[:,currInd+1] = nX
            currInd += 1
    return currX[:,:-1], currXdot, currX[:,-1]
