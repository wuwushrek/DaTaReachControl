import numpy as np
from DaTaReachControl import FOverApprox
from DaTaReachControl import GOverApprox
from DaTaReachControl import Interval

# File variable that is used for hyperparameter
_fixpointWidenCoeff = 0.1
# Detect interval with a small diameter
_zeroDiameter = 1e-5
# coefficient to widen interval with small diameter and close to 0
_widenZeroInterval = 1e-3

def fixpointRecursive(x,  dt, uOver, fOver, GOver,
    fixpointWidenCoeff=_fixpointWidenCoeff, zeroDiameter=_zeroDiameter,
    widenZeroInterval=_widenZeroInterval):
    """Compute an a priori enclosure, i.e. a loose over-approximation of
    the state for all time between t and  t+dt, that ensures the existence
    of a solution to the unknown dynamical system

    Parameters
    ----------
    :param x : A box containing the current state
    :param dt : The sampling time or integration step
    :param uOver : A box giving a range of the control
    :param fOver : Over-approximation of the unknown function f
    :param GOver : over-approximation of the unknown function G
    :param fixpointWidenCoeff : a parameter to speed up the computation of
        the a priori enclosure at the cost of having a loose a priori enclosure
    :param zeroDiameter : Threshold for intervals that are zero
    :param widenZeroInterval : Dilatation parameter for interval close to zero

    Returns
    -------
    An interval representing the a priori enclosure S_(x(x+dt)) in the paper
    """
    hVal =  fOver(x) + np.matmul(GOver(x), uOver)
    r =  x + hVal * Interval(0,dt)
    while True:
        hVal = fOver(r) + np.matmul(GOver(r), uOver)
        newX = x + hVal * Interval(0,dt)
        isIn = True
        for i in range(x.shape[0]):
            if not newX[i,0].contains(r[i,0]):
                widR = r[i,0].diam()
                if widR < zeroDiameter:
                    maxR = np.maximum(np.abs(r[i,0].ub()), np.abs(r[i,0].lb()))
                    radAdd = widenZeroInterval if maxR <= zeroDiameter\
                                else maxR*fixpointWidenCoeff
                else:
                    radAdd = widR * fixpointWidenCoeff
                r[i,0] += Interval(-radAdd,radAdd)
                isIn = False
        if isIn:
            r = newX
            break
    return r


def fixpointGronwall(x, dtCoeff, uOver, fOver, GOver, vectField=None):
    """Compute an a priori enclosure, i.e. a loose over-approximation of
    the state for all time between t and  t+dt, that ensures the existence
    of a solution to the unknown dynamical system. This is done using the
    explicit formula through Gronwall lemma.

    Parameters
    ----------
    :param x : A box containing the current state
    :param dtCoeff : The gronwall coefficient for finding apriori enclosure
    :param uOver : A box giving a range of the control
    :param fOver : Over-approximation of the unknown function f
    :param GOver : over-approximation of the unknown function G

    Returns
    -------
    An interval representing the a priori enclosure S_(x(x+dt)) in the paper
    """
    # (1.0/(1-np.sqrt(x.shape[0])*dt*beta_dt))*dt
    if vectField is None:
        hValAbs =  np.abs(fOver(x) + np.matmul(GOver(x), uOver))
    else:
        hValAbs =  np.abs(vectField)
    maxVal = np.max([hValAbs[i,0].ub for i in range(hValAbs.shape[0])])
    return x + dtCoeff * maxVal * np.full(x.shape, Interval(-1,1))

def DaTaReach(x0, t0, nPoint, dt, fOver, GOver, uOver, uDer, useFast=False):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ----------
    :param x0 : Intial state
    :param t0 : Initial time
    :param nPoint : Number of point
    :param dt : Integration time
    :param fOver : Over-approximation of the unknown function f
    :param GOver : Over-approximation of the unknown function G
    :param uOver : interval extension of the control signal u
    :param uDer : Interval extension of the derivative of the control signal u

    Returns
    -------
    list
        a list of different time at which the over-approximation is computed
    list
        a list of the over-approximation of the state at the above time
    """
    integTime = np.array([t0 + i*dt for i in range(nPoint+1)])
    stateTime = np.full((x0.shape[0], nPoint+1), Interval(0))
    stateTime[:,0] = x0[:,0]
    # np.tensordot(x,b, axes=([1,0]))[:,:,0]
    # lastX = IntervalVector(x0)
    # lastT = t0
    # Jf, Jgkl, J_G = compute_jac(lipF, lipG, depF, depG, sideInfoF, sideInfoG)
    for i in range(1,nPoint+1):
        lastX = stateTime[:,(i-1):i]
        Ut = uOver(Interval(integTime[i-1]))
        Ur = uOver(Interval(integTime[i-1],integTime[i]))
        if not useFast:
            rEncl = fixpointRecursive(lastX, dt, Ur, fOver, GOver)
        else:
            uAbs = np.abs(Ur)
            uSup = np.array([[uAbs[i,0].ub] for i in range(uAbs.shape[0])])
            beta_dt = np.linalg.norm(fOver.Lf + np.matmul(GOver.LG , uSup))
            coeffDt = (dt/(1-np.sqrt(x0.shape[0])*dt*beta_dt))
            rEncl = fixpointGronwall(lastX, coeffDt, Ur, fOver, GOver)
        GEncl = GOver(rEncl)
        fr = fOver(rEncl) + np.matmul(GEncl, Ur)
        fx = fOver(lastX) + np.matmul(GOver(lastX), Ut)
        sTerm = np.matmul((fOver.Jf(lastX) +
                    np.tensordot(GOver.JG(lastX), Ur, axes=([1,0]))[:,:,0]),fr) \
                + np.matmul(GEncl, uDer(Interval(integTime[i-1],integTime[i])))
        stateTime[:,i:(i+1)] = stateTime[:,(i-1):i] + fx * dt + sTerm * (0.5*dt**2)
        # print(integTime[i])
        # print(rEncl)
        # print(Ut, Ur)
        # print(stateTime[:,i])
        # print('----------------')
    return integTime, stateTime
