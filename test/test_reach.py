from DaTaReachControl import *
import numpy as np

def fFun (x):
    return np.array([x[1], 0])
lipF = np.array([[1], [0]])
nDepF = {0 : np.array([0])}
knownFunF = {1 : {-1 : lambda x : Interval(0),
                   0 : lambda x : Interval(0),
                   1 : lambda x : Interval(0)}}

def GFun (x):
    return np.array([[0],[x[0]]])
lipG = np.array([[0], [1]])
nDepG = {(1,0) : np.array([1])}
knownFunG = {(0,0) : {-1 : lambda x : Interval(0),
                   0 : lambda x : Interval(0),
                   1 : lambda x : Interval(0)}}

xInit = np.array([1.0,0])
uSeq = np.array([[0, 1, 1, 0, 1, 1, 0, 1]])
dt = 0.1

xTraj, xDotTraj, xNext = generateTraj(fFun, GFun, xInit, uSeq, dt, 1)
traj = {'x' : xTraj, 'xDot' : xDotTraj, 'u' : uSeq}
xTrajPrec, xDotTrajPrec, _ = generateTraj(fFun, GFun, xInit, uSeq, dt, 10)

def test_AprioriEnclosure():
    fover = FOverApprox(lipF, traj=traj, knownFun=knownFunF, nDep=nDepF)
    gover = GOverApprox(lipG, fover, traj=traj, knownFun=knownFunG, nDep=nDepG)
    currInd = np.random.randint(low=0, high=xTrajPrec.shape[1])
    xVal  = xTrajPrec[:,currInd]
    xValInt = np.array([[Interval(xVal[i])] for i in range(xVal.shape[0])])
    dt = 0.1
    uVal = 1
    uOver = np.array([[Interval(uVal)]])
    # Recursive over-approximation
    r = fixpointRecursive(xValInt, dt, uOver, fover, gover)
    # Gronwall over-approximation
    uAbs = np.abs(uOver)
    uSup = np.array([[uAbs[i,0].ub] for i in range(uAbs.shape[0])])
    beta_dt = np.linalg.norm(fover.Lf + np.matmul(gover.LG , uSup))
    coeffDt = (dt/(1-np.sqrt(xVal.shape[0])*dt*beta_dt))
    rEncl = fixpointGronwall(xValInt, coeffDt, uOver, fover, gover)
    xDt, xDotdt, _ = generateTraj(fFun, GFun, xVal, np.array([[uVal]]), dt, 100)
    for i in range(xDt.shape[1]):
        assert r[0,0].contains(xDt[0,i])
        assert r[1,0].contains(xDt[1,i])
        assert rEncl[0,0].contains(xDt[0,i])
        assert rEncl[1,0].contains(xDt[1,i])

def test_OverApprox():
    fover = FOverApprox(lipF, traj=traj, knownFun=knownFunF, nDep=nDepF)
    gover = GOverApprox(lipG, fover, traj=traj, knownFun=knownFunG, nDep=nDepG)
    for i in range(xTraj.shape[1]-1):
        currX = xTraj[:,i]
        currXInt = np.array([[Interval(currX[i])] for i in range(currX.shape[0])])
        nPoint = 10
        def uOver (intT):
            return np.array([[Interval(uSeq[j,i])] for j in range(uSeq.shape[0])])
        def uDer (intT):
            return np.array([[Interval(0)] for j in range(uSeq.shape[0])])
        listT1, overX = overApproxTube(currXInt, 0, nPoint, dt/nPoint, fover,
                            gover, uOver, uDer, useFast=False)
        listT1, overXf = overApproxTube(currXInt, 0, nPoint, dt/nPoint, fover,
                            gover, uOver, uDer, useFast=True)
        trueX , trueXdot, _ = generateTraj(fFun, GFun, currX, \
                                            uSeq[:,i:(i+1)], dt, nPoint)
        for i in range(trueX.shape[0]):
            for j in range(overX.shape[0]):
                assert overX[j,i].contains(trueX[j,i])
                assert overXf[j,i].contains(trueX[j,i])
                assert overXf[j,i+1].contains(overX[j,i+1])
