from DaTaReachControl import FOverApprox
from DaTaReachControl import GOverApprox
from DaTaReachControl import Interval
import numpy as np

def fFun (x):
    return np.array([x[1], x[0] + x[1]])

def GFun (x):
    return np.array([[np.sin(x[1]), 1.0], [0, np.cos(x[0])]])

lipF = np.array([[1],[np.sqrt(2)]])
lipG = np.array([[1,0],[0,1]])

def test_updateF():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    for i in range(sizeTest):
        xDotVal[:,i] = fFun(xVal[:,i])
    fover = FOverApprox(lipF)
    for i in range(sizeTest):
        fover.update(xVal[:,i:(i+1)], xDotVal[:,i:(i+1)])
    foverbis = FOverApprox(lipF, traj={'x': xVal, 'xDot' : xDotVal,
                                    'u' : np.zeros((1,sizeTest))})
    assert np.array_equal(foverbis.E0x, fover.E0x)  and \
            np.array_equal(foverbis.E0xDot, fover.E0xDot)

def test_partialKnowledgeF():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    for i in range(sizeTest):
        xDotVal[:,i] = fFun(xVal[:,i])
    knownFun = {0 :  {-1 : lambda x : Interval(x[1,0]),
                       0 : lambda x : Interval(0),
                       1 : lambda x : Interval(1)}}
    fover = FOverApprox(lipF, traj={'x': xVal, 'xDot' : xDotVal,
                                    'u' : np.zeros((1,sizeTest))},
                        knownFun=knownFun)
    for i in range(sizeTest):
        resVal = fover(xVal[:,i:(i+1)])
        assert resVal[0,0] == xVal[1,i]
        assert fover.Jf(xVal[:,i:(i+1)])[0,0] == 0
        assert fover.Jf(xVal[:,i:(i+1)])[0,1] == 1
        assert fover.Jf(xVal[:,i:(i+1)])[1,0] == np.sqrt(2)*Interval(-1,1)
        assert fover.Jf(xVal[:,i:(i+1)])[1,1] == np.sqrt(2)*Interval(-1,1)

def test_depSideInfoF():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    for i in range(sizeTest):
        xDotVal[:,i] = fFun(xVal[:,i])
    fover = FOverApprox(lipF, traj={'x': xVal, 'xDot' : xDotVal,
                                    'u' : np.zeros((1,sizeTest))},
                        nDep={0 : np.array([0])}, bf = {0 : Interval(-1,1)})
    for i in range(sizeTest):
        resVal = fover(xVal[:,i:(i+1)])
        assert resVal[0,0] == xVal[1,i]
        assert fover.Jf(xVal[:,i:(i+1)])[0,0] == 0
        assert fover.Jf(xVal[:,i:(i+1)])[0,1] == Interval(-1,1)
    xVal = np.random.random((2,sizeTest))
    for i in range(sizeTest):
        resVal = fover(xVal[:,i:(i+1)])
        trueVal = fFun(xVal[:,i])
        assert resVal[0,0].contains(trueVal[0])
        assert resVal[1,0].contains(trueVal[1])
    xVal = np.random.random((2,sizeTest)) + 1
    for i in range(sizeTest):
        resVal = fover(xVal[:,i:(i+1)])
        assert Interval(-1,1).contains(resVal[0,0])

def test_Gover():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    uSeq = np.random.random((2,sizeTest))
    while True:
        choiceVal = [-1,0,1]
        picVal = [False, False, False]
        for i in range(sizeTest):
            val = np.random.choice(choiceVal, p=[0.3,0.4, 0.3])
            if val == -1:
                uSeq[:,i] = 0
                picVal[0] = True
            else:
                picVal[val+1] = True
                uSeq[:val,i] = 0
                uSeq[(val+1):,i] = 0
            xDotVal[:,i] = fFun(xVal[:,i]) + \
                        np.matmul(GFun(xVal[:,i]), uSeq[:,i:(i+1)]).flatten()
        if picVal[0] and picVal[1] and picVal[2]:
            break
    # Some test
    knownFun = {0 :  {-1 : lambda x : Interval(x[1,0]),
                       0 : lambda x : Interval(0),
                       1 : lambda x : Interval(1)}}
    fover = FOverApprox(lipF, traj={'x': xVal, 'xDot' : xDotVal,
                                    'u' : uSeq}, knownFun=knownFun)
    knownFunG = {(0,1) :  {-1 : lambda x : Interval(1),
                            0 : lambda x : Interval(0),
                            1 : lambda x : Interval(0)},
                 (1,0) :  {-1 : lambda x : Interval(0),
                            0 : lambda x : Interval(0),
                            1 : lambda x : Interval(0)}}
    gover = GOverApprox(lipG, fover, traj={'x': xVal, 'xDot' : xDotVal, 'u' : uSeq},
                        nDep={(0,0) : np.array([0]), (1,1) : np.array([1])},
                        bG = {(0,0) : Interval(-1,1), (1,1) :  Interval(-1,1)},
                        knownFun = knownFunG)
    # Test if over-approximation holds
    xVal = np.random.random((2,sizeTest))
    for i in range(sizeTest):
        resVal = fover(xVal[:,i:(i+1)])
        trueVal = fFun(xVal[:,i])
        assert resVal[0,0].contains(trueVal[0])
        assert resVal[1,0].contains(trueVal[1])
        resValG = gover(xVal[:,i:(i+1)])
        resValGTrue = GFun(xVal[:,i])
        for k in range(resValG.shape[0]):
            for l in range(resValG.shape[1]):
                assert resValG[k,l].contains(resValGTrue[k,l])
    # Test jacobian
    currJg = gover.JG(xVal[:,0:1])
    assert currJg[0,1].all() == 0
    assert currJg[1,0].all() == 0
    assert currJg[0,0,0] == 0
    assert currJg[1,0,1] == 0
