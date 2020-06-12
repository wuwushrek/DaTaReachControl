from DaTaReachControl import FOverApprox
from DaTaReachControl import Interval
import numpy as np

def fFun (x):
    return np.array([x[1], x[0] + x[1]])

lipF = np.array([[1],[np.sqrt(2)]])

def test_updateF():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    for i in range(sizeTest):
        xDotVal[:,i] = fFun(xVal[:,i])
    fover = FOverApprox(lipF)
    for i in range(sizeTest):
        fover.update(xVal[:,i:(i+1)], xDotVal[:,i:(i+1)])
    foverbis = FOverApprox(lipF, E0={'x': xVal, 'xDot' : xDotVal})
    assert np.array_equal(foverbis.E0x, fover.E0x)  and \
            np.array_equal(foverbis.E0xDot, fover.E0xDot)

def test_partialKnowledgeF():
    sizeTest = 10
    xVal = np.random.random((2,sizeTest))
    xDotVal = np.zeros((2,sizeTest))
    for i in range(sizeTest):
        xDotVal[:,i] = fFun(xVal[:,i])
    knownFun = {0 :  {-1 : lambda x : x[1], 0 : lambda x : Interval(0),
                1 : lambda x : Interval(1)}}
    fover = FOverApprox(lipF, E0={'x': xVal, 'xDot' : xDotVal},
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
    fover = FOverApprox(lipF, E0={'x': xVal, 'xDot' : xDotVal},
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

