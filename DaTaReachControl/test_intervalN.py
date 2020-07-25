import numpy as np
from numpy import float64 as realN

from intervalN import *
from interval import Interval, and_numpy_int
from numba import jit

import time

def n2i(x_lb, x_ub):
    if isinstance(x_lb, int) or isinstance(x_lb, float):
        return Interval(float(x_lb), float(x_ub))
    res = np.full(x_lb.shape, Interval(0),dtype=Interval)
    if len(x_lb.shape) == 1:
        for i in range(res.shape[0]):
            res[i] = Interval(x_lb[i], x_ub[i])
    elif len(x_lb.shape) == 2:
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j] = Interval(x_lb[i,j], x_ub[i,j])
    else:
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                for k in range(res.shape[2]):
                    res[i,j,k] = Interval(x_lb[i,j,k], x_ub[i,j,k])
    return res

def i2n(intVal):
    if isinstance(intVal, Interval):
        return intVal.lb, intVal.ub
    res_lb = np.empty(intVal.shape, dtype=realN)
    res_ub = np.empty(intVal.shape, dtype=realN)
    if len(intVal.shape) == 1:
        for i in range(res_lb.shape[0]):
            res_lb[i], res_ub[i] = intVal[i].lb, intVal[i].ub
    elif len(intVal.shape) == 2:
        for i in range(res_lb.shape[0]):
            for j in range(res_lb.shape[1]):
                res_lb[i,j], res_ub[i,j] = intVal[i,j].lb, intVal[i,j].ub
    else:
        for i in range(res_lb.shape[0]):
            for j in range(res_lb.shape[1]):
                for k in range(res_lb.shape[2]):
                    res_lb[i,j,k], res_ub[i,j,k] = intVal[i,j,k].lb, intVal[i,j,k].ub
    return res_lb, res_ub

def gen_int(shape=None, minVal=-10, widthMax=10):
    if shape is None:
        lb = widthMax * np.random.random() + minVal
        ub = lb + widthMax * np.random.random()
        return Interval(float(lb), float(ub))
    else:
        lb = widthMax * np.random.random(shape) + minVal
        ub = lb + widthMax * np.random.random(shape)
        return n2i(lb,ub)

def test_add(verbose=False, sizeV=(10,), sizeM=(10,10)):
    aInt = gen_int()
    aIntN = i2n(aInt)
    bInt = gen_int()
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt + bInt
    finalI = time.time()
    if verbose:
        print ('Interval add : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = add_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval add : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

    aInt = gen_int(sizeV)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeV)
    bIntN = i2n(bInt)

    timeI = time.time()
    # Add the two intervals using Interval
    resInt = aInt + bInt
    finalI = time.time()
    if verbose:
        print ('Interval add V : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = add_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval add V: '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int(sizeM)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeM)
    bIntN = i2n(bInt)

    timeI = time.time()
    # Add the two intervals using Interval
    resInt = aInt + bInt
    finalI = time.time()
    if verbose:
        print ('Interval add M : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = add_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval add M: '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

def test_sub(verbose=False, sizeV=(10,), sizeM=(10,10)):
    aInt = gen_int()
    aIntN = i2n(aInt)
    bInt = gen_int()
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt - bInt
    finalI = time.time()
    if verbose:
        print ('Interval sub : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = sub_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval sub : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

    aInt = gen_int(sizeV)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeV)
    bIntN = i2n(bInt)

    timeI = time.time()
    # Add the two intervals using Interval
    resInt = aInt - bInt
    finalI = time.time()
    if verbose:
        print ('Interval sub V : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = sub_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval sub V: '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int(sizeM)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeM)
    bIntN = i2n(bInt)

    timeI = time.time()
    # Add the two intervals using Interval
    resInt = aInt - bInt
    finalI = time.time()
    if verbose:
        print ('Interval sub M : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = sub_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval sub M: '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

def test_mul(verbose=False, sizeV=(10,), sizeM=(10,10), sizeX=(10,4,10)):
    aInt = gen_int()
    aIntN = i2n(aInt)
    bInt = gen_int()
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt * bInt
    finalI = time.time()
    if verbose:
        print ('Interval mul : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_i(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

    aInt = gen_int()
    aIntN = i2n(aInt)
    bInt = gen_int()
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt * bInt.lb
    finalI = time.time()
    if verbose:
        print ('Interval mul S : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_i_scalar(*aIntN, bInt.lb)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul S : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

    aInt = gen_int(sizeV)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeV)
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt * bInt[0].lb
    finalI = time.time()
    if verbose:
        print ('Interval mul V : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_i_scalar(*aIntN, bInt[0].lb)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul V : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int(sizeM)
    aIntN = i2n(aInt)
    bInt = gen_int(sizeM)
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = aInt * bInt[0,0].lb
    finalI = time.time()
    if verbose:
        print ('Interval mul M : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_i_scalar(*aIntN, bInt[0,0].lb)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul M : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int(sizeV)
    aIntN = i2n(aInt)
    L = 10 * np.random.random()

    timeI = time.time()
    resInt = aInt * Interval(0,L)
    finalI = time.time()
    if verbose:
        print ('Interval mul V 0,dt : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_iv_0c(*aIntN, L)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul V 0,dt : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int()
    aIntN = i2n(aInt)
    L = 10 * np.random.random()

    timeI = time.time()
    resInt = aInt * Interval(-L,L)
    finalI = time.time()
    if verbose:
        print ('Interval mul Lip : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_i_lip(*aIntN, L)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul Lip : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

    aInt = gen_int(sizeM)
    aIntN = i2n(aInt)
    bInt = gen_int((sizeM[1],))
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = np.matmul(aInt, bInt.reshape(-1,1))[:,0]
    finalI = time.time()
    if verbose:
        print ('Interval mul Mv : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_iMv(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul Mv : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

    aInt = gen_int(sizeX)
    aIntN = i2n(aInt)
    bInt = gen_int((sizeX[1],))
    bIntN = i2n(bInt)

    timeI = time.time()
    resInt = np.tensordot(aInt, bInt.reshape(-1,1), axes=([1,0]))[:,:,0]
    finalI = time.time()
    if verbose:
        print ('Interval mul Tv : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = mul_iTv(*aIntN, *bIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval mul Tv : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert np.array_equal(resIntC, resInt)

def test_and(verbose=False, size=(30,)):
    aInt = gen_int(size)
    maxMin = np.max([aInt[i].lb for i in range(aInt.shape[0])])
    maxMax = np.max([aInt[i].ub for i in range(aInt.shape[0])])
    for i in range(aInt.shape[0]):
        aInt[i].ub = maxMin + (maxMax-maxMin) * np.random.random()
    aIntN = i2n(aInt)

    timeI = time.time()
    resInt = and_numpy_int(aInt)
    finalI = time.time()
    if verbose:
        print ('Interval and : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = and_iv(*aIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval and : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt

def test_norm(verbose=False, size=(10,)):
    aInt = gen_int(size)
    aIntN = i2n(aInt)

    timeI = time.time()
    resInt = np.linalg.norm(aInt)
    finalI = time.time()
    if verbose:
        print ('Interval and : '+ str(finalI-timeI))
    # Add the two interval using Numba
    timeI = time.time()
    resIntCt = norm_i(*aIntN)
    finalI = time.time()
    if verbose:
        print ('Numba Interval Norm : '+ str(finalI-timeI))
    resIntC = n2i(*resIntCt)
    assert resIntC == resInt


test_add()
test_add(verbose=True)

test_sub()
test_sub(verbose=True)

test_mul()
test_mul(verbose=True)

test_and()
test_and(verbose=True)

test_norm()
test_norm(verbose=True)

# @jit(nopython=True)
# def test_bis():
#     v_lb = np.zeros(3,dtype=np.float64)
#     v_ub = np.zeros(3, dtype=np.float64)
#     v_lb[:-1], v_ub[:-1] = add_i(np.array([1.0,3.0]), np.array([2.0,4.0]), 2.0)

# test_bis()
