import numpy as np

from numba import vectorize
from numba import generated_jit, jit, typeof, from_dtype, types, prange
from numba.experimental import jitclass
from numba import float64 as real
from numpy import float64 as realN

# Tolerance threshold for detecting the value zero
epsTolInt = 1e-6

########################################################################
@jit([types.UniTuple(real,2)(real, real, real, types.misc.Omitted(None)),\
      types.UniTuple(real,2)(real, real, real, real),\
      types.UniTuple(real[:],2)(real[:], real[:], real, types.misc.Omitted(None)),\
      types.UniTuple(real[:],2)(real[:], real[:], real[:], types.misc.Omitted(None)),\
      types.UniTuple(real[:],2)(real[:], real[:], real, real),\
      types.UniTuple(real[:],2)(real[:], real[:], real[:], real[:]),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real, types.misc.Omitted(None)),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real, real),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real[:,:], real[:,:])],\
      nopython=True, parallel=False, fastmath=True)
def add_i(x_lb, x_ub, y_lb, y_ub=None):
    """ Define the addition between an interval vector/matrix x=(x_lb,x_ub) and
        either a scalar y_lb or an interval vector/matrix  y = (y_lb, y_ub)"""
    res_lb = x_lb + y_lb
    if y_ub is None:
        res_ub = x_ub + y_lb
    else:
        res_ub = x_ub + y_ub
    return res_lb, res_ub
########################################################################

########################################################################
@jit([types.UniTuple(real,2)(real, real, real, types.misc.Omitted(None)),\
      types.UniTuple(real,2)(real, real, real, real),\
      types.UniTuple(real[:],2)(real[:], real[:], real, types.misc.Omitted(None)),\
      types.UniTuple(real[:],2)(real[:], real[:], real[:], types.misc.Omitted(None)),\
      types.UniTuple(real[:],2)(real[:], real[:], real, real),\
      types.UniTuple(real[:],2)(real[:], real[:], real[:], real[:]),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real, types.misc.Omitted(None)),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real, real),\
      types.UniTuple(real[:,:],2)(real[:,:], real[:,:], real[:,:], real[:,:])],\
      nopython=True, parallel=False, fastmath=True)
def sub_i(x_lb, x_ub, y_lb, y_ub=None):
    """ Define the substraction between an interval x=(x_lb,x_ub) and
        either a scalar y_lb or an interval  y = (y_lb, y_ub)"""
    if y_ub is None:
        res_lb = x_lb - y_lb
    else:
        res_lb = x_lb - y_ub
    res_ub = x_ub - y_lb
    return res_lb, res_ub
########################################################################

########################################################################
@jit([types.UniTuple(real,2)(real,real,real),\
      types.UniTuple(real[:],2)(real[:],real[:],real),\
      types.UniTuple(real[:,:],2)(real[:,:],real[:,:],real)],\
      nopython=True, parallel=False, fastmath=True)
def mul_i_scalar(x_lb, x_ub, val):
    """Define the multiplication between an interval x=(x_lb,x_ub) and a
       scalar given by val
    """
    if val <= 0:
        return x_ub * val, x_lb * val
    else:
        return x_lb * val, x_ub * val

@jit([types.UniTuple(real[:],2)(real[:],real[:],real[:])],\
      nopython=True, parallel=True, fastmath=True)
def mul_iv_sv(x_lb, x_ub, val):
    """Define the elementwise multiplication between an interval vector
       and a scalar vector """
    res_lb = np.empty(x_lb.shape[0], dtype=realN)
    res_ub = np.empty(x_lb.shape[0], dtype=realN)
    for i in prange(x_lb.shape[0]):
        res_lb[i], res_ub[i] = mul_i_scalar(x_lb[i], x_ub[i], val[i])
    return res_lb, res_ub

@jit([types.UniTuple(real,2)(real,real,real,real)],\
      nopython=True, parallel=False, fastmath=True)
def mul_i(x_lb, x_ub, y_lb, y_ub):
    """Define the multiplication between an interval x=(x_lb,x_ub) and an
       interval given by y = (y_lb, y_ub)
    """
    val_1 = x_lb * y_lb
    val_2 = x_lb * y_ub
    val_3 = x_ub * y_lb
    val_4 = x_ub * y_ub
    return np.minimum(val_1, np.minimum(val_2, np.minimum(val_3,val_4))),\
           np.maximum(val_1, np.maximum(val_2, np.maximum(val_3,val_4)))

@jit([types.UniTuple(real,2)(real,real,real)],\
      nopython=True, parallel=False, fastmath=True)
def mul_i_0c(x_lb, x_ub, val):
    """ Define the multiplication between an interval x=(x_lb,x_ub) and the
        interval (0,val) wher eval >=0 """
    if x_lb >= 0:
        return 0.0, val*x_ub
    if x_ub <= 0:
        return val* x_lb, 0.0
    return val*x_lb, val*x_ub

@jit([types.UniTuple(real,2)(real,real,real)],\
      nopython=True, parallel=False, fastmath=True)
def mul_i_lip(x_lb, x_ub, val):
    """ Define the multiplication between an interval vector x=(x_lb,x_ub) and the
        interval (-val,val) where val >=0
    """
    return np.minimum(-val*x_ub, val*x_lb), np.maximum(-val*x_lb, val*x_ub)


@jit([types.UniTuple(real[:],2)(real[:],real[:],real)],\
      nopython=True, parallel=True, fastmath=True)
def mul_iv_0c(x_lb, x_ub, val):
    """ Define the multiplication between an interval vector x=(x_lb,x_ub) and the
        interval (0,val) where val >=0 """
    res_lb = np.empty(x_lb.shape[0], dtype=realN)
    res_ub = np.empty(x_lb.shape[0], dtype=realN)
    for i in prange(x_lb.shape[0]):
        res_lb[i], res_ub[i] = mul_i_0c(x_lb[i], x_ub[i], val)
    return res_lb, res_ub

@jit([types.UniTuple(real[:],2)(real[:,:],real[:,:],real[:], real[:])],\
      nopython=True, parallel=True, fastmath=True)
def mul_iMv(x_lb, x_ub, y_lb, y_ub):
    """ Define the multiplication between an interval Matrix x=(x_lb,x_ub) and an
        interval vector y = (y_lb, y_ub)
    """
    res_lb = np.empty(x_lb.shape[0], dtype=realN)
    res_ub = np.empty(x_lb.shape[0], dtype=realN)
    for i in prange(x_lb.shape[0]):
        res_i_lb, res_i_ub = 0 , 0
        for k in prange(x_lb.shape[1]):
            t_lb, t_ub = mul_i(x_lb[i,k], x_ub[i,k], y_lb[k], y_ub[k])
            res_i_lb += t_lb
            res_i_ub += t_ub
        res_lb[i] = res_i_lb
        res_ub[i] = res_i_ub
    return res_lb, res_ub

@jit([types.UniTuple(real[:,:],2)(real[:,:,:],real[:,:,:],real[:], real[:])],\
      nopython=True, parallel=True, fastmath=True)
def mul_iTv(x_lb, x_ub, y_lb, y_ub):
    """ Define the multiplication between an interval Tensor x=(x_lb,x_ub) and an
        interval vector y = (y_lb, y_ub)
    """
    res_lb = np.empty((x_lb.shape[0],x_lb.shape[2]), dtype=realN)
    res_ub = np.empty((x_lb.shape[0],x_lb.shape[2]), dtype=realN)
    for i in prange(x_lb.shape[0]):
        for k in prange(x_lb.shape[2]):
            res_i_lb, res_i_ub = 0 , 0
            for l in prange(x_lb.shape[1]):
                if x_lb[i,l,k] == -x_ub[i,l,k]:
                    t_lb, t_ub  = mul_i_lip(y_lb[l], y_ub[l], x_ub[i,l,k])
                else:
                    t_lb, t_ub = mul_i(x_lb[i,l,k], x_ub[i,l,k], y_lb[l], y_ub[l])
                res_i_lb += t_lb
                res_i_ub += t_ub
            res_lb[i,k] = res_i_lb
            res_ub[i,k] = res_i_ub
    return res_lb, res_ub

# @jit([types.UniTuple(real,2)(real, real, real, types.misc.Omitted(None)),\
#       types.UniTuple(real,2)(real, real, real, real),\
#       types.UniTuple(real[:],2)(real[:], real[:], real, types.misc.Omitted(None)),\
#       types.UniTuple(real[:],2)(real[:], real[:], real[:], real[:])],\
#       nopython=True, parallel=False)
# def mul_i(x_lb, x_ub, y_lb, y_ub=None):
#     """ Define the multiplication between an interval x=(x_lb,x_ub) and
#         either a scalar y_lb or an interval  y = (y_lb, y_ub)"""
#     if y_ub is None:
#         if y_lb <=0:
#             return x_ub * y_lb, x_lb * y_lb
#         else:
#             return x_lb * y_lb, x_ub * y_lb
#     val_1 = np.multiply(x_lb, y_lb)
#     val_2 = np.multiply(x_lb, y_ub)
#     val_3 = np.multiply(x_ub, y_lb)
#     val_4 = np.multiply(x_ub, y_ub)
#     return np.minimum(val_1, np.minimum(val_2, np.minimum(val_3, val_4))),\
#            np.maximum(val_1, np.maximum(val_2, np.maximum(val_3, val_4)))
########################################################################

########################################################################
@jit(types.UniTuple(real,2)(real, real, real, real),\
     nopython=True, parallel=False, fastmath=True)
def div_i(x_lb, x_ub, y_lb, y_ub):
    """ Define the division between two intervals """
    assert y_lb > 0 or y_ub < 0
    return mul_i(x_lb, x_ub, 1.0/y_ub, 1.0/ y_lb)

# @jit([types.UniTuple(real[:],2)(real[:], real[:], real[:], real[:])],\
#       nopython=True, parallel=False)
# def div_mi(x_lb, x_ub, y_lb, y_ub):
#     assert np.all(y_lb > 0) or np.all(y_ub < 0)
#     return mul_i(x_lb, x_ub, 1.0/y_ub, 1.0/ y_lb)

# @jit([types.UniTuple(real,2)(real, real, real, types.misc.Omitted(None)),\
#       types.UniTuple(real,2)(real, real, real, real),\
#       types.UniTuple(real[:],2)(real[:], real[:], real, types.misc.Omitted(None))],\
#       nopython=True, parallel=False)
# def div_si(x_lb, x_ub, y_lb, y_ub):
#     if y_ub is None:
#         return mul_i(x_lb, x_ub, 1/y_lb)
#     assert y_lb > 0 or y_ub < 0
#     return mul_i(x_lb, x_ub, 1.0/y_ub, 1.0/ y_lb)

# @generated_jit(nopython=True)
# def div_i(x_lb, x_ub, y_lb, y_ub):
#     if isinstance(x_lb, types.Float) and isinstance(x_ub, types.Float) and \
#         isinstance(y_lb, types.Float) and (isinstance(y_ub, types.Float) or \
#                                             isinstance(y_ub, types.non)):
#         return  div_si
#     else:
#         return div_mi
########################################################################

#########################################################################
@jit([types.UniTuple(real,2)(real, real),
      types.UniTuple(real[:],2)(real[:], real[:])],\
      nopython=True, parallel=False, fastmath=True)
def sqrt_i(x_lb, x_ub):
    """ Compute the square root of an interval or interval vector """
    return np.sqrt(x_lb), np.sqrt(x_ub)
########################################################################

########################################################################
@jit(types.UniTuple(real,2)(real, real),\
     nopython=True, parallel=False, fastmath=True)
def cos_i(x_lb, x_ub):
    """ COmpute the cosinus over-approximation of an interval """
    if np.abs(x_lb-x_ub) < epsTolInt:
        valCos = np.cos(x_ub)
        return valCos, valCos
    scalMin = x_lb % (2*np.pi) if x_lb >= 0 \
                else x_lb + np.ceil(-x_lb / (2*np.pi)) * (2*np.pi)
    scalMax = x_ub % (2*np.pi) if x_ub >= 0 \
                else x_ub + np.ceil(-x_ub / (2*np.pi)) * (2*np.pi)
    cos1 = np.cos(scalMin)
    cos2 = np.cos(scalMax)
    if scalMin >= scalMax:
        return np.minimum(cos1,cos2), 1.0
    elif scalMin < np.pi and np.pi < scalMax:
        return -1.0, np.maximum(cos1,cos2)
    else:
        return np.minimum(cos1,cos2), np.maximum(cos1,cos2)

# @jit(types.UniTuple(real[:],2)(real[:], real[:]),\
        # nopython=True,parallel=True, fastmath=True)
# def cos_mi(x_lb, x_ub):
#     res_lb = np.empty(x_lb.shape[0], dtype=realN)
#     res_ub = np.empty(x_lb.shape[0], dtype=realN)
#     for i in prange(x_lb.shape[0]):
#         (lb, ub) = cos_si(x_lb[i], x_ub[i])
#         res_lb[i] = lb
#         res_ub[i] = ub
#     return res_lb, res_ub

# @generated_jit(nopython=True)
# def cos_i(x_lb, x_ub):
#     if isinstance(x_lb, types.Float) and isinstance(x_ub, types.Float):
#         return cos_si
#     else:
#         return cos_mi
########################################################################

########################################################################
@jit([types.UniTuple(real,2)(real, real)],\
      # types.UniTuple(real[:],2)(real[:], real[:])],\
        nopython=True, parallel=False, fastmath=True)
def sin_i(x_lb, x_ub):
    """ Compute the sinus of an interval """
    nx_lb, nx_ub = sub_i(x_lb, x_ub, np.pi/2)
    return cos_i(nx_lb, nx_ub)
########################################################################

########################################################################
@jit([types.UniTuple(real,2)(real, real)],\
      # types.UniTuple(real[:],2)(real[:], real[:])],\
        nopython=True, parallel=False, fastmath=True)
def tan_i(x_lb, x_ub):
    """ Compute the tan of an interval """
    sx_lb, sx_ub = sin_i(x_lb, x_ub)
    cx_lb, cx_ub = cos_i(x_lb, x_ub)
    return div_i(sx_lb, sx_ub, cx_lb, cx_ub)
########################################################################

########################################################################
@jit(types.UniTuple(real,2)(real, real, real, real),\
     nopython=True, parallel=False, fastmath=True)
def and_i(x_lb, x_ub, y_lb, y_ub):
    """ Compute the intersection of two intervals"""
    lb = np.maximum(x_lb,y_lb)
    ub = np.minimum(x_ub,y_ub)
    assert lb <= ub
    return lb, ub

@jit(types.UniTuple(real,2)(real[:], real[:]),\
     nopython=True, parallel=True, fastmath=True)
def and_iv(x_lb, x_ub):
    """ Compute the intersection of all intervals in a vector"""
    lb = np.max(x_lb)
    ub = np.min(x_ub)
    assert lb <= ub
    return lb, ub

########################################################################

########################################################################
@jit(types.UniTuple(real,2)(real, real),\
     nopython=True, parallel=False, fastmath=True)
def pow2_i(x_lb, x_ub):
    """ COmpute x^2 for the given interval x=(x_lb, x_ub)"""
    if x_ub <= 0:
        return x_ub**2, x_lb**2
    if x_lb >= 0:
        return x_lb**2, x_ub**2
    return 0.0, np.maximum(x_lb**2,x_ub**2)

@jit(types.UniTuple(real,2)(real[:], real[:]),\
     nopython=True, parallel=True, fastmath=True)
def norm_i(x_lb, x_ub):
    """ COmpute the norm of the interval vector given by x_lb and x_ub"""
    res_lb = 0
    res_ub = 0
    for i in prange(x_lb.shape[0]):
        t1, t2 = pow2_i(x_lb[i], x_ub[i])
        res_lb += t1
        res_ub += t2
    return sqrt_i(res_lb, res_ub)
########################################################################


# Predefined types of the Interval
spec = [('lb', real), ('ub', real)]

@jitclass(spec)
class IntervalN(object):
    """Represents a basic Interval and provides the basic mathematical
    operations on intervals.
    :param lb: The lower bound of the interval, can be +-np.Inf.
    :param ub: The upper bound of the interval, can be +-np.Inf.
    """
    def __init__(self, lb=0.0, ub=0.0):
        if np.abs(ub-lb) > epsTolInt:
            assert ub >= lb
            self.lb = lb
            self.ub = ub
        else:
            self.lb = ub
            self.ub = ub

    def __add__(self, other):
        res_lb, res_ub = add_i(self.lb, self.ub, other.lb, other.ub)
        return IntervalN(res_lb, res_ub)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res_lb, res_ub = sub_i(self.lb, self.ub, other.lb, other.ub)
        return IntervalN(res_lb, res_ub)

    def __rsub__(self, other):
        return other.__sub__(self)

    def __mul__(self, other):
        if np.abs(other.lb-other.ub) <= epsTolInt:
            res_lb, res_ub = mul_i_scalar(self.lb, self.ub, other.ub)
            return IntervalN(res_lb, res_ub)
        else:
            res_lb, res_ub = mul_i(self.lb, self.ub, other.lb, other.ub)
            return IntervalN(res_lb, res_ub)

    def __rmul__(self, other):
        return other.__mul__(self)

    def __truediv__(self, other):
        res_lb, res_ub = div_i(self.lb, self.ub, other.lb, other.ub)
        return IntervalN(res_lb, res_ub)

    def __neg__(self):
        return IntervalN(-self.ub, -self.lb)

    def sqrt(self):
        res_lb, res_ub = sqrt_i(self.lb, self.ub)
        return IntervalN(res_lb, res_ub)

    def cos(self):
        res_lb, res_ub = cos_i(self.lb, self.ub)
        return IntervalN(res_lb, res_ub)

    def sin(self):
        res_lb, res_ub = sin_i(self.lb, self.ub)
        return IntervalN(res_lb, res_ub)

    def tan(self):
        res_lb, res_ub = tan_i(self.lb, self.ub)
        return IntervalN(res_lb, res_ub)

    def mulLip(self, L):
        res_lb, res_ub = mul_i_lip(self.lb, self.ub, L)
        return IntervalN(res_lb, res_ub)

    def __abs__(self):
        return np.maximum(np.abs(self.lb), np.max(self.ub))

    def __pow__(self, val):
        # assert isinstance(val, int), 'Power: {val} not an integer'.format(val=val)
        if val == 0:
            return Interval(1,1)
        if val == 1:
            return Interval(self.lb,self.ub)
        if val == 2:
            return Interval(*pow2_i(self.lb, self.ub))
        div_2 = val // 2
        mod_2 = val % 2
        d2 = 2*div_2
        if self.ub <= 0:
            return self.__mul__(Interval(self.ub**d2, self.lb**d2))\
                    if mod_2 == 1 else \
                    Interval(self.ub**d2, self.lb**d2)
        if self.lb >= 0:
            return self.__mul__(Interval(self.lb**d2, self.ub**d2))\
                    if mod_2 == 1 else \
                    Interval(self.lb**d2, self.ub**d2)
        return self.__mul__(Interval(0, np.maximum(self.lb**d2,self.ub**d2)))\
                if mod_2 == 1 else \
                Interval(0, np.maximum(self.lb**d2,self.ub**d2))
