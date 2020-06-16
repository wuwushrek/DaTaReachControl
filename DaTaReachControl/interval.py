import numpy as np

class Interval:
    """Represents a basic Interval and provides the basic mathematical
    operations on intervals.
    :param lb: The lower bound of the interval, can be +-np.Inf.
    :param ub: The upper bound of the interval, can be +-np.Inf.
    """
    # Consider value below this threshold as 0
    epsTol = 1e-12

    def __init__(self, lb=None, ub=None):
        if isinstance(lb, Interval):
            self.lb = lb.lb
            self.ub = lb.ub
            return
        if isinstance(ub, Interval):
            self.lb = ub.lb
            self.ub = ub.ub
            return
        if lb is None and ub is None:
            lb = 0
            ub = 0
        if lb is None and ub is not None:
            lb = ub
        if lb is not None and ub is None:
            ub = lb
        if np.abs(lb-ub) >= Interval.epsTol:
            assert lb <= ub,\
                "Lower bound {} must be less than upper bound {}".format(lb,ub)
            self.lb = float(lb)
            self.ub = float(ub)
        else:
            self.lb = float(ub)
            self.ub = float(ub)

    @property
    def lb(self):
        """Return the lower bound of the interval"""
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value

    @property
    def ub(self):
        """Return the upper bound of the interval"""
        return self._ub

    @ub.setter
    def ub(self, value):
        self._ub = value

    def __repr__(self):
        return '[{lb:.3f} , {ub:.3f}]'.format(lb=self.lb, ub=self.ub)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Interval(self.lb+other, self.ub+other)
        if isinstance(other, Interval):
            return Interval(self.lb+other.lb, self.ub+other.ub)
        assert False, 'Addition not defined for type {}'.format(type(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Interval(self.lb-other, self.ub-other)
        if isinstance(other, Interval):
            return Interval(self.lb-other.ub, self.ub-other.lb)
        assert False, 'Substraction not defined for type {}'.format(type(other))

    def __rsub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Interval(other-self.ub, other-self.lb)
        if isinstance(other, Interval):
            return Interval(other.lb-self.ub, other.ub-self.lb)
        assert False, 'Substraction not defined for type {}'.format(type(other))

    def __mul__(self, other):
        if self is other:
            # print(self, other)
            return self.__pow__(2)
        if isinstance(other, int) or isinstance(other, float):
            if other <= 0:
                return Interval(self.ub*other, self.lb*other)
            return Interval(self.lb*other, self.ub*other)
        if isinstance(other, Interval):
            joint_prod = np.array([self.lb*other.lb, self.lb*other.ub,
                                self.ub*other.lb, self.ub*other.ub])
            return Interval(np.min(joint_prod), np.max(joint_prod))
        assert False, 'Multiplication not defined for type {}'.format(type(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, val):
        assert isinstance(val, int), 'Power: {val} not an integer'.format(val=val)
        if val == 0:
            return Interval(1,1)
        if val == 1:
            return Interval(self.lb,self.ub)
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

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__mul__(1.0/other)
        if isinstance(other, Interval):
            assert other.lb > 0 or other.ub < 0, "0 is inside the interval"
            return self.__mul__(Interval(1.0/other.ub, 1.0/other.lb))
        assert False, 'Division not defined for type {}'.format(type(other))

    def __rtruediv__(self, other):
        assert self.lb > 0 or self.ub < 0, "0 is inside the interval"
        new_int = Interval(1.0/self.ub, 1.0/self.lb)
        return new_int * other

    def __neg__(self):
        return self.__mul__(-1)

    def sqrt(self):
        assert self.__ge__(0), "Interval {} is negative".format(self)
        return Interval(np.sqrt(self.lb), np.sqrt(self.ub))

    def conjugate(self):
        return self

    def cos(self):
        if np.abs(self.lb-self.ub) < Interval.epsTol:
            return Interval(np.cos(self.lb))
        scalMin = self.lb % (2*np.pi) if self.lb >= 0 \
                    else self.lb + np.ceil(-self.lb / (2*np.pi)) * (2*np.pi)
        scalMax = self.ub % (2*np.pi) if self.ub >= 0 \
                    else self.ub + np.ceil(-self.ub / (2*np.pi)) * (2*np.pi)
        cos1 = np.cos(scalMin)
        cos2 = np.cos(scalMax)
        if scalMin >= scalMax:
            return Interval(np.minimum(cos1,cos2), 1)
        elif scalMin < np.pi and np.pi < scalMax:
            return Interval(-1, np.maximum(cos1,cos2))
        else:
            return Interval(np.minimum(cos1,cos2), np.maximum(cos1,cos2))

    def sin(self):
        return self.__sub__(np.pi/2).cos()


    def contains(self, val):
        if isinstance(val, int) or isinstance(val, float):
            return self.lb-val <= Interval.epsTol and \
                                self.ub-val >= -Interval.epsTol
        if isinstance(val, Interval):
            return self.lb-val.lb <= Interval.epsTol and \
                                self.ub-val.ub >= -Interval.epsTol
        assert False, 'Wrong type {} for contains'.format(type(val))

    def __abs__(self):
        if self.__ge__(0):
            return Interval(self.lb,self.ub)
        if self.__le__(0):
            return Interval(-self.ub, -self.lb)
        return Interval(0, np.maximum(-self.lb, self.ub))

    def __and__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            assert self.lb-other<=Interval.epsTol and \
                            self.ub-other >= -Interval.epsTol, \
                    " Empty intersection"
            return Interval(other, other)
        if isinstance(other, Interval):
            return Interval(max(self.lb,other.lb), min(self.ub, other.ub))
        assert False, 'Intersection not defined for type {}'.format(type(other))

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return np.abs(self.lb-other) <= Interval.epsTol and \
                                np.abs(self.ub-other) <= Interval.epsTol
        if isinstance(other, Interval):
            return np.abs(self.lb-other.lb)<=Interval.epsTol and \
                                    np.abs(self.ub-other.ub)<=Interval.epsTol
        assert False, 'Not the same object {}'.format(type(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.ub - other < Interval.epsTol
        if isinstance(other, Interval):
            return self.ub - other.lb < Interval.epsTol
        assert False, 'Not the same object {}'.format(type(other))

    def __gt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.lb - other > -Interval.epsTol
        if isinstance(other, Interval):
            return self.lb - other.ub > -Interval.epsTol
        assert False, 'Not the same object {}'.format(type(other))

    def __le__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.ub - other <= Interval.epsTol
        if isinstance(other, Interval):
            return self.ub - other.lb <= Interval.epsTol
        assert False, 'Not the same object {}'.format(type(other))

    def __ge__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.lb - other >= -Interval.epsTol
        if isinstance(other, Interval):
            return self.lb - other.ub >= -Interval.epsTol
        assert False, 'Not the same object {}'.format(type(other))
