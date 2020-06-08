from DaTaReachControl import Interval
import numpy as np

def test_cmpr():
    x1 = Interval(-1,2)
    x2 = Interval(2,6)
    x3 = Interval (-6,-2)
    assert x2 >= x3
    assert x3 <= x2
    assert not x2>=3
    assert x3 <= 0
    assert x2 >= 0

def test_addsub():
    x1 = Interval(-1,2)
    x2 = Interval(2,6)
    x3 = Interval (-6,-2)
    assert x1 + x2 == Interval(1,8)
    assert x2 - 2 == Interval(0,4)
    assert 2 - x2 == Interval(-4,0)
    assert -x3 == Interval(2,6)
    assert x3 + 2 == Interval(-4,0)

def test_muldiv():
    x1 = Interval(-1,2)
    x2 = Interval(2,6)
    x3 = Interval (-6,-2)
    assert x1 * x2 == Interval(-6,12)
    assert x2 * x3 == Interval(-36,-4)
    assert x1 * x3 == Interval(-12,6)
    assert x1 * -1 == Interval(-2,1)
    assert -1 * x1 == Interval(-2,1)
    assert 1.0 / x2 == Interval(1.0/6, 0.5)
    assert x3 / x2 == Interval(-3, -1.0/3)
    assert x2**2 == Interval(4,36)
    assert x1**2 == Interval(0,4)
    assert x1**3 == Interval(-4,8)

def test_sqrtabs():
    x1 = Interval(-1,4)
    x2 = Interval(4,9)
    assert x2.sqrt() == Interval(2,3)
    assert x1.__abs__().sqrt() == Interval(0,2)
