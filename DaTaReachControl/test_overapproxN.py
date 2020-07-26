import numpy as np
from numpy import float64 as realN

from overapprox_functionsN import *
from intervalN import *
from interval import Interval, and_numpy_int
from numba import jit, typeof, typed
# from numba import float64 as real
# from numba import int64 as indType

from reachN import *
from DaTaReachControl import generateTraj, synthNextState, synthTraj

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

xdot_i = 1.0
fx_i_lb = -0.5
fx_i_ub = 0.05
Gx_i_lb = np.array([0.9,-0.1])
Gx_i_ub = np.array([1.1,0.1])
u = np.array([1.0,0])
nf_lb, nf_ub = hc4Revise(xdot_i, fx_i_lb, fx_i_ub, Gx_i_lb, Gx_i_ub, u)
startT = time.time()
nf_lb, nf_ub = hc4Revise(xdot_i, fx_i_lb, fx_i_ub, Gx_i_lb, Gx_i_ub, u)
print(time.time()-startT)
print (nf_lb, nf_ub)
print (Gx_i_lb, Gx_i_ub)

###############################################################################
###############################################################################

# Define a seed for repeatability
np.random.seed(3338) # 801, 994, 3338
# val = int(np.random.uniform(0,4000))
# print (val)
# np.random.seed(val)

# Sampling time
sampling_time = 0.1

###### Define the one step dynamcis of the unicycle ############
def one_step_dyn(current_state, current_input):
    """
    Expects current_state and current_input as 2D matrices where each row is a
    unique time stamp

    Returns a 2D numpy vector of the same number of rows

    current state has 3 dimensions --- position (x, y) and heading (theta)
    current input has 2 dimensions --- velocity (v) and turning rate (w)
    sampling_time has been defined above
    """
    if current_state.ndim == 2 and current_input.ndim == 2:
        if current_state.shape[0] != current_input.shape[0]:
            raise ValueError('Expected current state and input to have the '
                'same number of rows.')
        x, y, theta = current_state.T
        v, w = current_input.T
        nearly_one = np.ones((current_state.shape[0],))
        nearly_zero = np.zeros((current_state.shape[0],))
    elif current_state.ndim == 1 and current_input.ndim == 1:
        x, y, theta = current_state[:]
        v, w = current_input[:]
        nearly_one = 1
        nearly_zero = 0
    else:
        print(current_state, current_input)
        raise ValueError('state and input must be numpy matrices 1D or 2D')

    delta_v = v * sampling_time
    delta_w = w * sampling_time

    if current_state.ndim == 2:
        # Vector delta_w
        nearly_one[abs(delta_w) > 1e-3] = np.sin(delta_w[abs(delta_w) > 1e-3])                                           / delta_w[abs(delta_w) > 1e-3]
        nearly_zero[abs(delta_w) > 1e-3] = (np.cos(delta_w[abs(delta_w) > 1e-3])
                                            - 1) / delta_w[abs(delta_w) > 1e-3]
    elif abs(delta_w) > 1e-3:
        # Scalar delta_w
        nearly_one = np.sin(delta_w) / delta_w
        nearly_zero = (np.cos(delta_w) - 1) / delta_w

    next_state_mat = np.vstack((x + delta_v * (np.cos(theta) * nearly_one
                                               + np.sin(theta) * nearly_zero),
                                y + delta_v * (np.sin(theta) * nearly_one
                                               - np.cos(theta) * nearly_zero),
                                theta + delta_w)).T
    current_state_der = np.vstack((v * np.cos(theta), v * np.sin(theta),
                                    w)).T
    return next_state_mat, current_state_der

# Define the initial satte and the axis limits
initial_state = np.array([-2, -2.5, np.pi/2])

# Number of data in initial trajectory
n_data_max = 10

# max number of iteration
max_iteration = 70 - n_data_max

################### Input bounds  #################################
v_max = 4
w_max = 0.5 * (2*np.pi)
v_min = -v_max
w_min = -w_max
input_lb = np.array([v_min, w_min])
input_ub = np.array([v_max, w_max])

# Generate input sequence
v_seq = -1 *(np.random.rand(n_data_max,1) - 0) * v_max       # Go only backwards
w_seq = 2 * (np.random.rand(n_data_max,1) - 0.5) * w_max
# The trajectory should try system response in each control direction
w_seq[0,0] = 0.0 #
v_seq[0,0] = 0.0
for i in range(1,v_seq.shape[0]):
  v_or_theta = np.random.randint(0,2)
  if v_or_theta == 0: # pick v
    w_seq[i,0] = 0
  else: # pick theta
    v_seq[i,0] = 0
rand_init_input_vec = np.hstack((v_seq,w_seq))
# print (rand_init_input_vec)
###################################################################

# Generate the random trajectory corresponding to random input sequence
rand_init_traj_vec = np.zeros((n_data_max + 1, initial_state.size))
rand_init_traj_der_vec = np.zeros((n_data_max, initial_state.size))
rand_init_traj_vec[0, :] = initial_state
for indx_data in range(n_data_max):
    # Get the next state based on the current state and current input
    rand_init_traj_vec[indx_data+1,:], rand_init_traj_der_vec[indx_data,:]=\
            one_step_dyn(rand_init_traj_vec[indx_data, :],
                                rand_init_input_vec[indx_data, :])
#######################################################################

# This might need to be adjust depending on the seed
xlim_tup = [-3,1.5]
ylim_tup = [-4,1.1]

# Bounds on the context and the input
context_u_lb = np.hstack((np.array([xlim_tup[0], ylim_tup[0], -1, -1]), input_lb))
context_u_ub = np.hstack((np.array([xlim_tup[1], ylim_tup[1], 1, 1]), input_ub))


Lf = np.array([0, 0, 0], dtype=realN)
LG = np.array([[1,0],[1,0],[0,0]], dtype=realN)
bG = Dict(*depTypebG)
bG[(0,0)] = (-1.0,1.0)
bG[(1,0)] = (-1.0,1.0)

nDepG = Dict(*depTypeG)
nDepG[(0,0)] = np.array([0,1],dtype=np.int64)
nDepG[(1,0)] = np.array([0,1],dtype=np.int64)

res = ReachDyn(Lf, LG,  Lfknown=None, LGknown=None, nvDepF=depTypeF,
        nvDepG=nDepG, bf=depTypebf , bG =bG , bGf = depTypeGradF,
        bGG=depTypeGradG, xTraj=rand_init_traj_vec.T,
        xDotTraj=rand_init_traj_der_vec.T, uTraj=rand_init_input_vec.T,
        useGronwall=False, verbose=True)

print (depTypeG)
print(nDepG)
print (res.vDepG)
# lipF = np.array([1.2])
# lipG = np.array([[0.7]])
# res = ReachDyn(lipF, lipG, verbose=True)
# print (depTypeG)

# depTypeF = Dict.empty(key_type=indType, value_type=indType[:])
# depTypeF[0] = np.array([0,1], dtype=np.int64)
# typeof(depTypeF)
# for i, value in depTypeF.items():
#     print(typeof(i), typeof(value))

# @jit(types.void(typeof(depTypeF)), nopython=True)
# def test(val):
#     Jf = np.empty((3,3), dtype=real)
#     Jf[0, :][np.array([0,1])] = 1
#     for i, value in val.items():
#         print(i, value)
#         # print (typeof(i), typeof(value))

# # @jit(types.void(types.void(typeof(depTypeF),)), nopython=True)
# # def test_bis(val):
# #     d = dict()
# #     d[0] = np.array([0,1], dtype=np.int64)
# #     val(d)

# # test_bis(test)

# spec = [('x', real)]
# @jitclass(spec)
# class Test:
#     def __init__(self, x):
#         self.x = x
#     def init_fun(self, f):
#         return f(self.x)

# @jit(real(real), nopython=True)
# def fTest(x):
#     return x*2

# t = Test(5.0)
# print(t.init_fun(fTest))
# # x= np.random.random(10)
# t = Test(x)
# print(x)
# print(t.f(np.zeros(x.shape[0])))
# @jit(real(real), nopython=True)
# def test(x):
#     return x

# @jit(nopython=True, parallel=True)
# def test_dict(x):
#     for key, val in x.items():
#         print (key,val)

# from numba.typed import Dict
# d = Dict()
# d[0] = 1
# d[1] = 2
# d[3] = 3
# test_dict(d)
