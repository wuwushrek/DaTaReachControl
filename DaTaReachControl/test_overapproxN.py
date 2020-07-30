import numpy as np
from numpy import float64 as realN

from overapprox_functionsN import *
from intervalN import *
from interval import and_numpy_int
from numba import jit, typeof, typed
# from numba import float64 as real
# from numba import int64 as indType

from reachN import *
from reach import *
from DaTaReachControl import generateTraj, synthNextState, synthTraj, FOverApprox, GOverApprox,Interval

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
n_data_max = 20

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
nSep = int(n_data_max /2)
sepIndexes = np.random.choice([ i for i in range(n_data_max)], nSep, replace=False)
# The trajectory should try system response in each control direction
w_seq[0,0] = 0.0 #
v_seq[0,0] = 0.0
for i in range(1,nSep):
  v_or_theta = np.random.randint(0,2)
  if v_or_theta == 0: # pick v
    w_seq[sepIndexes[i],0] = 0
  else: # pick theta
    v_seq[sepIndexes[i],0] = 0
rand_init_input_vec = np.hstack((v_seq,w_seq))
print (rand_init_input_vec)
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


########################################################################
Lf = np.array([0, 0, 0], dtype=realN)
LG = np.array([[1,0],[1,0],[0,0]], dtype=realN)
bG = Dict(*depTypebG)
bG[(0,0)] = (-1.0,1.0)
bG[(1,0)] = (-1.0,1.0)

nDepG = Dict(*depTypeG)
nDepG[(0,0)] = np.array([0,1],dtype=np.int64)
nDepG[(1,0)] = np.array([0,1],dtype=np.int64)

@jit(nopython=True)
def knownGfun(x_lb, x_ub):
    res_lb = np.zeros((3,2),dtype=realN)
    res_ub = np.zeros((3,2),dtype=realN)
    res_lb[2,1] = 1
    res_ub[2,1] = 1
    # res_lb[0,0], res_ub[0,0] = cos_i(x_lb[2], x_ub[2])
    return res_lb, res_ub

overApprox = initOverApprox(Lf, LG,  Lfknown=None, LGknown=None, nvDepF=depTypeF,
        nvDepG=nDepG, bf=depTypebf , bG =bG , bGf = depTypeGradF,
        bGG=depTypeGradG, xTraj=rand_init_traj_vec.T,
        xDotTraj=rand_init_traj_der_vec.T, uTraj=rand_init_input_vec.T,
        useGronwall=True, verbose=False, Gknown=knownGfun)
########################################################################

########################################################################
bGx = {}
bGx[(0,0)] = Interval(-1.0,1.0)
bGx[(1,0)] = Interval(-1.0,1.0)
nDepGx = {}
nDepGx[(0,0)] = np.array([0,1],dtype=np.int64)
nDepGx[(1,0)] = np.array([0,1],dtype=np.int64)
knownG = {(2,1) : {-1 : lambda x : 1,
                    0 : lambda x : 0,
                    1 : lambda x : 0,
                    2 : lambda x : 0}}
fOverO = FOverApprox(Lf.reshape(-1,1), traj={'x' : rand_init_traj_vec.T,
                                    'xDot' : rand_init_traj_der_vec.T,
                                    'u' : rand_init_input_vec.T},
                    nDep={}, bf={}, bGf={}, knownFun={},
                    Lknown=None, learnLip=False, verbose=True)
GoverO = GOverApprox(LG, fOverO, traj={'x' : rand_init_traj_vec.T,
                                    'xDot' : rand_init_traj_der_vec.T,
                                    'u' : rand_init_input_vec.T},
                    nDep=nDepGx, bG=bGx, bGG={}, knownFun=knownG,
                    Lknown=None, learnLip=False, verbose=False)
########################################################################

########################################################################
@jit(nopython=True)
def test_n(x, randve):
    for i in range(randve.shape[1]):
        Gover(x, randve[:,i], randve[:,i], knownGfun)

# @jit(nopython=True)
def test_o(x, randve):
    for i in range(randve.shape[1]):
        x(np.array([[randve[j,i]] for j in range(randve.shape[0])]))

def test_inclusion(numb, old, randve):
    for i in range(randve.shape[1]):
        val = Gover(numb, randve[:,i], randve[:,i], knownGfun)
        convVal = n2i(*val)
        val2 = old(np.array([[randve[j,i]] for j in range(randve.shape[0])]))
        for k in range(val2.shape[0]):
            for l in range(val2.shape[1]):
                # print(val2[k,l], convVal[k,l])
                assert val2[k,l].contains(convVal[k,l])


x_min = np.min(rand_init_traj_vec.T, axis=1)
x_max = np.max(rand_init_traj_vec.T, axis=1)
res_x = np.zeros((x_min.shape[0], 20))
for i in range(res_x.shape[1]):
    res_x[:,i] = ((x_max - x_min) * np.random.random() + x_min)[:]
    # print(typeof(res_x[:,i]))

test_inclusion(overApprox, GoverO, res_x)
test_n(overApprox, res_x)

s = time.time()
test_n(overApprox, res_x)
print('Numba : ', time.time()-s)

s = time.time()
test_o(GoverO, res_x)
print('Default : ', time.time()-s)

uVal = gen_int(2, minVal=-0.2, widthMax=0.4)
uN = i2n(uVal)
uVal = uVal.reshape(-1,1)

print(uVal)
dtCoeff = getCoeffGronwall(overApprox, sampling_time, *uN)

r_lb, r_ub = fixpoint(overApprox, res_x[:,0], res_x[:,0], sampling_time, *uN,
             knownf=None, knownG=knownGfun, hOver=None)

r_old = fixpointRecursive(np.array([[res_x[j,0]] for j in range(res_x.shape[0])]),
                sampling_time, uVal, fOverO, GoverO)

r_2 = fixpointGronwall(np.array([[res_x[j,0]] for j in range(res_x.shape[0])]),
                            dtCoeff, uVal, fOverO, GoverO)
print (n2i(r_lb,r_ub))
print(r_old)
print(r_2)
########################################################################

############ Test and plot differential inclusion ######################
c_vmax = 0.25 * v_max
c_wmax = 0.25 * w_max
c_rot = 2.0

@jit(nopython=True, parallel=False, fastmath=True)
def uOver(t_lb, t_ub):
    x_lb = np.empty(2, dtype=realN)
    x_ub = np.empty(2, dtype=realN)
    x_lb[0] = c_vmax
    x_ub[0] = c_vmax
    x_lb[1], x_ub[1] = sin_i(c_rot*t_lb, c_rot*t_ub)
    x_lb[1] *= c_wmax
    x_ub[1] *= c_wmax
    return x_lb, x_ub

def uOverO(intT):
    x_lb, x_lb = uOver(intT.lb, intT.ub)
    return n2i(x_lb, x_ub).reshape(-1,1)

@njit(nopython=True, parallel=False, fastmath=True)
def uOver_der(t_lb, t_ub):
    x_lb = np.zeros(2, dtype=realN)
    x_ub = np.zeros(2, dtype=realN)
    x_lb[1], x_ub[1] = cos_i(c_rot*t_lb, c_rot*t_ub)
    x_lb[1] *= c_wmax * c_rot
    x_ub[1] *= c_wmax * c_rot
    return x_lb, x_ub

def uOver_derO(intT):
    x_lb, x_lb = uOver(intT.lb, intT.ub)

########################################################################
