import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from DaTaReachControl import synthNextState, generateTraj,\
                            IDEALISTIC_APG, IDEALISTIC_GRB, OPTIMISTIC_GRB

# Define a seed for repeatability
np.random.seed(2266) # 3327 2266
# val = int(np.random.uniform(0,4000))
# print (val)
# np.random.seed(val)

Cvd = 0.25
Cphid = 0.02255
g = 9.81
m =1.25
l=0.5
Iyy = 0.03
midT1 = 18.39375/2

# Unknown dynamics of the aircraft
def fFun(x):
    res = np.zeros(len(x))
    res[0] = x[1]
    res[1] = (-1.0/m)*Cvd * x[1] # - midT1*(2.0/m) * np.sin(x[4])
    res[2] = x[3]
    res[3] = -g - (1.0/m)*Cvd * x[3] # + midT1*(2.0/m) * np.cos(x[4])
    res[4] = x[5]
    res[5] = (-1.0/Iyy) * Cphid * x[5]
    return res

def GFun(x):
    res = np.zeros((x.shape[0],2))
    res[1,0] = (-1.0/m) * np.sin(x[4])
    res[1,1] = (-1.0/m) * np.sin(x[4])
    res[3,0] = (1.0/m) * np.cos(x[4])
    res[3,1] = (1.0/m) * np.cos(x[4])
    res[5,0] = -l / Iyy
    res[5,1] =  l / Iyy
    return res


# Sampling time
sampling_time = 0.01

# Function giving the true next state of the dyamical system ##
true_next_step = synthNextState(fFun, GFun, sampling_time)

# Initial state
initial_state = np.array([0,0,5,0,0,0])

# Number of data in initial trajectory
n_data_max = 10

# max number of iteration
max_iteration = 200 - n_data_max

# Input bounds
u1_min = 0
u1_max = 18.39375
u2_min = 0
u2_max = 18.39375
input_lb = np.array([u1_min, u2_min], dtype=np.float64)
input_ub = np.array([u1_max, u2_max], dtype=np.float64)

# Generate random input sequence --> put some random zero
# in some direction of the control to have a trajectory with certain quality
u1_seq = (np.random.rand(n_data_max,1) - 0) * (u1_max/10.0)
u2_seq = (np.random.rand(n_data_max,1) - 0) * (u2_max/100.0)
# Number of data points with potentially zero for the value of the control in a direction
nSep = int(n_data_max /2)
# Randomly pick the indexes with zero control in a random direction
sepIndexes = np.random.choice([ i for i in range(n_data_max)], nSep, replace=False)
for i in range(1,nSep):
    zero_or_no = np.random.choice([0,1], p=[0.5,0.5])
    if zero_or_no == 0:
        u1_seq[sepIndexes[i],0] = 0
        u2_seq[sepIndexes[i],0] = 0
        continue
    u1_or_u2 = np.random.choice([0,1], p=[0.5,0.5]) # [0.4,0.6]
    if u1_or_u2 == 0: # pick u2
        u2_seq[sepIndexes[i],0] = 0
    else: # pick theta
        u1_seq[sepIndexes[i],0] = 0
rand_init_input_vec = np.hstack((u1_seq,u2_seq))

print (rand_init_input_vec)

# Generate the trajectory corresponding to random input sequence
# Build initial trajectory shape = (dim(state), trajSize+1) ###
rand_init_traj_vec, rand_init_traj_der_vec = generateTraj(fFun, GFun, initial_state,
                                            rand_init_input_vec, sampling_time, 1)

# The lower bound and upper bound on the state
state_lb = np.array([-30.0, -20, 0.0,  -20, -3.14, -20, u1_min, u2_min])
state_ub = np.array([30.0, 20, 20.0,  20, 3.14, 20, u1_max, u2_max])
print (rand_init_traj_vec[-1,:])

# Planning goals
target_position = np.array([0,5,8,5,0,0])
target_index = np.array([1]) # The states that matter in the cost function

# Compute the cost function
def computeCost(next_state_mat):
    """
    next_state_mat is a Nx4 dimensional vector

    Returns a numpy 1D matrix
    :param next_state_mat:
    :return:
    """
    if next_state_mat.ndim != 2:
        raise ValueError('next_state_mat must be a 2D numpy matrix')
    delta_x_y = next_state_mat[:, target_index] - np.tile(target_position[target_index],
                                                (next_state_mat.shape[0], 1))
    cost = 0.5 * (np.linalg.norm(delta_x_y, axis=1)) ** 2
    return cost

# Generate the cost of the initial trajectory
rand_init_cost_val_vec = computeCost(rand_init_traj_vec[:-1,:])

########## Objective funcions ########################################
def trueObjective(current_state, current_input):
    """
    Compute true objective function
    """
    next_state, _ = true_next_step(current_state, current_input)
    return computeCost(next_state.reshape(1,-1))

def sindycObjective(current_state, delta_state, current_action):
    """
    Redefining compute_cost for cvxpy (used by SINDYc)
    """
    return 0.5 * cp.quad_over_lin(current_state[0, target_index] + delta_state[target_index]\
                                    - target_position[target_index], 1)

def boObjective(context_u):
    """
    Redefining compute_cost used in CGP
    """
    cost_vec = []
    for z_u in context_u:
        x_state, uval = z_u[:-2], z_u[-2:]
        next_state, _ = true_next_step(x_state, uval)
        cost_vec.append(computeCost(next_state.reshape(1,-1)))
    return np.array(cost_vec)

def DaTaControlObjective():
    """ Get the quadratic matrices/vector for the cost function
    """
    Qtarget = np.zeros((initial_state.shape[0], initial_state.shape[0]))
    Rtarget = np.zeros((rand_init_input_vec.shape[1], rand_init_input_vec.shape[1]),
                        dtype=np.float64)
    Starget = np.zeros((initial_state.shape[0], rand_init_input_vec.shape[1]),
                        dtype=np.float64)
    qtarget = np.zeros(initial_state.shape[0], dtype=np.float64)
    rtarget = np.zeros(rand_init_input_vec.shape[1], dtype=np.float64)
    for i in range(target_index.shape[0]):
        Qtarget[target_index[i],target_index[i]] = 1.0
        qtarget[target_index[i]] = -2.0*target_position[target_index[i]]
    return Qtarget, Rtarget, Starget, qtarget, rtarget

def stateToContext(state, state_der=None):
    """
    Mapping from state to context + give the derivative in the new
    coordinate system
    """
    return state

# SINDYc parameters for the unicycle ############################
cvxpy_args_sindyc = {'solver':'MOSEK'}
sparse_thresh=1e-4
eps_thresh=1e-12
scaling_sparse_min=1e-6
scaling_sparse_max = 0.1
libraryFun = lambda state : (state, np.tan(state), np.sin(state), np.cos(state))

# CGP-LCB parameters for the unicycle ###########################
acquistion_type = 'LCB'

# DaTaControl parameters for the unicycle ############################
useGronwall = False
maxData = 10
threshUpdateLearn = 3.5
params_solver = (IDEALISTIC_APG, 0.7, 0.7, 0.7, 1e-12)
# params_solver = (OPTIMISTIC_GRB,)

# File prefix name for logging
log_file="quadrotor_dyn_vx_"
log_extension_file=".png"

# Save plot in a tex file
save_plot_tex = True
realtime_plot = False


time_axes = np.array([ i* sampling_time \
                        for i in range(n_data_max, n_data_max+max_iteration+1)])

def drawInitialPlot():
    """
    Plot the environment, initial starting point, initial data, and the setpoint
    """
    axes_name = [r'$\mathrm{p_x}$', r'$\mathrm{v_x}$', r'$\mathrm{p_y}$',\
                 r'$\mathrm{v_y}$', r'$\mathrm{\phi}$', r'$\mathrm{\omega}$']
    save_figs = list()
    time_meas = np.array([i*sampling_time for i in range(n_data_max+1)])
    for i in range(initial_state.shape[0]):
        fig = plt.figure()
        save_figs.append(fig)
        plt.plot(time_meas, rand_init_traj_vec[:,i], 'bs', markerSize=5,\
                label=r'$\mathscr{T}_{'+ str(n_data_max)+'}$')
        if i in target_index:
            plt.plot(time_axes, np.full( time_axes.shape[0],target_position[i]),
                    'red', linewidth=4, label=r'$\mathrm{Target\ state}$')

        plt.xlabel(r'$\mathrm{Time\ (seconds)}$')
        plt.ylabel(axes_name[i])
        plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.5)
        plt.grid(True)
        plt.tight_layout()
    return save_figs

def runtimePlot_aux(p_state, p_input, n_state, mt, mc, ms, ml, iterVal=0, saved_figs=[]):
    for i, fig in enumerate(saved_figs):
        ax = fig.gca()
        if iterVal == 0:
            ax.plot([time_axes[0], time_axes[1]], [p_state[0, i], n_state[0, i]],\
                linestyle='-',  color=mc, label=ml)
            ax.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left',
                columnspacing=1.5)
        else:
            ax.plot([time_axes[iterVal], time_axes[iterVal+1]],
                    [p_state[0, i], n_state[0, i]], linestyle='-',  color=mc)
        fig.canvas.draw_idle()
    plt.pause(0.0001)

if __name__ == "__main__":
    drawInitialPlot()
    plt.show()
