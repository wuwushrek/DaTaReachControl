import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from DaTaReachControl import synthNextState, generateTraj,\
                            IDEALISTIC_APG, IDEALISTIC_GRB, OPTIMISTIC_GRB

# Define a seed for repeatability
np.random.seed(1882) # 3372 (good)  # 1013 (bad), 1882
# val = int(np.random.uniform(0,4000))
# print (val)
# np.random.seed(val)

# Unknown dynamics of the unicycle
def fFun(x):
    return np.zeros(3, dtype=np.float64)

def GFun(x):
    return np.array([[np.cos(x[2]), 0], [np.sin(x[2]), 0], [0.0, 1.0]])

# Sampling time
sampling_time = 0.1

# Function giving the true next state of the dyamical system ##
true_next_step = synthNextState(fFun, GFun, sampling_time)

# Define the initial satte and the axis limits
initial_state = np.array([-2, -2.5, np.pi/2])

# Number of data in initial trajectory
n_data_max = 10

# max number of iteration
# max_iteration = 150 - n_data_max
max_iteration = 150 - n_data_max

# Input bounds
v_max = 3
w_max = 0.5 * (2*np.pi)
v_min = -v_max
w_min = -w_max
input_lb = np.array([v_min, w_min], dtype=np.float64)
input_ub = np.array([v_max, w_max], dtype=np.float64)

# Generate random input sequence --> put some random zero
# in some direction of the control to have a trajectory with certain quality
v_seq = -1 *(np.random.rand(n_data_max,1) - 0) * v_max # 2 * (np.random.rand(n_data_max,1) - 0.5) * v_max
w_seq = 2 * (np.random.rand(n_data_max,1) - 0.5) * w_max
# Number of data points with potentially zero for the value of the control in a direction
nSep = int(n_data_max /2)
# Randomly pick the indexes with zero control in a random direction
sepIndexes = np.random.choice([ i for i in range(n_data_max)], nSep, replace=False)
for i in range(1,nSep):
    zero_or_no = np.random.choice([0,1], p=[0.2,0.8])
    if zero_or_no == 0:
        w_seq[sepIndexes[i],0] = 0
        v_seq[sepIndexes[i],0] = 0
        continue
    v_or_theta = np.random.choice([0,1], p=[0.4,0.6]) # [0.4,0.6]
    if v_or_theta == 0: # pick v
        w_seq[sepIndexes[i],0] = 0
    else: # pick theta
        v_seq[sepIndexes[i],0] = 0
rand_init_input_vec = np.hstack((v_seq,w_seq))

print (rand_init_input_vec)

# Generate the trajectory corresponding to random input sequence
# Build initial trajectory shape = (dim(state), trajSize+1) ###
rand_init_traj_vec, rand_init_traj_der_vec = generateTraj(fFun, GFun, initial_state,
                                            rand_init_input_vec, sampling_time, 1)

# This might need to be adjust depending on the seed --> Dimension of the grid
# xlim_tup = [-3,1.5]
# ylim_tup = [-3.9,1.0]
# xlim_tup = [-3,1]
# ylim_tup = [-3.9,2]
xlim_tup = [-3,1.5]
ylim_tup = [-4.5,1.5]

# Bounds on the context and the input
context_u_lb = np.hstack((np.array([xlim_tup[0], ylim_tup[0], -1, -1]), input_lb))
context_u_ub = np.hstack((np.array([xlim_tup[1], ylim_tup[1], 1, 1]), input_ub))

# Planning goals
target_position = np.zeros(2)

# Terminate the sequential control if cost is below this threshold
cost_thresh = 0.5 * (0.5 ** 2)

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
    delta_x_y = next_state_mat[:, :2] - np.tile(target_position[:2],
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
    return 0.5 * cp.quad_over_lin(current_state[0, :2] + delta_state[:2]\
                                    - target_position, 1)

def boObjective(context_u):
    """
    Redefining compute_cost used in CGP
    """
    cost_vec = []
    for z_u in context_u:
        x, y, sh, ch, v, w = z_u[:]
        current_state = np.array([[x, y, np.arctan2(sh, ch)]])
        current_input = np.array([[v, w]])
        next_state, _ = true_next_step(current_state, current_input)
        cost_vec.append(computeCost(next_state.reshape(1,-1)))
    return np.array(cost_vec)

def DaTaControlObjective():
    """ Get the quadratic matrices/vector for the cost function
    """
    Qtarget = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    Rtarget = np.zeros((rand_init_input_vec.shape[1], rand_init_input_vec.shape[1]),
                        dtype=np.float64)
    Starget = np.zeros((initial_state.shape[0], rand_init_input_vec.shape[1]),
                        dtype=np.float64)
    qtarget = np.zeros(initial_state.shape[0], dtype=np.float64)
    rtarget = np.zeros(rand_init_input_vec.shape[1], dtype=np.float64)
    return Qtarget, Rtarget, Starget, qtarget, rtarget


########## Utilities functions #######################################
def exitCondition(current_state, current_input, next_state):
    """
    Exit condition
    """
    return computeCost(next_state) <= cost_thresh

def stateToContext(state, state_der=None):
    """
    Mapping from state to context + give the derivative in the new
    coordinate system
    """
    x, y, theta = state.T
    context_mat = np.vstack((x, y, np.sin(theta), np.cos(theta))).T
    if state_der is None:
        return context_mat
    else:
        x_der, y_der, theta_der = state_der.T
        context_der_mat = np.vstack((x_der,
                            y_der,
                            np.multiply(theta_der,np.cos(theta)),
                            -np.multiply(theta_der,np.sin(theta)))).T
        return context_mat, context_der_mat

def firstOrderOracle(z_u_data):
    """
    Takes in context+input combination and provides cost function value and
    its gradient with respect to context+input
    """
    if z_u_data.shape[1] != 6:
        print(z_u_data.shape)
        raise ValueError('Requires feature and input vector (6 col matrix)')

    cost_vec = []
    grad_cost_vec = []
    for z_u in z_u_data:
        x, y, sh, ch, v, w = z_u[:]
        if abs(sh ** 2 + ch ** 2 - 1) > 1e-8:
            print('Error in the user-provided value: sin', sh, '| cos', ch)
            raise ValueError('Expected sin(heading)^2 + cos(heading)^2 == 1')
        if abs(w * sampling_time) >= 1e-5:
            nearly_one = np.sin(w * sampling_time)/(w * sampling_time)
            nearly_zero = (np.cos(w * sampling_time) - 1)/(w * sampling_time)
            # https://www.wolframalpha.com/input/?i=simplify+d%2Fdw+sin%28wt%29%2F%28wt%29
            # (t w cos(t w) - sin(t w))/(t w^2)
            nearly_one_dw = (sampling_time * w * np.cos(sampling_time * w)
                             - np.sin(sampling_time * w)) \
                                                    /(sampling_time * (w ** 2))
            # https://www.wolframalpha.com/input/?i=simplify+d%2Fdw+%28cos%28wt%29+-+1%29%2F%28wt%29
            # -(t w sin(t w) + cos(t w) - 1)/(t w^2)
            nearly_zero_dw = - (sampling_time * w * np.sin( sampling_time * w)
                                + np.cos(sampling_time * w) - 1) \
                                                    /(sampling_time * (w ** 2))
        else:
            nearly_zero = 0
            nearly_one = 1
            # They are constants
            nearly_zero_dw = 0
            nearly_one_dw = 0

        # Change in position
        delta_x = v * sampling_time * (ch * nearly_one + sh * nearly_zero)
        delta_y = v * sampling_time * (sh * nearly_one - ch * nearly_zero)

        # Cost definition
        current_state = np.array([x, y, np.arctan2(sh, ch)])
        current_input = np.array([v, w])
        next_state, _ = true_next_step(current_state, current_input)
        cost_vec = np.hstack((cost_vec, computeCost(next_state.reshape(1,-1))))

        # Components of the gradient via chain rule
        cost_dx = 2 * (x + delta_x - target_position[0])
        cost_dy = 2 * (y + delta_y - target_position[1])
        cost_dsh = cost_dx * (v * sampling_time * nearly_zero) +\
                   cost_dy * (v * sampling_time * nearly_one)
        cost_dch = cost_dx * (v * sampling_time * nearly_one) -\
                   cost_dy * (v * sampling_time * nearly_zero)
        cost_dv = cost_dx * sampling_time * (ch * nearly_one + sh* nearly_zero)\
                + cost_dy * sampling_time * (sh * nearly_one - ch* nearly_zero)
        cost_dw = cost_dx*v*sampling_time*(ch*nearly_one_dw+sh*nearly_zero_dw)\
                + cost_dy*v*sampling_time*(sh*nearly_one_dw-ch*nearly_zero_dw)
        grad_cost = [cost_dx, cost_dy, cost_dsh, cost_dch, cost_dv, cost_dw]
        grad_cost_vec.append(grad_cost)

    return cost_vec, np.array(grad_cost_vec)/2

# SINDYc parameters for the unicycle ############################
cvxpy_args_sindyc = {'solver':'MOSEK'}

# CGP-LCB parameters for the unicycle ###########################
acquistion_type = 'LCB'

# C2Opt parameters for the unicycle ############################
grad_lips_constant_c2opt = 1e1
solver_str_c2opt = 'cvxpy'

# DaTaControl parameters for the unicycle ############################
useGronwall = False
maxData = 10
threshUpdateLearn = 0.5
params_solver = (IDEALISTIC_APG, 0.7, 0.7, 0.7, 1e-6)
# params_solver = (IDEALISTIC_GRB, 0.7, 0.7, 0.7)
# params_solver = (OPTIMISTIC_GRB,)

############ Plotting details ###################################
# scatter_size = 8
# cost_markersize = 3
# fig_fontsize = 12
# params = {
#    'axes.labelsize': fig_fontsize,
#    'font.family': 'serif',
#    'font.size': fig_fontsize,
#    'legend.fontsize': fig_fontsize,
#    'xtick.labelsize': fig_fontsize,
#    'ytick.labelsize': fig_fontsize,
#    'mathtext.fontset': 'cm',
#    }
# plt.rcParams.update(params)
cost_markersize = 5
traj_markersize = 10
target_markersize=30
initstate_markersize =30

# File prefix name for logging
log_file="unicycle_dyn_crap"
log_extension_file=".png"

# Save plot in a tex file
save_plot_tex = True
realtime_plot = False

def drawInitialPlot(xlim_tup, ylim_tup, target_position, cost_thresh,
                      initial_state, rand_init_traj_vec):
    """
    Plot the environment, distance contours, initial starting point, initial
    data, and the target set
    """
    # Draw the plot
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')                                 # Equal x and y axis
    plt.xlabel(r'$\mathrm{p_x}$')
    plt.ylabel(r'$\mathrm{p_y}$')
    ax.set_xlim(xlim_tup)
    ax.set_xticks(np.round(np.arange(xlim_tup[0], xlim_tup[1] + 1, 2)))
    ax.set_ylim(ylim_tup)
    plt.grid()

    draw_theta = np.linspace(0, 2 * np.pi, 100)         # For plotting circles
    zorder_init = 1e4                                   # Zorder for plotting
    skip_marker = 1

    # Draw contour plots
    for r_temp in np.arange(1, 20, 1):
        plt.plot(target_position[0] + r_temp * np.cos(draw_theta),
                 target_position[1] + r_temp * np.sin(draw_theta),
                 color='k', linewidth=1)

    # Draw the initial state
    plt.scatter(initial_state[0], initial_state[1], initstate_markersize,
            marker='d', color='y', label=r'$\mathrm{Initial\ state}$',
            zorder=zorder_init + 2)

    # Draw initial trajectory
    plt.scatter(rand_init_traj_vec[1::skip_marker, 0],
                rand_init_traj_vec[1::skip_marker, 1], traj_markersize,
                marker='s', color='b', zorder=zorder_init,
                label=r'$\mathrm{Initial\ data}$')

    plt.scatter(target_position[0], target_position[1], target_markersize,
                marker = '*', color='r', zorder=11,
                label=r'$\mathrm{Target\ state}$')

    # Draw the acceptable closeness to the target
    dist_thresh = np.sqrt(cost_thresh * 2)
    plt.plot(target_position[0] + dist_thresh * np.cos(draw_theta),
             target_position[1] + dist_thresh * np.sin(draw_theta),
             color='k', linestyle=':', linewidth=1, zorder=zorder_init,
             label=r'$\mathrm{Target\ set}$')

    # Interpolate the data points --- approximation
    plt.plot(rand_init_traj_vec[1:, 0], rand_init_traj_vec[1:, 1], color='b')
    ax.legend(loc='center left', ncol=1, labelspacing=0.25, framealpha=1,
              bbox_to_anchor=(1.,0.5))
    plt.tight_layout()
    return ax

def runtimePlot(p_state, p_input, n_state, mt, mc, ms, ml, iterVal=0):
    ax = plt.gca()
    if iterVal == 0:
        ax.plot([p_state[0, 0], n_state[0, 0]],
            [p_state[0, 1], n_state[0, 1]], linestyle='-', marker=mt,
            markerSize=ms, color=mc, label=ml)
        ax.legend(loc='center left', ncol=1, labelspacing=0.25, framealpha=1,
              bbox_to_anchor=(1.,0.5))
    else:
       ax.plot([p_state[0, 0], n_state[0, 0]],
            [p_state[0, 1], n_state[0, 1]], linestyle='-', marker=mt,
            markerSize=ms, color=mc)
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    drawInitialPlot(xlim_tup, ylim_tup, target_position, cost_thresh,
                      initial_state, rand_init_traj_vec)
    plt.show()
