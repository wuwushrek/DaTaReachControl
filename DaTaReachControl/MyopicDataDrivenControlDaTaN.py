import numpy as np
import time

from MyopicDataDrivenControl import MyopicDataDrivenControl

from controlN import DaTaControlN, OPTIMISTIC_GRB, MIDPOINT_GRB, MIDPOINT_APG
from reachN import depTypebG, depTypeG
from numba import jit
from numba.typed import Dict

from numpy import float64 as realN


Lf = np.array([0, 0, 0], dtype=realN)
LG = np.array([[1,0],[1,0],[0,0]], dtype=realN)
bG = Dict(*depTypebG)
bG[(0,0)] = (-1.0,1.0)
bG[(1,0)] = (-1.0,1.0)

nDepG = Dict(*depTypeG)
nDepG[(0,0)] = np.array([0,1],dtype=np.int64)
nDepG[(1,0)] = np.array([0,1],dtype=np.int64)
# LG = np.array([[0,0],[1,0],[0,0]])

@jit(nopython=True)
def knownGfun(x_lb, x_ub):
    res_lb = np.zeros((3,2),dtype=realN)
    res_ub = np.zeros((3,2),dtype=realN)
    res_lb[2,1] = 1
    res_ub[2,1] = 1
    # res_lb[0,0], res_ub[0,0] = cos_i(x_lb[2], x_ub[2])
    return res_lb, res_ub

class MyopicDataDrivenControlDaTaN(MyopicDataDrivenControl):
    def __init__(self, training_data, input_lb, input_ub, delta_time=0.01,
                Q=None, q=None, R=None, S=None, r=None, one_step_dyn=None,
                exit_condition=None, state_to_context = None):
        # Unpack training data
        self.trajectory = training_data['trajectory'][:-1,:]
        self.trajectory_der = training_data['trajectory_der']
        self.input_seq = training_data['input_seq']

        # Get the states and its derivative in the new coordinate
        if state_to_context is not None:
            transform_traj = state_to_context(training_data['trajectory'])
            _ , transform_traj_der = \
                    state_to_context(self.trajectory, self.trajectory_der)
        else:
            transform_traj, transform_traj_der = self.trajectory,\
                            self.trajectory_der

        # Save the state to context
        self.state_to_context = state_to_context
        self.delta_time = delta_time

        self.synth_control = DaTaControlN(delta_time, Lf, LG, input_lb, input_ub, Q=Q,
            q=q, R=R, S=S, r=r, xTraj=transform_traj.T,
            xDotTraj = transform_traj_der.T, uTraj = self.input_seq.T,
            nvDepG=nDepG, bG =bG, useGronwall=True, verbOverApprox=False, verbCtrl=False,
            knownG=knownGfun, verbSolver=False, threshUpdateApprox=0.5, threshMeanTraj=1e-5,
            coeffLearning=0.1, probLearning=[0.01,0.9,0.09], params=None)


        # Constants
        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn,
            current_state=training_data['trajectory'][-1, None],
            current_state_der = self.trajectory_der[-1, None],
            context_arg_dim=None,  marker_type='o',
            marker_color='magenta', method_name='DaTaControlN')

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Obtain the current context from the current state
        2. Get the decision for the current context
        """
        if self.state_to_context is not None:
            state_e_coord = self.state_to_context(self.current_state).reshape(-1,1)
            _ , state_der_e_coord = self.state_to_context(self.trajectory[-1,:],
                                    self.current_state_der).reshape(-1,1)
        else:
            state_e_coord = self.current_state.flatten()
            state_der_e_coord = self.current_state_der.flatten()

        if verbose:
            print('Current state: {:s}'.format(np.array_str(
                self.current_state, precision=2)))

        # Compute the decision
        query_timer_start = time.time()
        uOpt = self.synth_control(state_e_coord,
                                state_der_e_coord)
        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start

        uOpt = uOpt.reshape(1,-1)

        if verbose:
            print('Best action = {:s} | Time = '
                  '{:1.4f} s '.format(np.array_str(uOpt,
                                                   precision=2), query_time))

        self.trajectory = np.vstack((self.trajectory, self.current_state))
        self.input_seq = np.vstack((self.input_seq, uOpt))
        # self.cost_val_vec = np.hstack((self.cost_val_vec, prob.value))
        solution_dict = {'next_query': uOpt,
                         'query_time': query_time}
        return solution_dict
