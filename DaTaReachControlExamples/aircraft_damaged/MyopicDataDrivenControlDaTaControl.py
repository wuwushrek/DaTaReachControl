import numpy as np
import time

from DaTaReachControlExamples.MyopicDataDrivenControl import MyopicDataDrivenControl

from DaTaReachControl import DaTaControl, OPTIMISTIC_GRB, IDEALISTIC_GRB, IDEALISTIC_APG
from DaTaReachControl import depTypeG, depTypeF
from DaTaReachControl.interval import *

from numba import jit
from numba.typed import Dict

from numpy import float64 as realN

Lfknown = np.array([0, 0, 0, 1, 2.5])
# Lfknown = np.array([0, 0, 0, 1.0, 1.0])
Lf = np.array([0.35, 2.5, 3.5, 0, 0])

LGknown = np.zeros((5,2), dtype=realN)
LG = np.zeros((5,2), dtype=realN)
LG[0,0] = 0.01
LG[0,1] = 0.01
LG[1,0] = 0.01
LG[1,1] = 0.01
LG[2,0] = 0.01
LG[2,1] = 0.5

nDepf = Dict(*depTypeF)
nDepf[0] = np.array([3,4], dtype=np.int64)
nDepf[1] = np.array([3,4], dtype=np.int64)
nDepf[2] = np.array([3,4], dtype=np.int64)
# nDepf[3] = np.array([0,1,3,4], dtype=np.int64)
# nDepf[4] = np.array([0,2,4], dtype=np.int64)

nDepG = Dict(*depTypeG)
nDepG[(0,0)] = np.array([0,1,3,4],dtype=np.int64)
nDepG[(0,1)] = np.array([0,1,3,4],dtype=np.int64)
nDepG[(1,0)] = np.array([0,1,3,4],dtype=np.int64)
nDepG[(1,1)] = np.array([0,1,3,4],dtype=np.int64)
nDepG[(2,0)] = np.array([0,1,3,4],dtype=np.int64)
nDepG[(2,1)] = np.array([0,2,3,4],dtype=np.int64)


@jit(nopython=True)
def knownGfun(x_lb, x_ub):
    res_lb = np.zeros(LG.shape,dtype=realN)
    res_ub = np.zeros(LG.shape,dtype=realN)
    # res_lb[0,0], res_ub[0,0] = 0.01, 0.01
    # res_lb[0,1], res_ub[0,1] = 1, 1
    # res_lb[1,0], res_ub[1,0] = -0.064, -0.064
    # res_lb[1,1], res_ub[1,1] = -0.044, -0.044
    # res_lb[2,0], res_ub[2,0] = -0.378, -0.378
    # res_lb[2,1] = 1
    # res_ub[2,1] = 1
    return res_lb, res_ub

@jit(nopython=True)
def knownFfun(x_lb, x_ub):
    res_lb = np.zeros(Lf.shape[0],dtype=realN)
    res_ub = np.zeros(Lf.shape[0],dtype=realN)
    res_lb[3], res_ub[3] = x_lb[2], x_ub[2]
    res_lb[4], res_ub[4] = sub_i(2.21*x_lb[3], 2.21*x_ub[3], x_lb[1], x_ub[1])
    # res_lb[4], res_ub[4] = -x_ub[1], -x_lb[1] # -x_2
    return res_lb, res_ub

@jit(nopython=True)
def knownJacFfun(x_lb, x_ub):
    res_lb = np.zeros((Lf.shape[0],Lf.shape[0]), dtype=realN)
    res_ub = np.zeros((Lf.shape[0],Lf.shape[0]), dtype=realN)
    res_lb[3,2], res_ub[3,2] = 1.0 , 1.0
    res_lb[4,1], res_ub[4,1] = -1.0, -1.0
    res_lb[4,3], res_ub[4,3] = 2.21, 2.21
    return res_lb, res_ub


class MyopicDataDrivenControlDaTaControl(MyopicDataDrivenControl):
    def __init__(self, training_data, input_lb, input_ub, DaTaControlObjective,
                    delta_time=0.1, one_step_dyn=None, exit_condition=None,
                    useGronwall=False, maxData=20, threshUpdateApprox=0.5,
                    verbCtrl = False, params=(IDEALISTIC_APG, 0.7, 0.7, 0.7, 1e-6)):
        # Unpack training data
        self.trajectory = training_data['trajectory'][:-1,:]
        self.trajectory_der = training_data['trajectory_der']
        self.input_seq = training_data['input_seq']

        # Get the matrices for the cost function
        Q, R, S, q, r = DaTaControlObjective()

        # Save the state to context
        self.delta_time = delta_time

        self.synth_control = DaTaControl(delta_time, Lf, LG, input_lb, input_ub, Q=Q,
            q=q, R=R, S=S, r=r, xTraj=self.trajectory, Lfknown=Lfknown,
            LGknown=LGknown, xDotTraj = self.trajectory_der,
            uTraj = self.input_seq, nvDepF=nDepf, nvDepG=nDepG,
            useGronwall=useGronwall, verbOverApprox=False, tolChange=1e-2,
            maxInvariantIter=2, verbCtrl=verbCtrl, knownG=knownGfun,
            knownf=knownFfun, gradKnownf=knownJacFfun, gradKnownG=None,
            verbSolver=False, threshUpdateApprox=threshUpdateApprox, coeffLearning=0.1,
            probLearning=[0.6,0.15,0.25], maxData=maxData, params=params)


        # Constants
        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn,
            current_state=training_data['trajectory'][-1, None],
            current_state_der = self.trajectory_der[-1, None],
            context_arg_dim=None,  marker_type='*',
            marker_color='darkgreen', method_name='DaTaControl')

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Obtain the the current state
        2. Get the near-optimal control input for the current state
        """
        state_e_coord = np.squeeze(self.current_state)
        state_der_e_coord = np.squeeze(self.current_state_der)

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
