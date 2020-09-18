import numpy as np
import time

from MyopicDataDrivenControl import MyopicDataDrivenControl

from DaTaReachControl import *

# knownG = {(2,1) : {-1 : lambda x : 1,
#                     0 : lambda x : 0,
#                     1 : lambda x : 0,
#                     2 : lambda x : 0},
#            (0,0) : {-1 : lambda x : np.cos(x[2,0]),
#                      0 : lambda x : 0,
#                      1 : lambda x : 0,
#                      2 : lambda x : -np.sin(x[2,0])}}
knownG = {(2,1) : {-1 : lambda x : 1,
                    0 : lambda x : 0,
                    1 : lambda x : 0,
                    2 : lambda x : 0}}
nondepG = {(0,0): np.array([0,1]), (1,0) : np.array([0,1])}
bG = {(0,0) : Interval(-1,1), (1,0) : Interval(-1,1)}
Lf = np.array([[0], [0], [0]], dtype=np.float64)
LG = np.array([[1,0],[1,0],[0,0]], dtype=np.float64)
LGknown = None
# LGknown = np.argmin([[1,0],[0,0],[0,0]])
# LG = np.array([[1,0],[1,0],[0,0]])
# LG = np.array([[0,0],[1,0],[0,0]])

class MyopicDataDrivenControlDaTaControl(MyopicDataDrivenControl):
    def __init__(self, training_data, input_lb, input_ub,
                 cost_fun, lipF=Lf, lipG=LG, nDepf={}, bf={}, bGf={},
                knownFunF={}, nDepG=nondepG, bG=bG, bGG={}, knownFunG=knownG,
                optVerb=False, solverVerb=False, solopts={}, probLearning=[0.01,0.9,0.09],
                threshUpdateApprox=0.5, thresMeanTraj=1e-3, coeffLearning=0.1,
                minDataF=1 , maxDataF=20, minDataG=1, maxDataG=20, delta_time=0.01,
                one_step_dyn=None, exit_condition=None,
                state_to_context = None):
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

        # Get range of u as an interval
        uRange = np.full((self.input_seq.shape[1],1),Interval(0))
        for i in range(uRange.shape[0]):
            uRange[i,0] = Interval(input_lb[i], input_ub[i])

        # Save the state to context
        self.state_to_context = state_to_context
        self.delta_time = delta_time

        self.synth_control = DaTaControl(cost_fun, uRange, lipF, lipG,
                traj={'x' : transform_traj.T, 'xDot' : transform_traj_der.T,
                        'u' : self.input_seq.T}, nDepf=nDepf, bf=bf, bGf=bGf,
                nDepG=nDepG, bG=bG, bGG=bGG, knownFunG=knownFunG,
                optVerb=optVerb, solverVerb=solverVerb, learnLipF=False,
                solopts=solopts, probLearning=probLearning, learnLipG=False,
                threshUpdateApprox=threshUpdateApprox, thresMeanTraj=thresMeanTraj,
                coeffLearning=coeffLearning,fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
                widenZeroInterval=1e-3, minDataF=minDataF , maxDataF=maxDataF,
                minDataG=minDataG, maxDataG=maxDataG, dt=self.delta_time,
                gronwall=False, Lgknown=LGknown, term1=True, term2=True)

        # Constants
        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn,
            current_state=training_data['trajectory'][-1, None],
            current_state_der = self.trajectory_der[-1, None],
            context_arg_dim=None,  marker_type='*',
            marker_color='green', method_name='DaTaControl')

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
            state_e_coord = self.current_state.reshape(-1,1)
            state_der_e_coord = self.current_state_der.reshape(-1,1)

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
