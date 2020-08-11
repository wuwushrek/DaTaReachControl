import numpy as np
import scipy.optimize as spo
from MyopicDataDrivenControl import MyopicDataDrivenControl
import time


class MyopicDataDrivenControlTrue(MyopicDataDrivenControl):
    """
    Use scipy.optimize for one-step optimal control
    """

    def __init__(self, training_data, input_lb, input_ub, objective,
            one_step_dyn=None, exit_condition=None, solver_style='L-BFGS-B'):
        """
        Based on
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html,
        we use L-BFGS-B, the recommended approach for bound minimization

        self.trajectory excludes the current state
        """

        # Unpack training data
        self.trajectory = training_data['trajectory'][:-1, :]
        self.input_seq = training_data['input_seq']
        self.cost_val_vec = training_data['cost_val']

        # Bounds
        self.bounds = spo.Bounds(input_lb, input_ub)
        self.solver_style = solver_style

        # Functions
        self.objective = objective

        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn, current_state=self.trajectory[-1, None],
            marker_type='^', marker_color='deepskyblue',
            method_name='Opt.\ traj.', marker_default_size=5)

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Get the decision for the current state
        2. Use scipy.optimize to compute the best control action
        3. Return the decision
        """

        # Alternative initial action guess
        #   np.mean(np.vstack((self.bounds.lb, self.bounds.ub)), axis=0)
        initial_action = self.bounds.lb
        if verbose:
            print('Current state: {:s} | Initial action guess: {:s}'.format(
                np.array_str(self.current_state, precision=2),
                np.array_str(initial_action, precision=2)))

        # Compute the decision
        query_timer_start = time.time()
        res = spo.minimize(
                lambda u: self.objective(self.current_state[0, :], u),
                initial_action, bounds=self.bounds, method=self.solver_style)
        current_decision = np.array([res.x])
        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start

        if verbose:
            # Update user
            print('Solver method: {:s}'.format(self.solver_style))
            print('Best action = {:s} | Attained cost = {:1.4f} | Time = '
                  '{:1.4f} s '.format(np.array_str(current_decision,
                                                   precision=2),
                      res.fun[0], query_time))

        # Get the next state
        # self.current_state = self.one_step_dyn(self.current_state,
        #                                        current_decision)

        # Update the data matrices
        self.trajectory = np.vstack((self.trajectory, self.current_state))
        self.input_seq = np.vstack((self.input_seq, res.x))
        self.cost_val_vec = np.hstack((self.cost_val_vec, res.fun))

        # Return decision
        solution_dict = {'next_query': current_decision,
                         'lb_opt_val': res.fun,
                         'query_time': query_time}
        return solution_dict
