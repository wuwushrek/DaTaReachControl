import numpy as np
import matplotlib.pyplot as plt
import traceback                    # For displaying caught exceptions


class MyopicDataDrivenControl:
    """
    Base class for a one-step control. This script is inspired by congol Python
    module available at https://github.com/abyvinod/congol

    Each subclass should populate its function
    compute_decision_for_current_state
    """

    def __init__(self, exit_condition=None, one_step_dyn=None,
                 current_state=None, current_state_der=None,
                 marker_type='x', marker_color='k',
                 method_name='MyopicCtrl', zorder=10,
                 context_arg_dim=None, marker_default_size=4):
        # Functions
        self.exit_condition = exit_condition
        self.one_step_dyn = one_step_dyn
        # Current state and context
        self.current_state = current_state
        self.current_state_der = current_state_der
        # Constants
        self.context_arg_dim = context_arg_dim
        # Plotting constants
        self.marker_type = marker_type
        self.marker_color = marker_color
        self.marker_label = r'$\mathrm{' + method_name + '}$'
        self.zorder = zorder
        self.marker_default_size = marker_default_size

    def solve(self, max_time_steps, runtime_info=None, verbose=False):
        """
        Solve the contextual optimization problem at most max_time_steps or
        until an exit condition is met. This function calls the one-step
        dynamics when max_step > 1 is provided.

        If the user only requires the decision for the current state, then set
        max_step = 1. In this case, the user must update the current_state to
        the next_state based on whichever was action

        If runtime_info is provided, it do whatever needs to be execute
        by runtime_info on the fly (for example plotting, logging, etc...)
        """
        res = []
        # Loop till the maximum number of iterations have not been reached
        for iter_count in range(max_time_steps):
            if verbose:
                print('\n' + str(iter_count) + '. ', end='')
            # Step 1: Compute the current decision
            try:
                res_iter = self.compute_decision_for_current_state(
                    verbose=verbose)
            except RuntimeError:
                traceback.print_exc()
                procedure_name_temp = self.marker_label.strip('r$\mathrm{')
                procedure_name = procedure_name_temp.strip('}')
                print('\n\n>>> ' + procedure_name + ' approach failed due to '
                                                    'numerical issues!')
                print('Terminating early!')
                if not res:
                    res = [{'query_time': np.Inf, 'lb_opt_val': np.NaN,
                            'next_query': np.NaN}]
                return res
            res.append(res_iter)
            current_decision = res_iter['next_query'][0:, self.context_arg_dim:]

            # Quit here if the user does not want us to propagate the dynamics
            if max_time_steps == 1:
                if verbose:
                    print('self.current_state was not updated')
                continue

            # Step 2: ASSUMES one_step_dyn exists and use it to update the
            # current state | Collect the (x_t, u_t, x_{t+1}). For ease of
            # coding, x_{t+1} is self.current_state
            past_state = self.current_state
            past_input = current_decision               # Make it 2D
            self.current_state, self.current_state_der = \
                                self.one_step_dyn(past_state, past_input)
            self.current_state_der = self.current_state_der.reshape(1,-1)
            self.current_state = self.current_state.reshape(1,-1)
            if runtime_info is not None:
                runtime_info(p_state=past_state, p_input=past_input,
                    n_state=self.current_state, mt=self.marker_type,
                    mc = self.marker_color, ms=self.marker_default_size,
                    ml = self.marker_label, first= iter_count==0)

            # Step 3: Break early if a user-provided exit condition is met
            if self.exit_condition is not None and\
                 self.exit_condition(past_state, past_input, self.current_state):
                break
        return res
