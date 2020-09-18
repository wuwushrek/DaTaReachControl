import numpy as np
from .MyopicDataDrivenControl import MyopicDataDrivenControl
import GPyOpt as gpy
import time


class MyopicDataDrivenControlContextGP(MyopicDataDrivenControl):
    """
    Use contextual Gaussian Process for one-step control
    """

    def __init__(self, training_data, state_to_context, context_u_lb,
                 context_u_ub, objective, one_step_dyn=None,
                 exit_condition=None, solver_style='LCB'):

        # Unpack training data
        training_trajectory = training_data['trajectory']
        training_input_seq = training_data['input_seq']
        training_cost = training_data['cost_val']
        training_state_vec = training_trajectory[:-1, :]
        training_context_vec = state_to_context(training_state_vec)
        training_context_u_vec = np.hstack((training_context_vec,
                                            training_input_seq))

        # Domain definition
        self.domain = [{'name': 'var_' + str(indx), 'type': 'continuous',
                        'domain': (lb, ub)}
                       for indx, lb, ub in zip(range(context_u_lb.size),
                            context_u_lb, context_u_ub)]

        # Current context
        self.current_context = None
        self.solver_style = solver_style

        # Functions
        self.state_to_context = state_to_context
        self.objective = objective

        # Bayesian optimizer
        # Y must be a 2-D matrix with values arranged along the column
        self.bo_step = gpy.methods.BayesianOptimization(None,
                domain=self.domain, X=training_context_u_vec,
                Y=np.array([training_cost]).T, exact_feval=True,
                acquisition_type=self.solver_style)

        # Constants
        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn,
            current_state=training_trajectory[-1, None],
            context_arg_dim=training_context_vec.shape[1],  marker_type='+',
            marker_color='slategray', method_name='CGP-' + self.solver_style)

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Obtain the current context from the current state
        2. Get the decision for the current context
        3. Query the oracle and update the bayesian optimizer object
        4. Return the decision
        """

        # Get the context
        context_vec = self.state_to_context(self.current_state)
        if verbose:
            print('Current context: {:s}'.format(np.array_str(context_vec,
                                                              precision=2)))
        self.current_context = {'var_' + str(indx): context_vec[0, indx]
                for indx in range(self.context_arg_dim)}

        # Get the next decision for the next context
        query_timer_start = time.time()
        next_z = self.bo_step.suggest_next_locations(
                context=self.current_context)
        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start

        next_z_cost = self.objective(next_z)
        if verbose:
            # Update user
            print('Acquisition type: {:s} | Acquisition optimizer type: '
                  '{:s}'.format(self.bo_step.acquisition_type,
                                self.bo_step.acquisition_optimizer_type))
            print('z_query = {:s} | Cost estimate = {:1.4f} | Time = '
                  '{:1.4f} s '.format(np.array_str(next_z, precision=2),
                                      next_z_cost[0, 0], query_time))

        # Update the Bayesian optimizer
        new_X = np.vstack((self.bo_step.X, next_z))
        new_Y = np.vstack((self.bo_step.Y, next_z_cost))
        self.bo_step = gpy.methods.BayesianOptimization(None,
                domain=self.domain, X=new_X, Y=new_Y, exact_feval=True,
                acquisition_type=self.solver_style)

        # Return decision
        solution_dict = {'next_query': next_z,
                         'lb_opt_val': next_z_cost[0, 0],
                         'query_time': query_time}
        return solution_dict
