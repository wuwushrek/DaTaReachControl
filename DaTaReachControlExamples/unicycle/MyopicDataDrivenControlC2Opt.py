import numpy as np
import congol as cg

from DaTaReachControlExamples.MyopicDataDrivenControl import MyopicDataDrivenControl

class MyopicDataDrivenControlC2Opt(MyopicDataDrivenControl):
    """
    Myopic data-driven control for apriori unknown smooth systems via
    convexified contextual optimization
    """
    def __init__(self, training_data, state_to_context, context_u_lb,
                 context_u_ub, first_order_oracle_cost, grad_lips_constant_cost,
                 one_step_dyn=None, exit_condition=None,
                 solver='cvxpy', solver_style=None):

        # Unpack training data
        training_trajectory = training_data['trajectory']
        training_input_seq = training_data['input_seq']
        training_cost = training_data['cost_val']
        training_cost_grad = training_data['cost_grad']
        training_state_vec = training_trajectory[:-1, :]
        training_context_vec = state_to_context(training_state_vec)
        training_context_u_vec = np.hstack((training_context_vec,
                                            training_input_seq))

        # Functions
        self.state_to_context = state_to_context
        self.first_order_oracle_cost = first_order_oracle_cost

        # Current context
        self.current_context = None

        # Constants
        context_arg_dim = training_context_vec.shape[1]
        self.context_u_lb = context_u_lb.astype(float)
        self.context_u_ub = context_u_ub.astype(float)
        self.grad_lips_constant_cost = grad_lips_constant_cost
        if solver == 'cvxpy':
            self.solver_style = 'midgap-lp-cvxpy'
        elif solver == 'gurobi':
            self.solver_style = 'midgap-lp'
        else:
            self.solver_style = solver_style

        # Contextual optimizer object
        self.contextual_optimizer = cg.ContextOptProbNoH(
            first_order_oracle_cost, grad_lips_constant_cost,
            (training_context_u_vec, training_cost, training_cost_grad),
            context_u_lb, context_u_ub, context_arg_dim)

        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn,
            current_state=training_trajectory[-1, None],
            context_arg_dim=context_arg_dim, marker_type='x',
            marker_color='darkorange', method_name='C2Opt')

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Obtain the current context from the current state
        2. Get the decision for the current context
        3. Query the oracle and update the contextual_optimizer object
        4. Return the decision (numpy 2D matrix)
        """
        # Obtain the current context from the current state
        self.current_context = self.state_to_context(self.current_state)
        if verbose:
            print('Current Context: {:s}'.format(
                np.array_str(self.current_context, precision=2)))

        # Solve the contextual optimization problem
        res_iter = self.contextual_optimizer.solve(self.solver_style,
                self.current_context, verbose=verbose)

        # Query and update the contextual optimizer
        self.contextual_optimizer.query_and_update(res_iter['next_query'])

        # Return decision (numpy 2D matrix)
        return res_iter
