import numpy as np
import cvxpy as cp
from .MyopicDataDrivenControl import MyopicDataDrivenControl
import time

def defaultLib(state):
    return (state, state ** 2, state ** 3, state ** 4, state ** 5, state ** 6,
                np.sin(state), np.cos(state))

class MyopicDataDrivenControlSINDYc(MyopicDataDrivenControl):
    """
    Use SINDYc for system identification followed by one-step optimal control
    """

    def __init__(self, training_data, input_lb, input_ub, cvxpy_objective,
            sampling_time, one_step_dyn=None, exit_condition=None,
            cvxpy_args=None, sparse_thresh=1e-3, eps_thresh=1e-12,
            scaling_sparse_min=1e5, scaling_sparse_max = 1e-6,
            libraryFun=defaultLib):
        """
        self.trajectory excludes the current state
        """

        # Unpack training data
        self.trajectory = training_data['trajectory'][:-1, :]
        self.input_seq = training_data['input_seq']
        self.cost_val_vec = training_data['cost_val']

        # Bounds
        self.input_lb = input_lb
        self.input_ub = input_ub

        # Functions
        self.cvxpy_objective = cvxpy_objective

        MyopicDataDrivenControl.__init__(self, exit_condition=exit_condition,
            one_step_dyn=one_step_dyn, current_state=self.trajectory[-1, None],
            marker_type='v', marker_color='m', method_name='SINDYc')

        # Uses sparse_thresh to declare if a coeff is zero or non-zero
        self.sparse_thresh = sparse_thresh
        self.eps_thresh = eps_thresh            # Round problem coefficients to
                                            # zero if below eps_thresh
        if cvxpy_args is None:
            self.cvxpy_args = {'solver': 'ECOS'}
        else:
            self.cvxpy_args = cvxpy_args
        self.sampling_time = sampling_time
        # Sweep the scaling sparse parameter from max to min and quit on
        # feasibility
        self.scaling_sparse_max = scaling_sparse_max
        self.scaling_sparse_min = scaling_sparse_min
        # Save the library function
        self.libraryFun = libraryFun

    def compute_decision_for_current_state(self, verbose=False):
        """
        1. Get the decision for the current state
            a. Get the SINDYc coefficients for the dynamics
            b. Do the computation
        2. Use scipy.optimize to compute the best control action
        3. Return the decision
        """
        if verbose:
            print('Current state: {:s}'.format(np.array_str(
                self.current_state, precision=2)))

        # Compute the decision
        query_timer_start = time.time()

        # Compute the SINDYc coefficients for the dynamics
        coeff_state, coeff_input = self.compute_sindyc_coeff()

        # Compute the sublibrary for dynamics
        L_x = self.evaluate_sub_library_at_state(self.current_state)

        # One-step optimal control: Solve
        # minimize       objective(delta_state, current_action)
        # subject to     delta_state = Model(current_action)
        #                lb <= u <= ub
        # for delta_state and current_action
        #
        # Here, the estimated next state is given by current_state + delta_state
        # under the influence of current_action
        current_action = cp.Variable((self.input_seq.shape[1],))
        delta_state = cp.Variable((self.trajectory.shape[1],))
        L_input = cp.hstack((current_action[input_indx] * L_x for input_indx
                             in range(self.input_seq.shape[1])))
        const = [current_action <= self.input_ub,
                 current_action >= self.input_lb,
                 delta_state == coeff_state.T@L_x + coeff_input.T@L_input]
        obj = self.cvxpy_objective(self.current_state, delta_state,
                                   current_action)

        prob = cp.Problem(cp.Minimize(obj), const)

        try:
            prob.solve(**self.cvxpy_args)
        except cp.SolverError:
            raise RuntimeError('CVXPY solver error')

        if prob.status in ['optimal', 'optimal_inaccurate']:
            current_decision = np.array([current_action.value])
        else:
            raise RuntimeError('compute_decision_for_current_state failed'
                    ' with CVXPY status: {:s}'.format(prob.status))

        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start

        if verbose:
            # Update user
            print('Best action = {:s} | Estimated cost = {:1.4f} | Time = '
                  '{:1.4f} s '.format(np.array_str(current_decision,
                                                   precision=2),
                      prob.value, query_time))

        # # Get the next state
        # self.current_state = self.one_step_dyn(self.current_state,
        #                                        current_decision)

        # Update the data matrices
        self.trajectory = np.vstack((self.trajectory, self.current_state))
        self.input_seq = np.vstack((self.input_seq, current_decision))
        self.cost_val_vec = np.hstack((self.cost_val_vec, prob.value))

        # Return decision
        solution_dict = {'next_query': current_decision,
                         'lb_opt_val': prob.value,
                         'query_time': query_time}
        return solution_dict

    def compute_sindyc_coeff(self, verbose=False):
        """
        Given history, perform SINDYc

        We vary scaling_sparse variable between sparse_thresh_min and
        sparse_thresh_max, starting at sparse_thresh_max and in multiples of
        1/10.
        """
        # Dimension of the system
        n_dim = self.trajectory.shape[1]

        # Obtain the library
        L, delta_x_plus, n_L_x = self.get_library_and_delta_x()

        coeff = np.zeros((L.shape[1], n_dim))
        scaling_sparse = self.scaling_sparse_max

        while True:
            for indx in range(n_dim):
                # Compute the sparse coeff at scaling_sparse
                coeff[:, indx:indx+1] = self.compute_sparse_coeff_1D(L,
                    delta_x_plus[:,indx:indx+1], self.cvxpy_args,
                    self.eps_thresh, scaling_sparse)

            # Check if at least one dimension has all coeff=0 according to
            # sparsity_thresh
            coeff_invalid = any(np.max(coeff, axis=0) <= self.sparse_thresh)

            if coeff_invalid:
                # Reduce scaling_sparse with the hope of better fit
                scaling_sparse = scaling_sparse/10
                if scaling_sparse < self.scaling_sparse_min:
                    # Quit
                    raise RuntimeError('Scaling_sparse required below minimum '
                        '%1.0e' % self.scaling_sparse_min)
            else:
                break
        coeff[abs(coeff) < self.eps_thresh] = 0
        coeff_state = coeff[:n_L_x, :]    # M x n
        coeff_input = coeff[n_L_x:, :]
        return coeff_state, coeff_input

    def get_library_and_delta_x(self):
        """
        - Evaluate the library at all time snapshots so far
        - Once L_x is obtained from evaluate_sub_library_at_state, we define the
          library as [L_x, u * L_x, w * L_x]. Recall that L_x has a
          sampling_time factor.
        - This library is meant to model x_{t+1} - x_t.
        """
        L = []
        for state, control in zip(self.trajectory, self.input_seq):
            L_x = self.evaluate_sub_library_at_state(np.array([state]))
            L.append(np.hstack((L_x, np.kron(control, L_x))))

        delta_x_plus_vec = \
            np.vstack((self.trajectory[1:,:], self.current_state)) \
            - self.trajectory
        delta_x_plus_vec[abs(delta_x_plus_vec) < self.eps_thresh] = 0
        return np.array(L), delta_x_plus_vec, L_x.shape[0]

    def evaluate_sub_library_at_state(self, state):
        """
        Evaluate the sublibrary at the given state. The actual library is
        computed by get_library_and_delta_x
        """

        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0, :]
        else:
            print(state)
            raise ValueError('Expected 2D state row vector')

        # Change this to change
        L_x = np.vstack(self.libraryFun(state))
        # Repeat L_x for each of the state
        L_x = L_x.flatten('F')
        # Add constant bias term
        return np.hstack((1, L_x)) * self.sampling_time

    @staticmethod
    def compute_sparse_coeff_1D(L, delta_x_plus_vec_1D, cvxpy_args, eps_thresh,
                                scaling_sparse):
        """
        Given data (x_t, u_t, x_{t+1} - x_{t}) for a component of the state, we
        solve the following optimization problem for w,

        minimize ||delta_x_plus_vec - L@w||_2 + scaling_sparse * ||w||_1

        where
            - each row of delta_x_plus_vec_1D is (x_{t+1} - x_t), with 1D
              referring to fact that it is a component of x, and NOT the whole x
            - L is the library computed via get_library_and_delta_x, a function
              of x_t and u_t, with L@w
            - scaling_sparse is the weighting of the objectives (sparsity vs fit
              quality)
        """

        if delta_x_plus_vec_1D.shape[1] != 1:
            raise ValueError('Expected delta_x_plus_vec to be a column matrix')

        L[abs(L) <= eps_thresh] = 0
        delta_x_plus_vec_1D[abs(delta_x_plus_vec_1D) <= eps_thresh] = 0
        w = cp.Variable((L.shape[1], 1))
        obj = scaling_sparse*cp.norm1(w) + cp.norm2(delta_x_plus_vec_1D-L@w)
        prob = cp.Problem(cp.Minimize(obj))

        try:
            prob.solve(**cvxpy_args, verbose=False)
        except cp.SolverError as e:
            print(e)
            raise RuntimeError('CVXPY solver error in compute_sparse_coeff')

        if prob.status in ['optimal', 'optimal_inaccurate']:
            if prob.status == 'optimal_inaccurate':
                print('CVXPY returned optimal_inaccurate!')
            return w.value
        else:
            raise RuntimeError('compute_sparse_coeff failed with CVXPY status:'
                    ' {:s}'.format(prob.status))

