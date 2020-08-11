import numpy as np
from numpy import float64 as realN

# Import the optimizations tools
import optimisticControlGrb as optGrb
import weightedPointControlGRb as midGrb
import weightedPointControlPGDAR as midPGA

# Import the necessary function from reachN
from intervalN import norm_i, add_i, mul_Ms_i
from reachN import initOverApprox, fover, Gover, update, nextStateOverApprox,\
                    controlAffineOverApprox, canApproximate, getCoeffGronwall

# Import the necessary types
from reachN import depTypebf, depTypebG
from overapprox_functionsN import depTypeG, depTypeGradG, depTypeF, depTypeGradF

# The different Optimization method
OPTIMISTIC_GRB = 0 # optimistic problem using Gurobi
MIDPOINT_GRB = 1 # Midpoint problem using Gurobi
MIDPOINT_APG = 2 # Midpoint problem using Approximated proximal gradient with restart scheme

class DaTaControlN:
    """
    Main class for the systhesis of a 1-step "optimal" control in the sense
    defined in the paper for unknown dynamical systems based on a finite-horizon
    of a single trajectory

    """
    def __init__(self, dt, Lf, LG, U_lb, U_ub, Q=None, S=None, R=None, q=None, r=None,
        Lfknown=None, LGknown=None, nvDepF=depTypeF, nvDepG=depTypeG, bf=depTypebf,
        bG =depTypebG , bGf = depTypeGradF, bGG=depTypeGradG, xTraj=None,
        xDotTraj = None, uTraj = None, useGronwall=False, verbOverApprox=False,
        knownf=None, knownG=None, gradKnownf=None, gradKnownG=None,
        fixpointWidenCoeff=0.2, zeroDiameter=1e-5, widenZeroInterval=1e-3,
        maxData=20, verbSolver=False, verbCtrl=False, threshUpdateApprox=0.1,
        threshMeanTraj=1e-8, coeffLearning=0.1, probLearning=[], params=None):

        # Build the Overaproximation model
        self.overApprox = initOverApprox(Lf, LG, Lfknown, LGknown, nvDepF,
            nvDepG, bf, bG, bGf, bGG, xTraj, xDotTraj, uTraj, useGronwall,
            verbOverApprox, knownf, knownG, fixpointWidenCoeff, zeroDiameter,
            widenZeroInterval, maxData)

        # Save the known parts of the dynamic
        self.knownf = knownf
        self.knownG =  knownG
        self.gradKnownf = gradKnownf
        self.gradKnownG = gradKnownG


        # Save the verbose parameters
        self.optVerb = verbSolver
        self.ctrlVerb = verbCtrl

        # Update the threshold for updating over-approx
        self.threshUpdateApprox = threshUpdateApprox
        # Threshold for calling the optimization problem with mean value
        # This happen when the optimistic approach has a close to zero cost
        self.thresMeanTraj = threshMeanTraj
        # Whhen learning the dynamics just apply a ratio of the full range of u
        self.coeffLearning = coeffLearning
        # Label and prob of learning different components
        self.labLearning = np.array([-1]+[i for i in range(self.overApprox.nC)])
        self.probLearning = np.full(self.overApprox.nC+1, 1.0/(self.overApprox.nC+1)) \
                                if len(probLearning)==0 else probLearning

        # Delta time
        self.dt = dt

        # Range of the control u
        self.U_lb = U_lb
        self.U_ub = U_ub

        # Variable for updating the over-approximations f and G
        self.indexUpdate = -2
        self.updateMeas = False

        # Select which optimization tool to use
        self.initializeOptimizer(params)

        # Update cost function and create underlying optimizations problems
        self.updateCost(Q, q, R, S, r)

        # Check if there's sufficient data to approximate f and G
        self.canDoApprox = canApproximate(self.overApprox)

        # Update the scaling coefficient of gronwall
        self.updateScalingGronwall()

        # Some temporarty variable in the problem
        self.nextStateOverApprox_lb = None
        self.nextStateOverApprox_ub = None
        self.currentX = None
        self.currentU = None

    def initializeOptimizer(self, params):
        if params is None:
            params = (MIDPOINT_APG, 0.7, 0.7, 0.7, 1e-8)
        self.method = params[0]
        if params[0] == OPTIMISTIC_GRB:
            optGrb.initOptimisticProblem(self.overApprox.nS, self.overApprox.nC,
                None, None, None, None, None, self.U_lb, self.U_ub)
            self.optSolve = optGrb.solveOptimisticProblem
            self.extra_params = (self.optVerb,)
        elif params[0] == MIDPOINT_GRB:
            midGrb.initMidpointProblem(self.U_lb, self.U_ub)
            self.extra_params = (*params[1:], self.optVerb)
            self.optSolve = midGrb.solveWeightedProblem
        elif params[0] == MIDPOINT_APG:
            self.extra_params = params[1:]
            self.optSolve = midPGA.solveWeightedProblemAPGDAR


    def updateCost(self, Q, S, R, q, r):
        """
        Initialize the optimizations problems that are going to be used for
        the synthesis of a controller
        """
        # Save the target function
        if Q is None:
            Q = np.zeros((self.overApprox.nS, self.overApprox.nS), dtype=np.float64)
        if R is None:
            R = np.zeros((self.overApprox.nC, self.overApprox.nC), dtype=np.float64)
        if S is None:
            S = np.zeros((self.overApprox.nS, self.overApprox.nC), dtype=np.float64)
        if q is None:
            q = np.zeros(self.overApprox.nS, dtype=np.float64)
        if r is None:
            r = np.zeros(self.overApprox.nC, dtype=np.float64)
        if self.method == OPTIMISTIC_GRB:
            optGrb.updateCost(Q, S, R, q, r)
            self.solve_params = self.extra_params
        elif self.method == MIDPOINT_GRB or self.method == MIDPOINT_APG:
            self.solve_params = (Q, S, R, q, r, *self.extra_params)


    def updateScalingGronwall(self):
        """Compute the coefficient coeffDt needed to returning the
        a priori enclosure by gronwall theorem"""
        self.betaValCoeff = getCoeffGronwall(self.overApprox, self.dt, self.U_lb,
                                        self.U_ub)

    def noInitialData(self):
        """ When there's no initial data, perform control to generate
            useful point for over-approximation of f and G
            This routine assumes CONTROL VALLUES OF 0 can be applied
        """
        canApproxf, canApproxG = canApproximate(self.overApprox)
        if not canApproxf:
            self.currentU = np.zeros(self.overApprox.nC, dtype=realN)
            self.indexUpdate = -1
            return
        if not (canApproxf and canApproxG):
            self.indexUpdate = np.random.choice(self.labLearning,
                                    p=self.probLearning)
            self.currentU = np.zeros(self.overApprox.nC, dtype=realN)
            if self.indexUpdate != -1:
                while True:
                    self.currentU[self.indexUpdate] = self.coeffLearning * \
                            np.random.uniform(self.U_lb[self.indexUpdate],
                                              self.U_ub[self.indexUpdate])
                    if not (np.abs(self.currentU[self.indexUpdate]) < 1e-8):
                        break
        else:
            self.indexUpdate = -2
            self.canDoApprox = (True, True)

    def shouldUpdateTraj(self, nextState):
        """
        Check if the over-approxmations of f and G need to be updated. That's
        done by comparing the over-approximation at the next time-step given the
        synthesized control and the tre state and the next time step that will
        be received
        """
        if self.nextStateOverApprox_lb is None:
            return False
        d_lb, d_ub =norm_i(self.nextStateOverApprox_lb-nextState,
                        self.nextStateOverApprox_ub-nextState)
        return (d_ub-d_lb) > self.threshUpdateApprox

    def synthControlUpdate(self):
        """
        When updating over-approximations of f and G, impose constraints
        on the optimal control value such that it's either 0 or all
        components are equal to 0 except from one.
        """
        self.indexUpdate = np.random.choice(self.labLearning, p=self.probLearning)

    def __call__(self, currX, currXdot):
        """
        Function to call when trying to obtain the one-step optimal control
        to minimize the distance to the setpoint. This function is assumed
        to be called every dt
        """

        # print ('No init Data : ', self.noInitData)
        if not (self.canDoApprox[0] and self.canDoApprox[1]):
            if self.currentX is not None:
                update(self.overApprox, self.currentX, currXdot, self.currentU,
                        knownf=self.knownf, knownG=self.knownG)
            self.currentX = currX
            self.noInitialData()
            return self.currentU

        # Update f and G with any new given data points
        if self.currentX is not None:
            update(self.overApprox, self.currentX, currXdot, self.currentU,
                    knownf=self.knownf, knownG=self.knownG)

        # Check if we need to update f and G
        self.updateMeas = self.shouldUpdateTraj(currX)

        # Do some printing if required
        if self.ctrlVerb:
            nDataPoint = self.overApprox.nbData
            print('No. of data points: ', nDataPoint)

        # Set the index to update to no excitation based control
        self.indexUpdate = -2

        # Synthesis of the controller
        if self.updateMeas:
            print('Update Next')
            self.synthControlUpdate()

        b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub = \
                controlAffineOverApprox(self.overApprox, currX, self.dt,
                    self.U_lb, self.U_ub, knownf=self.knownf, knownG=self.knownG,
                    gradKnownf=self.gradKnownf, gradKnownG=self.gradKnownG,
                    gronwallCoeff=self.betaValCoeff)

        # Synthesize control
        # uOpt, optCost = self.optSolve(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
        #         learnConstr=self.learningConstr, verbose = self.optVerb,
        #         w1=self.weight1, w2=self.weight2, w3=self.weight3)
        uOpt, optCost = self.optSolve(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
                self.U_lb, self.U_ub, self.indexUpdate, *self.solve_params)
        if self.ctrlVerb:
            print('Approximate optimal cost : ', optCost)
            # print('Mean trajectory? ', optCost < self.thresMeanTraj)
            print('Update over-approximations: ', self.updateMeas, self.indexUpdate)
            print('State: ', currX)

        # Do some logging
        self.currentX = currX
        self.currentU = uOpt
        self.nextStateOverApprox_lb, self.nextStateOverApprox_ub = \
            nextStateOverApprox(b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub, uOpt)

        # print(self.nextStateOverApprox_lb)
        # print(self.nextStateOverApprox_ub)

        if self.ctrlVerb:
            print('Next state overapprox:')
            print(self.nextStateOverApprox_lb)
            print(self.nextStateOverApprox_ub)

        return self.currentU
