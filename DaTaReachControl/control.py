import numpy as np
import copy
import time

from DaTaReachControl import fixpointGronwall, fixpointRecursive
from DaTaReachControl import FOverApprox, GOverApprox
from DaTaReachControl import Interval

import cvxpy as cp

def computeBounds(bj, Aj):
    """
    Compute the lower and upper bounds on both bj and Aj. Noe that bj and Aj
    are repectively one dimensional and two dimensional arrays.
    """
    b_lb = np.zeros(bj.shape)
    b_ub = np.zeros(bj.shape)
    A_lb = np.zeros(Aj.shape)
    A_ub = np.zeros(Aj.shape)
    for i in range(bj.shape[0]):
        b_lb[i,0] = bj[i,0].lb
        b_ub[i,0] = bj[i,0].ub
        for j in range(Aj.shape[1]):
            A_lb[i,j] = Aj[i,j].lb
            A_ub[i,j] = Aj[i,j].ub
    return b_lb, b_ub, A_lb, A_ub


def computeAffineLinear(x, dt, fOver, GOver, uRange, dtCoeff):
    """From the given current state x, compute the interval coefficients
    bj and Aj such that x(t+dt) in bj + Aj * u for any control u constant
    between time t and t+dt. See paper for specific details

    Parameters
    ----------
    :param x : A box containing the current state
    :param dt : The sampling time or integration step
    :param fOver : Over-approximation of the unknown function f
    :param GOver : over-approximation of the unknown function G
    :param uRange : The domain set of teh function u
    :param dtCoeff : The gronwall coefficient for finding apriori enclosure

    Returns
    -------
    The interval vector bi and the interval matrix Ai (Eq 23 paper)
    """
    dt_2 = 0.5*dt**2
    # COmpute f and G at the current over-approximation x
    fx = fOver(x)
    Gx = GOver(x)

    # Get the a priori enclosure
    Si = fixpointGronwall(x, dtCoeff, uRange, fOver, GOver,
                            vectField=fx+np.matmul(Gx, uRange))

    # Compute f and G at the a priori enclosure
    fSi = fOver(Si)
    GSi = GOver(Si)

    # Compute the different Jacobian
    Jf = fOver.Jf(x)
    JG = GOver.JG(x)

    # Compute Bi
    bi = x + fx * dt + np.matmul(Jf, fSi) * dt_2

    # Compute Ai
    JG_t = np.transpose(JG,(0,2,1))
    Ai = Gx*dt + (np.tensordot(JG_t, fSi, axes=([1,0]))[:,:,0] + \
                    np.matmul(np.tensordot(JG, uRange, axes=([1,0]))[:,:,0]+Jf,
                    GSi))* dt_2
    return bi, Ai

def computeOptimisticProblem(Al, Au, bl, bu, uRange, costFun, learningConstr=None):
    """Generate the 2^q convex optimization problems that need to be solve
    to find the optimal control.
    q is the number of control variable allowing positive and negative
    values, e.g., the number of component of the control having 0 in its
    range of value.
    """
    # Create a dictionary of constrainst with their constraints
    # Depending on the range of u. we have 2^q problem to solve
    # where q is the number of components admitting negative and positive values
    listConstr = [dict()]
    for i in range(Al.shape[1]):
        if uRange[i,0].lb >= 0 or uRange[i,0].ub <= 0:
            for d in listConstr:
                d[i] = uRange[i,0].lb >= 0
            continue
        list_p = copy.deepcopy(listConstr)
        for d in list_p:
            d[i] = True
        list_n = copy.deepcopy(listConstr)
        for d in list_n:
            d[i] = False
        listConstr = list_p + list_n

    # Create the optimization problem
    listConstrOpt = []
    listConstrMid = []
    counterPb = 0
    dictU = dict()
    listCost = []
    for d in listConstr:
        u = cp.Variable(Al.shape[1])
        x_var = cp.Variable(Al.shape[0])
        # Constraints on teh variable x_var
        for i in range(Al.shape[0]):
            x_min = bl[i,0]
            x_max = bu[i,0]
            for j in range(Al.shape[1]):
                if d[j]:
                    x_max += Au[i,j] * u[j]
                    x_min += Al[i,j] * u[j]
                else:
                    x_max += Al[i,j] * u[j]
                    x_min += Au[i,j] * u[j]
            listConstrOpt.append(x_var[i] <= x_max)
            listConstrOpt.append(x_var[i] >= x_min)
            listConstrMid.append(x_var[i] == 0.5*(x_max+x_min))
        # Constraints on the control variable u
        for j in range(Al.shape[1]):
            if d[j]:
                listConstrOpt.append(u[j] >= np.maximum(0,uRange[j,0].lb))
                listConstrMid.append(u[j] >= np.maximum(0,uRange[j,0].lb))
                listConstrOpt.append(u[j] <= np.maximum(0,uRange[j,0].ub))
                listConstrMid.append(u[j] <= np.maximum(0,uRange[j,0].ub))
            else:
                listConstrOpt.append(u[j] <= np.minimum(0, uRange[j,0].ub))
                listConstrMid.append(u[j] <= np.minimum(0, uRange[j,0].ub))
                listConstrOpt.append(u[j] >= np.minimum(0, uRange[j,0].lb))
                listConstrMid.append(u[j] >= np.minimum(0, uRange[j,0].lb))
            if learningConstr is not None:
                listConstrOpt.append(u[j] * learningConstr[j] == 0)
                listConstrMid.append(u[j] * learningConstr[j] == 0)
        dictU[counterPb] = u
        listCost.append(costFun(x_var,u))
        counterPb = counterPb + 1
    pbOpt = cp.Problem(cp.Minimize(sum(listCost)), listConstrOpt)
    pbMid = cp.Problem(cp.Minimize(sum(listCost)), listConstrMid)
    return dictU, listCost, pbOpt, pbMid

def solveProblems(problem, u_var, cost_var, solver=cp.GUROBI, verbose=True,
                    warm_start=True, solopts=dict()):
    """ Solve the 2^q problem and return the problem that
    has the minimum possivle cost.
    """
    # Solve the problems
    t = time.time()
    problem.solve(solver=solver, verbose=verbose,warm_start=warm_start,
                    **solopts)
    print(time.time()-t)
    # Get the minimum possible cost amongst the different problems
    costValue = np.array([cost.value for cost in cost_var])
    ind = np.argmin(costValue)
    return u_var[ind].value, costValue[ind]

class DaTaControl:
    """
    Main class for the systhesis of a 1-step "optimal" control in the sense
    defined in the paper for unknown dynamical systems based on a finite-horizon
    of a single trajectory

    """
    def __init__(self, cost_fun, uRange, lipF, lipG, traj={}, nDepf={}, bf={}, bGf={},
                knownFunF={}, nDepG={}, bG={}, bGG={}, knownFunG={}, optVerb=False,
                solverVerb=True, solver=cp.GUROBI, solopts={}, probLearning=[],
                threshUpdateApprox=0.1, thresMeanTraj=1e-8, coeffLearning=0.1,
                minDataF=15 , maxDataF=25, minDataG=5, maxDataG=10, dt=0.01):

        self.minDataF = minDataF
        self.maxDataF = maxDataF
        self.minDataG = minDataG
        self.maxDataG = maxDataG
        self.nState = lipF.shape[0]
        self.nControl = lipG.shape[1]

        # self
        # self.current_x = np.copy(initXtraj[:,-1])
        # self.current_u = np.copy(uSeq[:,-1])

        self.optVerb = optVerb
        self.solverVerb = solverVerb
        self.solver = solver
        self.solopts = solopts

        self.threshUpdateApprox = threshUpdateApprox
        self.thresMeanTraj = thresMeanTraj
        self.coeffLearning = coeffLearning
        self.labLearning = np.array([-1]+[i for i in range(self.nControl)])
        self.probLearning = np.full(self.nControl+1, 1.0/(self.nControl+1)) \
                                if len(probLearning)==0 else probLearning

        self.dt = dt
        self.uRange = uRange
        self.indexUpdate = -1
        self.updateMeas = False

        # Update cost function and create underlying optimizations problems
        self.updateCost(cost_fun)

        # Create and update the over-approximating functions
        self.fover = FOverApprox(lipF, traj, nDepf, bf, bGf, knownFunF)
        self.gover = GOverApprox(lipG, self.fover, traj, nDepG, bG, bGG, knownFunG)
        self.updateLipOverapprox(lipF, lipG)
        self.noInitData = not (self.fover.canApproximate() and self.gover.canApproximate())

        # Update the cost scaling coefficient of gronwall
        self.updateScalingGronwall()

        self.nextStateOverApprox = None
        self.currentX = None
        self.currentU = None

    def updateCost(self, costFun):
        """
        Create the optimizations problems that are going to be used for
        the synthesis of a controller
        """
        self.Al = cp.Parameter((self.nState, self.nControl))
        self.Au = cp.Parameter((self.nState, self.nControl))
        self.bl = cp.Parameter((self.nState,1))
        self.bu = cp.Parameter((self.nState,1))
        self.learningConstr = cp.Parameter(self.nControl)
        self.dictU, self.listCost, self.pbOpt, self.pbMid = \
                computeOptimisticProblem(self.Al, self.Au, self.bl, self.bu,
                                            self.uRange, costFun,
                                            self.learningConstr)

    def updateLipOverapprox(self, lipF, lipG):
        """Update the Lipschite bounds of the over-approximatiosn"""
        self.fover.Lf = lipF
        self.gover.LG = lipG
        self.updateScalingGronwall()

    def updateScalingGronwall(self):
        """Compute the coefficient coeffDt needed to returning the
        a priori enclosure by gronwall theorem"""
        uAbs = np.abs(self.uRange)
        uSup = np.array([[uAbs[i,0].ub] for i in range(uAbs.shape[0])])
        beta_dt = np.linalg.norm(self.fover.Lf + np.matmul(self.gover.LG , uSup))
        self.coeffDt = (self.dt/(1-np.sqrt(self.nState)*self.dt*beta_dt))

    def noInitialData(self):
        if not (self.fover.canApproximate() and self.gover.canApproximate()):
            self.indexUpdate = np.random.choice(self.labLearning,
                                    p=self.probLearning)
            self.currentU = np.zeros((self.nControl,1))
            if self.indexUpdate != -1:
                while True:
                    self.currentU[self.indexUpdate,0] = self.coeffLearning * \
                            np.random.uniform(self.uRange[self.indexUpdate,0].lb,
                                              self.uRange[self.indexUpdate,0].ub)
                    if not (np.abs(self.currentU[self.indexUpdate,0]) < 1e-8):
                        break
        else:
            self.noInitData = False

    def shouldUpdateTraj(self, nextState):
        """
        Check if the over-approxmations of f and G need to be updated. In case
        no data has been given, we need to learn the dynamics locally
        """
        if self.nextStateOverApprox is None:
            return False
        distVal = np.linalg.norm(self.nextStateOverApprox-nextState)
        return (distVal.ub-distVal.lb) > self.threshUpdateApprox

    def synthControlUpdate(self):
        """
        When updating over-approximations of f and G, impose constraints
        on the optimal control value such that it's either 0 or all
        components are equal to 0 except from one.
        """
        self.learningConstr.value = np.full(self.nControl, 1)
        self.indexUpdate = np.random.choice(self.labLearning, p=self.probLearning)
        if self.indexUpdate != -1:
            self.learningConstr.value[self.indexUpdate] = 0

    def updateOverapprox(self, xVal, xDotVal, uVal, currTime=None):
        """
        Update our over-approximation of f and G when synth_control_update
        has been executed. The update are done
        when the over-approximation diameter is large than the threshold
        threshUpdateApprox
        """
        if self.indexUpdate == -1:
            self.fover.update(xVal, xDotVal, uVal, currTime)
            self.fover.removeData(xVal, self.minDataF, self.maxDataF)
        else:
            self.gover.update(xVal, xDotVal, uVal, currTime)
            self.gover.removeData(xVal, self.minDataG, self.maxDataG)

    def __call__(self, currX, currXdot,t=None):
        """
        Function to call when trying to obtain the one-step optimal control
        to minimize the distance to the setpoint. This function is assumed
        to be called every dt
        """
        currXInt = np.array([[Interval(currX[i,0])] for i in range(self.nState)])

        # print ('No init Data : ', self.noInitData)
        if self.noInitData:
            if self.currentX is not None:
                self.updateOverapprox(self.currentX, currXdot, self.currentU,
                                        None if t is None else t-self.dt)
            self.currentX = currX
            self.noInitialData()
            return self.currentU

        # Check if an update on f and G was done on the previous step
        if self.updateMeas:
            self.updateOverapprox(self.currentX, currXdot, self.currentU)

        # Compute interval coefficients bj and Aj
        bj, Aj = computeAffineLinear(currXInt, self.dt, self.fover,
                            self.gover, self.uRange, self.coeffDt)

        # Compute the lower bounds and upper bounds from bj and Aj
        self.bl.value, self.bu.value, self.Al.value, self.Au.value = \
                computeBounds(bj, Aj)

        # No addditional constraint on the control
        self.learningConstr.value = np.zeros(self.nControl)

        # Check if we need to update f and G
        self.updateMeas = self.shouldUpdateTraj(currX)

        # Do some printing if required
        if self.solverVerb:
            nDataPoint = self.fover.E0x.shape[1] +\
                sum([val[0].shape[1] for x, val in self.gover.Ej.items()])
            print('No. of data points: ', nDataPoint)

        # Synthesis of the controller
        if self.updateMeas:
            self.synthControlUpdate()
            self.currentX = currX

        # Synthesize control
        uOpt, optCost = solveProblems(self.pbOpt, self.dictU,
                            self.listCost, solver=self.solver,
                            verbose=self.optVerb, warm_start=True,
                            solopts=self.solopts)

        # Use the mean trajectory when we are very close to the target
        if optCost < self.thresMeanTraj:
            uOpt, optCost = solveProblems(self.pbMid, self.dictU,
                            self.listCost, solver=self.solver,
                            verbose=self.optVerb, warm_start=True,
                            solopts=self.solopts)

        if self.solverVerb:
            print('Optimal cost : ', optCost)
            print('Mean trajectory? ', optCost < self.thresMeanTraj)
            print('Update over-approximations: ', self.updateMeas, self.indexUpdate)

        # Do some logging
        self.currentU = uOpt.reshape(-1,1)
        self.nextStateOverApprox =  bj + np.matmul(Aj, self.currentU)

        if self.solverVerb:
            print('Next state overapprox: ', self.nextStateOverApprox.flatten())

        return self.currentU
