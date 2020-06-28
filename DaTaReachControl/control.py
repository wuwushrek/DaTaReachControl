import numpy as np
import copy
import time

from DaTaReachControl import fixpointGronwall, fixpointRecursive
from DaTaReachControl import FOverApprox, GOverApprox
from DaTaReachControl import Interval

import gurobipy as gp


def computeAffineLinear(x, dt, fOver, GOver, uRange, dtCoeff,
                        fixpointWidenCoeff, zeroDiameter, widenZeroInterval,
                        gronwall=True):
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
    if gronwall:
        Si = fixpointGronwall(x, dtCoeff, uRange, fOver, GOver,
                            vectField=fx+np.matmul(Gx, uRange))
    else:
        Si = fixpointRecursive(x, dt, uRange, fOver, GOver, fixpointWidenCoeff,
            zeroDiameter, widenZeroInterval)

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

def gurobiControlProblem(bl, Al, uRange, costFun, learningConstr):
    """Compute an initial gurobi model that finds the near optimal
    control value"""
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
    counterPb = 0
    dictConstr = dict()
    dictU = dict()
    listCost = []

    # Create a new model
    mOpt = gp.Model("noptimal")

    for d in listConstr:
        # Variables of the sub problems are u and v
        u = [mOpt.addVar(lb=-gp.GRB.INFINITY, name='u_{}_{}'.format(counterPb,i))\
                 for i in range(Al.shape[1])]
        x_var = [mOpt.addVar(lb=-gp.GRB.INFINITY, name='x_{}_{}'.format(counterPb,i))\
                     for i in range(Al.shape[0])]

        # Constraints on the variable x_var
        for i, xi in enumerate(x_var):
            coeffSup = list()
            coeffInf = list()
            for j in range(Al.shape[1]):
                if d[j]:
                    coeffSup.append(Al[i,j].ub)
                    coeffInf.append(Al[i,j].lb)
                else:
                    coeffSup.append(Al[i,j].lb)
                    coeffInf.append(Al[i,j].ub)

            # first component x <= sup(Au + b)
            # second component x >= inf(Au + b)
            # last component x = (sup(Au+b) + inf(Au+b))/2
            dictConstr[(counterPb,i)] = \
                (mOpt.addLConstr(gp.LinExpr(coeffSup + [-1], u + [xi]),
                        gp.GRB.GREATER_EQUAL, -bl[i,0].ub,
                        name='Up_{}_{}'.format(counterPb,i)),
                mOpt.addLConstr(gp.LinExpr(coeffInf + [-1], u + [xi]),
                        gp.GRB.LESS_EQUAL, -bl[i,0].lb,
                        name='Lo_{}_{}'.format(counterPb,i)),
                mOpt.addLConstr(gp.LinExpr(coeffSup + coeffInf + [-2],
                        u+u+[xi]), gp.GRB.EQUAL, -(bl[i,0].lb+bl[i,0].ub),
                        name='Mid_{}_{}'.format(counterPb,i)))

        # Constraints on the control variable u
        for j, uj in enumerate(u):
            if d[j]:
                uj.lb = np.maximum(0,uRange[j,0].lb)
                uj.ub = np.maximum(0,uRange[j,0].ub)
            else:
                uj.ub = np.minimum(0, uRange[j,0].ub)
                uj.lb = np.minimum(0, uRange[j,0].lb)

            # When learning the dynamics, encode some components are zero
            if learningConstr is not None:
                dictConstr[(counterPb,-j-1)] = \
                    mOpt.addConstr(gp.LinExpr([learningConstr[j]], [uj]),
                        gp.GRB.EQUAL, 0, name='B_{}_{}'.format(counterPb,j))

        # Save the intermediate variables
        dictU[counterPb] = (u, x_var, d)

        # Add the subproblem cost function
        listCost.append(costFun(x_var,u))
        counterPb = counterPb + 1
    # Objective is the sum of the different objectiv
    mOpt.setObjective(gp.quicksum(listCost))
    return dictU, dictConstr, listCost, mOpt

def solveGBproblem(bl, Al, learningConstr, dictU, dictConstr,
                    listCost, mOpt, useMs=False, verbose=True):
    """Given the values of Al, bl, the possible imposed learningConstr,
    compute a solution of the near-optimal control problem
    """
    for nbProblem, (u, x_var, d) in dictU.items():
        for i , x_i in enumerate(x_var):
            (c1, c2, c3) = dictConstr[(nbProblem,i)]
            c1.RHS = -bl[i,0].ub
            c2.RHS = -bl[i,0].lb
            if useMs:
                c3.RHS = -(bl[i,0].lb+bl[i,0].ub)
                mOpt.chgCoeff(c3, x_i, -2)
            else:
                c3.RHS = 0
                mOpt.chgCoeff(c3, x_i, 0)
            for j, uj in enumerate(u):
                if d[j]:
                    mOpt.chgCoeff(c1, uj, Al[i,j].ub)
                    mOpt.chgCoeff(c2, uj, Al[i,j].lb)
                else:
                    mOpt.chgCoeff(c1, uj, Al[i,j].lb)
                    mOpt.chgCoeff(c2, uj, Al[i,j].ub)
                if useMs:
                    mOpt.chgCoeff(c3, uj, Al[i,j].ub+Al[i,j].lb)
                else:
                    mOpt.chgCoeff(c3, uj, 0)
        # Constraints on the control variable u
        for j, uj in enumerate(u):
            if learningConstr is not None:
                c3 = dictConstr[(nbProblem,-j-1)]
                mOpt.chgCoeff(c3, uj, learningConstr[j])
    mOpt.Params.OutputFlag = verbose
    mOpt.optimize()
    costVal = np.array([obj.getValue() for obj in listCost])
    ind = np.argmin(costVal)
    return np.array([uVal.x for uVal in dictU[ind][0]]), costVal[ind]


class DaTaControl:
    """
    Main class for the systhesis of a 1-step "optimal" control in the sense
    defined in the paper for unknown dynamical systems based on a finite-horizon
    of a single trajectory

    """
    def __init__(self, cost_fun, uRange, lipF, lipG, traj={}, nDepf={}, bf={},
                bGf={}, knownFunF={}, learnLipF=False, Lfknown=None, nDepG={},
                bG={}, bGG={}, knownFunG={}, learnLipG=False, Lgknown=None,
                optVerb=False, solverVerb=True, solopts={}, probLearning=[],
                threshUpdateApprox=0.1, thresMeanTraj=1e-8, coeffLearning=0.1,
                minDataF=15 , maxDataF=25, minDataG=5, maxDataG=10, dt=0.01,
                gronwall=True, fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
                widenZeroInterval=1e-3):

        # Safe data limits for unknonwn f and unkonw G
        self.minDataF = minDataF
        self.maxDataF = maxDataF
        self.minDataG = minDataG
        self.maxDataG = maxDataG

        # Parametes for the recursive fixpoint computation
        self.fixpointWidenCoeff = fixpointWidenCoeff
        self.zeroDiameter = zeroDiameter
        self.widenZeroInterval = widenZeroInterval

        # Save the number of State and the number of Control
        self.nState = lipF.shape[0]
        self.nControl = lipG.shape[1]

        # Save if gronwall needs to be used or not
        self.gronwall = gronwall

        # Save the
        self.optVerb = optVerb
        self.solverVerb = solverVerb
        self.solopts = solopts

        # Update the threshold for updating over-approx
        self.threshUpdateApprox = threshUpdateApprox
        # Threshold for calling the optimization problem with mean value
        # This happen when the optimistic approach has a close to zero cost
        self.thresMeanTraj = thresMeanTraj
        # Whhen learning the dynamics just apply a ratio of the full range of u
        self.coeffLearning = coeffLearning
        # Label and prob of learning different components
        self.labLearning = np.array([-1]+[i for i in range(self.nControl)])
        self.probLearning = np.full(self.nControl+1, 1.0/(self.nControl+1)) \
                                if len(probLearning)==0 else probLearning

        # Delta time
        self.dt = dt
        # Range of the control u
        self.uRange = uRange
        # Variable for updating the over-approximations f and G
        self.indexUpdate = -1
        self.updateMeas = False

        # Update cost function and create underlying optimizations problems
        self.updateCost(cost_fun)

        # Create and update the over-approximating functions
        self.fover = FOverApprox(lipF, traj, nDepf, bf, bGf, knownFunF,
                        Lknown=Lfknown, learnLip=learnLipF, verbose=solverVerb)
        self.gover = GOverApprox(lipG, self.fover, traj, nDepG, bG, bGG,
                knownFunG, Lknown=Lgknown, learnLip= learnLipG, verbose=solverVerb)

        # Check if there's sufficient data to approximate f and G
        self.noInitData = not (self.fover.canApproximate(self.minDataF/2) \
                            and self.gover.canApproximate(self.minDataG/2))

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
        self.dictU, self.dictConstr, self.listCost, self.mOpt = \
            gurobiControlProblem(np.full((self.nState,1),Interval(0)),
                np.full((self.nState,self.nControl),Interval(0)),
                self.uRange, costFun, np.zeros(self.nControl))

    def updateLipOverapprox(self, lipF, lipG):
        """Update the Lipschitz bounds of the over-approximatiosn"""
        # Update Lipschitz of unknonw f
        self.fover.updateLip(lipF)
        # Update Lipschitz of unknonw G
        self.gover.updateLip(lipG)
        # Update gronwall coeff
        self.updateScalingGronwall()

    def updateScalingGronwall(self):
        """Compute the coefficient coeffDt needed to returning the
        a priori enclosure by gronwall theorem"""
        uAbs = np.abs(self.uRange)
        uSup = np.array([[uAbs[i,0].ub] for i in range(uAbs.shape[0])])
        beta_dt = np.linalg.norm(self.fover.Lf + np.matmul(self.gover.LG , uSup))
        self.coeffDt = (self.dt/(1-np.sqrt(self.nState)*self.dt*beta_dt))

    def noInitialData(self):
        if not self.fover.canApproximate(minSize=self.minDataF/2):
            self.currentU = np.zeros((self.nControl,1))
            self.indexUpdate = -1
            return
        if not (self.fover.canApproximate(minSize=self.minDataF/2) and \
                    self.gover.canApproximate(minSize=self.minDataG/2)):
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
        self.learningConstr = np.full(self.nControl, 1)
        self.indexUpdate = np.random.choice(self.labLearning, p=self.probLearning)
        if self.indexUpdate != -1:
            self.learningConstr[self.indexUpdate] = 0

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
                self.gover, self.uRange, self.coeffDt, self.fixpointWidenCoeff,
                self.zeroDiameter, self.widenZeroInterval,gronwall=self.gronwall)

        # No addditional constraint on the control
        self.learningConstr = np.zeros(self.nControl)

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

        # Synthesize control
        uOpt, optCost = solveGBproblem(bj, Aj, self.learningConstr, self.dictU,
                                        self.dictConstr, self.listCost,
                                        self.mOpt, verbose=self.optVerb)

        # Use the mean trajectory when we are very close to the target
        if optCost < self.thresMeanTraj:
            uOpt, optCost = solveGBproblem(bj, Aj, self.learningConstr, self.dictU,
                            self.dictConstr, self.listCost, self.mOpt,
                            verbose=self.optVerb, useMs=True)

        if self.solverVerb:
            print('Optimal cost : ', optCost)
            print('Mean trajectory? ', optCost < self.thresMeanTraj)
            print('Update over-approximations: ', self.updateMeas, self.indexUpdate)
            print('State: ', currX.flatten())

        # Do some logging
        self.currentX = currX
        self.currentU = uOpt.reshape(-1,1)
        self.nextStateOverApprox =  bj + np.matmul(Aj, self.currentU)

        if self.solverVerb:
            print('Next state overapprox: ', self.nextStateOverApprox.flatten())

        return self.currentU
