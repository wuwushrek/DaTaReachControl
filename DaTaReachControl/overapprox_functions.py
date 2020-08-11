import numpy as np
from .interval import Interval

# TODO: parrallelization of loop with, e.g, Numba
class FOverApprox:
    """Compute an over-approximation of the unknown function f
    given the separation E_0, side information, and the upper bounds on the
    Lispchitz constant of the unknown function f

    Parameters
    ----------
    :param traj: The data being used to over-approximate f
    :param Lf: The upper bound on the Lipschitz constants of f
    :param vDep: A dictionary where the value of a key k gives the indexes
                 of the states for which f_k is dependent
    :param bf: A dictionary specifying side information tighter bounds/range
                 for each component f_k
    :param bGf: A dictionary specifying side information tighter bounds/range
                 for each gradient component f_k. bGf[(k,i)] specify the
                 partial derivative of f_k with respect to x_i
    :param knownFun: A dictionary specifying the component of f that are known.
                     Side information of partial knowledge of the dynamics.
                     Each value of the dictionary provides a function and the
                     gradient functions.

    Returns
    -------
    function
        a function taking as input an interval vector and returning
        an over-approximation of the unknown function f over the range
        of that interval vector.
    """

    def __init__(self, Lf, traj={}, nDep={}, bf={}, bGf={}, knownFun={},
                    Lknown=None, learnLip=False, verbose=False):
        # Save the known Lipschit constant
        if Lknown is None:
            self.Lkn = np.zeros((Lf.shape[0],1))
        else:
            self.Lkn = Lknown

        # Save verbose flag
        self.verbose = verbose

        # Pre-save the number of states
        self.nS = Lf.shape[0]

        # Save the flag for learning LIpschitz constants
        self.learnLip = learnLip

        # Update the known function per component
        self.nDep = nDep
        self.bGf = bGf
        self.updateKnowFun(knownFun)

        # Define and save the current tarjectory
        self.bf = bf
        self.E0x = None
        self.E0xDot = None
        self.time = None

        # Set for each state the variable that they directly depends on
        self.updateVarDependency(nDep)

        # Initialize the Lipschitz constants of unknown term and build
        # the jacobian of such unknown fuction
        self.updateLip(Lf)

        # Build E0x and E0xDot if they are given
        if 'x' in traj and 'xDot' in traj and 'u' in traj:
            xVal = traj['x']
            xValDot = traj['xDot']
            uVal = traj['u']
            time = traj.get('t', [None for i in range(xVal.shape[1])])
            for i in range(uVal.shape[1]):
                self.update(xVal[:,i:(i+1)], xValDot[:,i:(i+1)],
                            uVal[:,i:(i+1)], time[i])
        # Build the component over-approximation
        self.fOver = dict()
        for i in range(self.nS):
            self.fOver[i] = self.createApproxFk(i)

    def updateKnowFun(self, knownFun):
        """Update the known function given as side information. In case
        no value is given on an exis, return 0 for the corresponding function
        and its derivatives with respect to the state
        """
        self.knownFun = dict()
        for i in range(self.nS):
            if i in knownFun:
                self.knownFun[i] = knownFun[i]
            else:
                dictVal = dict()
                dictVal[-1] = lambda x : 0
                for j in range(self.nS):
                    dictVal[j] = lambda x : 0
                self.knownFun[i] = dictVal

    def updateVarDependency(self, nDep):
        """Update the variable dependency of each function fk"""
        self.vDep = {}
        for i in range(self.nS):
            depVar = np.array([k for k in range(self.nS)])
            if i in nDep:
                depVar = np.setdiff1d(depVar, nDep[i])
            self.vDep[i] = depVar

    def updateLip(self, lipVal):
        """the function f is the sum of the known term and the unknown term
        hence the Lipschitz value can eb sum up
        """
        self.Lukn = lipVal
        self.Lf = self.Lukn + self.Lkn
        # Build the jacobian function, assuming the the ndep and known
        # function are already saved as attribute of the class
        self.buildJf(lipVal)
        if self.verbose:
            print('[f] Unknown fun Lipschitz: ', self.Lukn.flatten())
            print('[f] Known fun Lipschitz: ', self.Lkn.flatten())
            print('[f] Jacobian unknown f: ')
            print(self.Jf_init)

    def buildJf(self, Lf):
        """Build the Jacobian matrix Jf in the paper based on upper bounds on the
        Lipschitz constants, side information such as tighter bounds on the
        gradient, partial knowledge of the dynamics or decoupling in the states.
        """
        # Initialize the jacobian matrix with Lf * [-1,1] (see paper)
        Jf_init = np.full((Lf.shape[0], Lf.shape[0]), Interval(-1,1))
        Jf_init = np.multiply(np.repeat(Lf, Lf.shape[0], axis=1), Jf_init)
        # For states that do not depend on certain variables, Jacobian is zero
        for i , value in self.nDep.items():
            Jf_init[i, value] = Interval(0,0)
        # For states for which we have a tghter Jacobian, we use it
        for (i,j), value in self.bGf.items():
            Jf_init[i,j] = Jf_init[i,j] & value
        self.Jf_init = Jf_init
        # Jf(x) provide the jacobian when partial knowledge of the dynamics is
        # given to obtain tighter Jac based on the truth function
        def Jf(x):
            Jf_val = np.full((self.nS,self.nS), Interval(0))
            for k, value in self.knownFun.items():
                for j in range(self.nS):
                    Jf_val[k,j] = value[j](x) + self.Jf_init[k,j]
            return Jf_val
        self.Jf = Jf


    def update(self, xVal, xDotVal, uVal = None, time=None):
        """Update the trajectory of f based with the new measurement
        xVal and the derivatives xDotVal.
        """
        # No control is applied when updating the function f
        if not (uVal is None or np.array_equal(uVal, np.zeros(uVal.shape))):
            return
        # Modified xDotVal -> according to know function
        knownDerValue = np.array([[self.knownFun[i][-1](xVal)] \
                                        for i in range(xVal.shape[0])])
        xModDot = xDotVal - knownDerValue

        if self.verbose:
            print('[f] xVal : ', xVal.flatten())
            print('[f] xDot : ', xDotVal.flatten())
            print('[f] knowDer : ', knownDerValue.flatten())
        # Check the first data
        if self.E0x is None:
            self.E0x = xVal
            self.E0xDot = xModDot
            if time is not None:
                self.time = np.array([time])
            return
        # Update the Lipschitz constants if required
        if self.learnLip:
            normVal = np.zeros(self.E0x.shape)
            for i in range(normVal.shape[0]):
                normVal[i,:] = np.linalg.norm((self.E0x-xVal)[self.vDep[i],:], axis=0)
                zerosIndx = normVal[i,:] <= 1e-10
                normVal[i,zerosIndx] = -1 # Zero norm shoould generate neg Lip
            # diffVal = np.abs(xModDot - self.E0xDot) / np.linalg.norm(self.E0x-xVal, axis=0)
            diffVal = np.abs(xModDot - self.E0xDot) / normVal
            maxDiffVal = np.max(diffVal, axis=1)
            # Zero Lipschitz values should not be changed
            maxDiffVal[self.Lukn.flatten() == 0] = 0
            # Update the new Lipschitz constant
            newLip = np.maximum(self.Lukn.flatten(), maxDiffVal).reshape(-1,1)
            self.updateLip(newLip)

        # Append the new data to the set of data points
        self.E0x = np.concatenate((self.E0x, xVal), axis=1)
        self.E0xDot = np.concatenate((self.E0xDot, xModDot), axis=1)
        if time is not None:
            self.time = np.concatenate((self.time,np.array([time])))

    def removeData(self, currX, finalSize=10, thresholdSize=15):
        """Remove data that are "the farthest away" from currX. Specifically,
        we keep only 'finalSize' number of points that are the closest to currX.
        """
        # Do not remove anything if the data set is less that the desired size
        if self.E0x is None or self.E0x.shape[1] < thresholdSize:
            if self.verbose:
                print('Not enough data : E0x.size={}, Thresh={}'.format(
                        self.E0x.shape[1], thresholdSize))
            return
        if self.verbose:
            print('[f] Old E0x Data : ')
            print(self.E0x)
            print('[f] Old E0xDot Data : ')
            print(self.E0xDot)
            print('[f] E0x.size={}, Thresh={}'.format(
                                        self.E0x.shape[1], thresholdSize))
        # Find the data points with the less distance to the current point
        distValues = np.linalg.norm(self.E0x, axis=0)
        ascendingOrder = np.argsort(distValues)[:finalSize]
        # Preserve only the closest point using norm 2 to the current state
        self.E0x = self.E0x[:,ascendingOrder]
        self.E0xDot = self.E0xDot[:,ascendingOrder]
        if self.time is not None:
            self.time = self.time[ascendingOrder]
        if self.verbose:
            print('[f] Data deleted : New E0x.size={}'.format(self.E0x.shape[1]))

    def createApproxFk(self, k):
        """Create the over-approximation of fk based on the given LIpschitz
        and side information. The returned function takes as input the current
        state x.
        """
        def fkOver(x):
            if self.Lukn[k,0] <= 0:
                return Interval(self.knownFun[k][-1](x))
            assert (self.E0x is not None and self.E0x.shape[1] >= 1), \
                        "No data to estimate f[{}]".format(k)
            # Compute the distance of x to every point in the data
            norm_v = np.linalg.norm(
                        np.repeat(x[self.vDep[k],:], self.E0x.shape[1],axis=1)\
                        - self.E0x[self.vDep[k],:], axis=0)
            # COmpute the cone around each point in the given trajectory
            fk_val = self.E0xDot[k,:] + norm_v * \
                        np.full(norm_v.shape[0], self.Lukn[k,0]*Interval(-1,1))
            # Compute the intersection of each cone to get the tightest approx
            finalVal = fk_val[0]
            for i in range(1, fk_val.shape[0]):
                finalVal = finalVal & fk_val[i]
            finalVal = finalVal if k not in self.bf else (finalVal & self.bf[k])
            return finalVal + self.knownFun[k][-1](x)
        return fkOver

    def __call__(self, x):
        """Return the over-approximation of the unknown function f taken
        at the state x.
        """
        resVal = np.full((self.nS,1), Interval(0))
        for i in range(resVal.shape[0]):
            resVal[i,0] = self.fOver[i](x)
        return resVal

    def canApproximate(self, minSize =1):
        return np.sum(self.Lukn.flatten()) <=0 or \
                    self.E0x is not None and self.E0x.shape[1] >= minSize

class GOverApprox:
    """Compute an over-approximation of the unknown function G
    given the separation E_j, side information, and the upper bounds on the
    Lispchitz constant of the unknown function G

    Parameters
    ----------
    :param traj: The data being used to over-approximate G
    :param LG: The upper bound on the Lipschitz constants of G
    :param vDep: A dictionary where the value of a key k,l gives the indexes
                 of the states for which G_kl is dependent
    :param bG: A dictionary specifying side information tighter bounds/range
                 for each component G_kl
    :param bGG: A dictionary specifying side information tighter bounds/range
                 for each gradient component Gkl. bGf[(k,l,i)] specify the
                 partial derivative of G_kl with respect to x_i
    :param knownFun: A dictionary specifying the component of G that are known.
                     Side information of partial knowledge of the dynamics.
                     Each value of the dictionary provides a function and the
                     gradient functions.

    Returns
    -------
    function
        a function taking as input an interval vector and returning
        an over-approximation of the unknown function G over the range
        of that interval vector.
    """
    def __init__(self, LG, Fover, traj={}, nDep={}, bG={}, bGG={}, knownFun={},
                 Lknown=None, learnLip=False, verbose=False):
        # Save the known Lipschit constant
        if Lknown is None:
            self.Lkn = np.zeros(LG.shape)
        else:
            self.Lkn = Lknown

        # Save verbose flag
        self.verbose = verbose

        # Pre-save the number of states
        self.nS = LG.shape[0]
        self.nC = LG.shape[1]

        # Save the flag for learning LIpschitz constants
        self.learnLip = learnLip

        # Update the known function per component
        self.nDep = nDep
        self.bGG = bGG
        self.updateKnowFun(knownFun)

        # Define and save the current tarjectory
        self.bG = bG
        self.Ej = dict()
        self.xDot = dict()
        self.time = dict()

        # Save the over-approximation of F
        self.Fover = Fover

        # Set for each state the variable that they directly depends on
        self.updateVarDependency(nDep)

        # Initialize the Lipschitz constants of unknown term and build
        # the jacobian of such unknown fuction
        self.updateLip(LG)

        # Build the data separation given Ej
        if 'x' in traj and 'xDot' in traj and 'u' in traj:
            xVal = traj['x']
            xValDot = traj['xDot']
            uVal = traj['u']
            time = traj.get('t', [None for i in range(xVal.shape[1])])
            for i in range(uVal.shape[1]):
                self.update(xVal[:,i:(i+1)], xValDot[:,i:(i+1)],
                            uVal[:,i:(i+1)], time[i])
        # Build the component over-approximation
        self.GOver = dict()
        for k in range(self.nS):
            for l in range(self.nC):
                self.GOver[(k,l)] = self.createApproxGkl(k,l)

    def updateKnowFun(self, knownFun):
        """Update the known function given as side information. In case
        no value is given on an exis, return 0 for the corresponding function
        and its derivatives with respect to the state
        """
        self.knownFun = dict()
        for i in range(self.nS):
            for l in range(self.nC):
                if (i,l) in knownFun:
                    self.knownFun[(i,l)] = knownFun[(i,l)]
                else:
                    dictVal = dict()
                    dictVal[-1] = lambda x : 0
                    for j in range(self.nS):
                        dictVal[j] = lambda x : 0
                    self.knownFun[(i,l)] = dictVal

    def updateVarDependency(self, nDep):
        """Update the variable dependency of each function Gkl"""
        self.vDep = {}
        for k in range(self.nS):
            for l in range(self.nC):
                depVar = np.array([k for k in range(self.nS)])
                if (k,l) in nDep:
                    depVar = np.setdiff1d(depVar, nDep[(k,l)])
                self.vDep[(k,l)] = depVar

    def updateLip(self, lipVal):
        """the function f is the sum of the known term and the unknown term
        hence the Lipschitz value can eb sum up
        """
        self.Lukn = lipVal
        self.LG = self.Lukn + self.Lkn
        # Build the jacobian function, assuming the the ndep and known
        # function are already saved as attribute of the class
        self.buildJG(lipVal)
        if self.verbose:
            print('[G] Unknown fun Lipschitz: ')
            print(self.Lukn)
            print('[G] Known fun Lipschitz: ')
            print(self.Lkn)
            print('[G] Jacobian unknown: ')
            print(self.JG_init)

    def buildJG(self, LG):
        """Build the Jacobian matrix JG in the paper based on upper bounds on the
        Lipschitz constants, side information such as tighter bounds on the
        gradient, partial knowledge of the dynamics or decoupling in the states.
        """
        # Initialize the jacobian matrix with LG * [-1,1] (see paper)
        JG_init = np.full((self.nS, self.nC, self.nS), Interval(-1,1))
        for k in range(self.nS):
            for l in range(self.nC):
                JG_init[k,l] *= LG[k,l]
        # For states that do not depend on certain variables, Jacobian is zero
        for (k,l) , value in self.nDep.items():
            JG_init[k,l, value] = Interval(0,0)
        # For states for which we have a tighter Jacobian, we use it
        for (k,l,i), value in self.bGG.items():
            JG_init[k,l,i] = JG_init[k,l,i] & value
        self.JG_init = JG_init
        # Jf(x) provide the jacobian when partial knowledge of the dynamics is
        # given to obtain tighter Jac based on the truth function
        def JG(x):
            JG_val = np.full((self.nS, self.nC, self.nS), Interval(0))
            for (k,l), value in self.knownFun.items():
                for i in range(self.nS):
                    JG_val[k,l,i] = value[i](x) + self.JG_init[k,l,i]
            return JG_val
        self.JG = JG

    def update(self, xVal, xDotVal, uVal, time=None):
        """Update the trajectory of G based on the new measurement
        xVal, the derivatives xDotVal and the control uVal."""

        # Obtain the values of uVal that are non zeros
        uVal[np.abs(uVal) < 1e-5] = 0
        (zVal, nzVal) = np.nonzero(uVal)
        # If there are multiples, the data can't be separated
        if zVal.shape[0] != 1:
            return
        # Otherwise he data point is in Ej for j = nZInd
        nZInd = zVal[0]
        # COmpute the known function and remove it from the current xdot
        knownDerValue = np.array([[self.knownFun[(i,nZInd)][-1](xVal)] \
                                        for i in range(xVal.shape[0])])
        xModDot = ((xDotVal - self.Fover(xVal)) / uVal[nZInd,0]) - knownDerValue
        if self.verbose:
            print('[G] xVal : ', xVal.flatten())
            print('[G] xDot : ', xDotVal.flatten())
            print('[G] uValue : ', uVal[nZInd,0])
            print('[G] knowDer : ', knownDerValue.flatten())
            print('[G] index update : ', nZInd)
        # If no data is present inside Ej, fix it accordingly
        if nZInd not in self.Ej:
            self.Ej[nZInd] = (xVal, xDotVal, np.array([uVal[nZInd,0]]))
            self.xDot[nZInd] = xModDot
            if time is not None:
                self.time[nZInd] = np.array([time])
            return
        # If the system is learning the LIpschitz constants, do it
        if self.learnLip:
            normVal = np.zeros(self.Ej[nZInd][0].shape)
            for i in range(normVal.shape[0]):
                normVal[i,:] = np.linalg.norm(
                    (self.Ej[nZInd][0]-xVal)[self.vDep[(i,nZInd)],:], axis=0)
                zerosIndx = normVal[i,:] <= 1e-5
                normVal[i,zerosIndx] = -1 # Zero norm shoould generate neg Lip
            diffVal = np.abs(self.xDot[nZInd] - xModDot) / normVal
            diffVal = np.array([[diffVal[i,j].ub for j in range(diffVal.shape[1])] \
                                    for i in range(diffVal.shape[0])])
            maxDiffVal = np.max(diffVal, axis=1)
            # Zero Lipschitz values should not be changed
            maxDiffVal[self.Lukn[:,nZInd] == 0] = 0
            # Update the new Lipschitz constant
            newLipCol = np.maximum(self.Lukn[:,nZInd], maxDiffVal)
            self.Lukn[:,nZInd] = newLipCol
            self.updateLip(self.Lukn)

        # Append the new data
        (currX, currXdot, currU) = self.Ej[nZInd]
        currX = np.concatenate((currX, xVal),axis=1)
        currXdot = np.concatenate((currXdot, xDotVal), axis=1)
        currU = np.concatenate((currU, np.array([uVal[nZInd,0]])))
        self.Ej[nZInd] =  (currX, currXdot, currU)
        self.xDot[nZInd] = np.concatenate((self.xDot[nZInd], xModDot), axis=1)

        if time is not None:
            self.time[nZInd] = np.concatenate((self.time[nZInd],np.array([time])))


    def removeData(self, currX, finalSize=10, thresholdSize=15):
        """Remove data that are "the farthest away" from currX. Specifically,
        we keep only 'finalSize' number of points that are the closest to currX.
        """
        # Do not remove anything if the data set is less that the desired size
        for ind, (xVal, xDotVal, uVal) in self.Ej.items():
            if  xVal.shape[1] < thresholdSize:
                if self.verbose:
                    print('[G] Not enough data ind {}: E0x.size={}, Thresh={}'.format(
                        ind, xVal.shape[1], thresholdSize))
                continue
            if self.verbose:
                print('[G] Old Ejx[{}] Data : '.format(ind))
                print(xVal)
                print('[G] Old EjxDot Data : ')
                print(xDotVal)
                print('[G] Ejx[{}].size={}, Thresh={}'.format(ind,
                                        xVal.shape[1], thresholdSize))
            # Find the data points with the less distance to the current point
            distValues = np.linalg.norm(xVal, axis=0)
            ascendingOrder = np.argsort(distValues)[:finalSize]
            # Preserve only the closest point using norm 2 to the current state
            nxVal = xVal[:,ascendingOrder]
            nxDotVal = xDotVal[:,ascendingOrder]
            nuVal = uVal[ascendingOrder]
            self.Ej[ind] = (nxVal, nxDotVal, nuVal)
            self.xDot[ind] = self.xDot[ind][:, ascendingOrder]
            if ind in self.time:
                self.time[ind] = self.time[ind][ascendingOrder]
            if self.verbose:
                print('[G] Data deleted : New Ejx[{}].size={}'.format(
                        ind, self.Ej[ind][0].shape[1]))


    def createApproxGkl(self, k, l):
        """Create the over-approximation of Gkl based on the given Lipschitz
        and side information. The returned function takes as input the current
        state x.
        """
        def GklOver(x):
            if self.Lukn[k,l] <=0:
                return Interval(self.knownFun[(k,l)][-1](x))
            assert (l in self.Ej) and self.Ej[l][0].shape[1] >= 1, \
                                "No data to estimate G[{},{}]".format(k,l)
            # Get the corresponding data to kl
            (xVal, xDotVal, uVal) = self.Ej[l]
            # Compute the distance of x to every point in the data
            norm_v = np.linalg.norm(
                        np.repeat(x[self.vDep[(k,l)],:], xVal.shape[1],axis=1)\
                        - xVal[self.vDep[(k,l)],:], axis=0)
            # COmpute the cone around each point in the given trajectory
            Gkl_val = self.xDot[l][k,:] +  norm_v * \
                        np.full(norm_v.shape[0], self.LG[k,l] * Interval(-1,1))
            # Compute the intersection of each cone to get the tightest approx
            finalVal = Gkl_val[0]
            for i in range(1, Gkl_val.shape[0]):
                finalVal = finalVal & Gkl_val[i]
            finalVal = finalVal if (k,l) not in self.bG else (finalVal & self.bG[(k,l)])
            return finalVal + self.knownFun[(k,l)][-1](x)
        return GklOver

    def __call__(self, x):
        """Return the over-approximation of the unknown function G taken
        at the state x.
        """
        resVal = np.full((self.nS,self.nC), Interval(0))
        for k in range(resVal.shape[0]):
            for l in range(resVal.shape[1]):
                resVal[k,l] = self.GOver[(k,l)](x)
        return resVal

    def canApproximate(self, minSize = 1):
        for j in range(self.nC):
            if j not in self.Ej:
                return False
            elif self.Ej[j][0].shape[1] < minSize:
                return False
        return True
