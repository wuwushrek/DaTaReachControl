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

    def __init__(self, Lf, traj={}, nDep={}, bf={}, bGf={}, knownFun={}):
        self.Lf = Lf
        self.knownFun = knownFun
        self.bf = bf
        self.E0x = None
        self.E0xDot = None
        # Set for each state the variable that they directly depends on
        self.updateVarDependency(nDep)
        # Build the jacobian of f given Lf and the dep+ gradient side info
        self.Jf = self.buildJf(Lf, nDep, bGf, knownFun)
        # Build E0x and E0xDot if they are given
        if 'x' in traj and 'xDot' in traj and 'u' in traj:
            xVal = traj['x']
            xValDot = traj['xDot']
            uVal = traj['u']
            for i in range(xVal.shape[1]):
                self.update(xVal[:,i:(i+1)], xValDot[:,i:(i+1)], uVal[:,i:(i+1)])
        # Build the component over-approximation
        self.fOver = dict()
        for i in range(self.Lf.shape[0]):
            self.fOver[i] = self.createApproxFk(i)

    def update(self, xVal, xDotVal, uVal = None):
        """Update the trajectory of f based with the new measurement
        xVal and the derivatives xDotVal.
        """
        # No control is applied when updating the function f
        if not (uVal is None or np.array_equal(uVal, np.zeros(uVal.shape))):
            return
        # Check the first data
        if self.E0x is None:
            self.E0x = xVal
            self.E0xDot = xDotVal
            return
        # Append the new data to the set of data points
        self.E0x = np.concatenate((self.E0x, xVal), axis=1)
        self.E0xDot = np.concatenate((self.E0xDot, xDotVal), axis=1)

    def removeData(self, currX, finalSize=10):
        """Remove data that are "the farthest away" from currX. Specifically,
        we keep only 'finalSize' number of points that are the closest to currX.
        """
        # Do not remove anything if the data set is less that the desired size
        if self.E0x is None or self.E0x.shape[1] < finalSize:
            return
        # Find the data points with the less distance to the current point
        distValues = np.linalg.norm(self.E0x, axis=0)
        ascendingOrder = np.argsort(distValues)[:finalSize]
        # Preserve only the closest point using norm 2 to the current state
        self.E0x = self.E0x[:,ascendingOrder]
        self.E0xDot = self.E0xDot[:,ascendingOrder]

    def createApproxFk(self, k):
        """Create the over-approximation of fk based on the given LIpschitz
        and side information. The returned function takes as input the current
        state x.
        """
        # If the function os given, just use it
        if k in self.knownFun:
            return self.knownFun[k][-1]
        def fkOver(x):
            assert self.E0x.shape[1] >= 1, "No data to estimate f[{}]".format(k)
            # Compute the distance of x to every point in the data
            norm_v = np.linalg.norm(
                        np.repeat(x[self.vDep[k],:], self.E0x.shape[1],axis=1)\
                        - self.E0x[self.vDep[k],:], axis=0)
            # COmpute the cone around each point in the given trajectory
            fk_val = self.E0xDot[k,:] + self.Lf[k,0] * norm_v * \
                                        np.full(norm_v.shape[0], Interval(-1,1))
            # Compute the intersection of each cone to get the tightest approx
            finalVal = fk_val[0]
            for i in range(1, fk_val.shape[0]):
                finalVal = finalVal & fk_val[i]
            return finalVal if k not in self.bf else (finalVal & self.bf[k])
        return fkOver

    def buildJf(self, Lf, nDep, bGf, knownFun):
        """Build the Jacobian matrix Jf in the paper based on upper bounds on the
        Lipschitz constants, side information such as tighter bounds on the
        gradient, partial knowledge of the dynamics or decoupling in the states.
        """
        # Initialize the jacobian matrix with Lf * [-1,1] (see paper)
        Jf_init = np.full((Lf.shape[0], Lf.shape[0]), Interval(-1,1))
        Jf_init = np.multiply(np.repeat(Lf, Lf.shape[0], axis=1), Jf_init)
        # For states that do not depend on certain variables, Jacobian is zero
        for i , value in nDep.items():
            Jf_init[i, value] = Interval(0,0)
        # For states for which we have a tghter Jacobian, we use it
        for (i,j), value in bGf.items():
            Jf_init[i,j] = Jf_init[i,j] & value
        self.Jf_init = Jf_init
        # Jf(x) provide the jacobian when partial knowledge of the dynamics is
        # given to obtain tighter Jac based on the truth function
        def Jf(x):
            for k, value in knownFun.items():
                for j in range(self.Lf.shape[0]):
                    self.Jf_init[k,j] = value[j](x)
            return self.Jf_init
        return Jf

    def updateVarDependency(self, nDep):
        """Update the variable dependency of each function fk"""
        self.vDep = {}
        for i in range(self.Lf.shape[0]):
            depVar = np.array([k for k in range(self.Lf.shape[0])])
            if i in nDep:
                depVar = np.setdiff1d(depVar, nDep[i])
            self.vDep[i] = depVar

    def __call__(self, x):
        """Return the over-approximation of the unknown function f taken
        at the state x.
        """
        resVal = np.full((self.Lf.shape[0],1), Interval(0))
        for i in range(resVal.shape[0]):
            resVal[i,0] = self.fOver[i](x)
        return resVal

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
    def __init__(self, LG, Fover, traj={}, nDep={}, bG={}, bGG={}, knownFun={}):
        self.Fover = Fover
        self.LG = LG
        self.knownFun = knownFun
        self.bG = bG
        self.Ej = dict()
        self.xDot = dict()
        # Set for each state the variable that they directly depends on
        self.updateVarDependency(nDep)
        # Build the jacobian of G given LG and the dep+ gradient side info
        self.JG = self.buildJG(LG, nDep, bGG, knownFun)
        # Build the data separation given Ej
        if 'x' in traj and 'xDot' in traj and 'u' in traj:
            xVal = traj['x']
            xValDot = traj['xDot']
            uVal = traj['u']
            for i in range(xVal.shape[1]):
                self.update(xVal[:,i:(i+1)], xValDot[:,i:(i+1)], uVal[:,i:(i+1)])
        # Build the component over-approximation
        self.GOver = dict()
        for k in range(self.LG.shape[0]):
            for l in range(self.LG.shape[1]):
                self.GOver[(k,l)] = self.createApproxGkl(k,l)

    def updateVarDependency(self, nDep):
        """Update the variable dependency of each function Gkl"""
        self.vDep = {}
        for k in range(self.LG.shape[0]):
            for l in range(self.LG.shape[1]):
                depVar = np.array([k for k in range(self.LG.shape[0])])
                if (k,l) in nDep:
                    depVar = np.setdiff1d(depVar, nDep[(k,l)])
                self.vDep[(k,l)] = depVar

    def buildJG(self, LG, nDep, bGG, knownFun):
        """Build the Jacobian matrix JG in the paper based on upper bounds on the
        Lipschitz constants, side information such as tighter bounds on the
        gradient, partial knowledge of the dynamics or decoupling in the states.
        """
        # Initialize the jacobian matrix with LG * [-1,1] (see paper)
        JG_init = np.full((LG.shape[0], LG.shape[1], LG.shape[0]), Interval(-1,1))
        for k in range(LG.shape[0]):
            for l in range(LG.shape[1]):
                JG_init[k,l] *= LG[k,l]
        # For states that do not depend on certain variables, Jacobian is zero
        for (k,l) , value in nDep.items():
            JG_init[k,l, value] = Interval(0,0)
        # For states for which we have a tighter Jacobian, we use it
        for (k,l,i), value in bGG.items():
            JG_init[k,l,i] = JG_init[k,l,i] & value
        self.JG_init = JG_init
        # Jf(x) provide the jacobian when partial knowledge of the dynamics is
        # given to obtain tighter Jac based on the truth function
        def JG(x):
            for (k,l), value in knownFun.items():
                for i in range(self.LG.shape[0]):
                    self.JG_init[k,l,i] = value[i](x)
            return self.JG_init
        return JG

    def update(self, xVal, xDotVal, uVal):
        """Update the trajectory of G based on the new measurement
        xVal, the derivatives xDotVal and the control uVal."""

        # Obtain the values of uVal that are non zeros
        (zVal, nzVal) = np.nonzero(uVal)
        # If there are multiples, the data can't be separated
        if zVal.shape[0] != 1:
            return
        # Otherwise he data point is in Ej for j = nZInd
        nZInd = zVal[0]
        if nZInd not in self.Ej:
            self.Ej[nZInd] = (xVal, xDotVal, np.array([uVal[nZInd,0]]))
            self.xDot[nZInd] = (xDotVal - self.Fover(xVal)) / uVal[nZInd,0]
            return
        (currX, currXdot, currU) = self.Ej[nZInd]
        currX = np.concatenate((currX, xVal),axis=1)
        currXdot = np.concatenate((currXdot, xDotVal), axis=1)
        currU = np.concatenate((currU, np.array([uVal[nZInd,0]])))
        self.Ej[nZInd] =  (currX, currXdot, currU)
        self.xDot[nZInd] = np.concatenate((self.xDot[nZInd],
                            (xDotVal - self.Fover(xVal))/uVal[nZInd,0]), axis=1)


    def removeData(self, currX, finalSize=10):
        """Remove data that are "the farthest away" from currX. Specifically,
        we keep only 'finalSize' number of points that are the closest to currX.
        """
        # Do not remove anything if the data set is less that the desired size
        for ind, (xVal, xDotVal, uVal) in self.Ej.items():
            if  xVal.shape[1] < finalSize:
                continue
            # Find the data points with the less distance to the current point
            distValues = np.linalg.norm(xVal, axis=0)
            ascendingOrder = np.argsort(distValues)[:finalSize]
            # Preserve only the closest point using norm 2 to the current state
            nxVal = xVal[:,ascendingOrder]
            nxDotVal = xDotVal[:,ascendingOrder]
            nuVal = uVal[ascendingOrder]
            self.Ej[ind] = (nxVal, nxDotVal, nuVal)
            self.xDot[ind] = self.xDot[ind][:, ascendingOrder]


    def createApproxGkl(self, k, l):
        """Create the over-approximation of Gkl based on the given Lipschitz
        and side information. The returned function takes as input the current
        state x.
        """
        # If the function is given, just use it
        if (k,l) in self.knownFun:
            return self.knownFun[(k,l)][-1]
        def GklOver(x):
            assert (l in self.Ej) and self.Ej[l][0].shape[1] >= 1, \
                                "No data to estimate G[{},{}]".format(k,l)
            # Get the corresponding data to kl
            (xVal, xDotVal, uVal) = self.Ej[l]
            # Compute the distance of x to every point in the data
            norm_v = np.linalg.norm(
                        np.repeat(x[self.vDep[(k,l)],:], xVal.shape[1],axis=1)\
                        - xVal[self.vDep[(k,l)],:], axis=0)
            # COmpute the cone around each point in the given trajectory
            Gkl_val = self.xDot[l][k,:] + self.LG[k,l] * norm_v * \
                                        np.full(norm_v.shape[0], Interval(-1,1))
            # Compute the intersection of each cone to get the tightest approx
            finalVal = Gkl_val[0]
            for i in range(1, Gkl_val.shape[0]):
                finalVal = finalVal & Gkl_val[i]
            return finalVal if (k,l) not in self.bG else (finalVal & self.bG[(k,l)])
        return GklOver

    def __call__(self, x):
        """Return the over-approximation of the unknown function G taken
        at the state x.
        """
        resVal = np.full((self.LG.shape[0],self.LG.shape[1]), Interval(0))
        for k in range(resVal.shape[0]):
            for l in range(resVal.shape[1]):
                resVal[k,l] = self.GOver[(k,l)](x)
        return resVal
