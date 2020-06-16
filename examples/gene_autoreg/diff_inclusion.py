import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib

from DaTaReachControl import *

# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def computeData(f, G, fover, Gover, xData,  uData):
    true_f = np.zeros(xData.shape)
    true_G = np.zeros((xData.shape[1], xData.shape[0], uData.shape[0]))
    true_xdot = np.zeros(xData.shape)
    over_f_lb = np.zeros(xData.shape)
    over_f_ub = np.zeros(xData.shape)
    over_G_lb = np.zeros((xData.shape[1], xData.shape[0], uData.shape[0]))
    over_G_ub = np.zeros((xData.shape[1], xData.shape[0], uData.shape[0]))
    over_xDot_lb = np.zeros(xData.shape)
    over_xDot_ub = np.zeros(xData.shape)
    for i in range(xData.shape[1]):
        xIntVal = np.array([[xData[j,i]] for j in range(xData.shape[0])])
        fVal = f(xData[:,i])
        GVal = G(xData[:,i])
        fOverVal = fover(xIntVal)
        GOverVal = Gover(xIntVal)
        xDot = fVal + np.matmul(GVal , uData[:,i:(i+1)]).flatten()
        xOverVal = (fOverVal + np.matmul(GOverVal, uData[:,i:(i+1)])).flatten()
        true_f[:,i] = fVal[:]
        true_G[i,:,:] = GVal[:,:]
        true_xdot[:,i] = xDot[:]
        for j in range(xData.shape[0]):
            over_f_lb[j,i] =  fOverVal[j,0].lb
            over_f_ub[j,i] =  fOverVal[j,0].ub
            over_xDot_lb[j,i] =  xOverVal[j].lb
            over_xDot_ub[j,i] =  xOverVal[j].ub
            for k in range(uData.shape[0]):
                over_G_lb[i,j,k] = GOverVal[j,k].lb
                over_G_ub[i,j,k] = GOverVal[j,k].ub
    return true_f, true_G, true_xdot, (over_f_lb,over_f_ub), (over_G_lb,over_G_ub),\
                (over_xDot_lb, over_xDot_ub)

# Unknown function f
def f(x):
    return np.array([0.4*np.cos(3*x[0])])
lipF = np.array([[1.2]])

# UNknown function G
def G(x):
    return np.array([[(x[0]+1)**2 / (1 + (x[0]+1)**2)]])
lipG = np.array([[0.7]])

# Control signal applied for the initial trajectory
uSeq = np.array([[0,1.0,0, 1.0,0,1.0,0,1.0,0,1.0,0.0]])

# Initial state
xInit = np.array([0.1])

# Sampling time
deltaT = 0.4

# Generate the initial trajectory
xTraj, xDotTraj, xNext = generateTraj(f, G, xInit, uSeq, deltaT, 1)
time_meas = np.array([ i*deltaT for i in range(xTraj.shape[1])])
traj = {'x' : xTraj, 'xDot' : xDotTraj, 'u' : uSeq, 't' : time_meas}

# Compute the over-approximation fI and GI
# fover = FOverApprox(lipF, traj=traj, knownFun=knownFunF, nDep=nDepF)
# Gover = GOverApprox(lipG, fover, traj=traj, knownFun=knownFunG, nDep=nDepG)
fover = FOverApprox(lipF, traj=traj)
Gover = GOverApprox(lipG, fover, traj=traj)

# Compare the differential inclusion with the true but unknown xdot
pointLearning = 40 # number of point to plot between each deltaT of initial traj
tend = time_meas[-1] + deltaT # last time measured in the trajectory
xFinal = xNext # the value of the state at tend

# Compute a high resolution trajectory from the measurements
xTrajP, xTrajDotP, _ = generateTraj(f, G, xInit, uSeq, deltaT, pointLearning)
time_p = np.array([i*(deltaT/pointLearning) for i in range(xTrajP.shape[1])])

# Signal that's being applied after t = tend
def uSig(t):
    return np.array([[-0.5*np.cos(4*(t - tend))]])

# Generate a solution of the ODE ( a trajectory) starting at tend with
# control signal uSig and for a duration of 2s
evalTime = np.linspace(tend,tend+2,61)
time_l, x_after, xdot_after = synthTraj(f, G, uSig, xFinal, evalTime)

# Collect the control values used from both the high resolution trajectory
# and the trajectory after tend
uData = np.zeros((uSeq.shape[0],time_p.shape[0]+time_l.shape[0]))
currInd = -1
for i in range(time_p.shape[0]-1):
    if i % pointLearning == 0:
        currInd += 1
    uData[:,i] = uSeq[:, currInd]
for i in range(time_l.shape[0]):
    uData[:,time_p.shape[0]+i] = uSig(time_l[i])[:]

# Concatenate both trajectory
x_true = np.concatenate((xTrajP,x_after[:,1:]), axis=1)
xdot_true = np.concatenate((xTrajDotP, xdot_after[:,:]), axis=1)
full_time = np.concatenate((time_p,time_l[1:]), axis=None)

# Compute the over-approximations and the differential inclusion
true_f, true_G, true_xdot, \
(over_f_lb,over_f_ub),\
(over_G_lb,over_G_ub),\
(over_xDot_lb, over_xDot_ub)=computeData(f, G, fover, Gover, x_true,  uData)

# Plot the comparison
plt.figure()
plt.fill_between(full_time, over_xDot_lb[0,:],over_xDot_ub[0,:], alpha=0.7,\
                    facecolor="tab:cyan", label="$\mathbf{f}(x)+\mathbf{G}(x)u$")

plt.plot(full_time, true_xdot[0,:], 'magenta', label="$f(x)+G(x)u$")

plt.plot(fover.time, fover.E0xDot[0,:], "k*", markerSize=5, label="$\mathcal{E}_0$")

plt.plot(Gover.time[0], Gover.Ej[0][1][0,:], "ro", markerSize=5, label="$\mathcal{E}_1$")
plt.autoscale(enable=True, axis='y', tight=True)
plt.legend(ncol=4, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.5)
plt.grid(True)
plt.tight_layout()


# plt.savefig("diff_incl_illus_xdot.svg")
# tikzplotlib.save('diff_incl_illus_xdot.tex')
plt.show()
