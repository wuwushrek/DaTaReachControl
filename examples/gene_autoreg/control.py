import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tikzplotlib

from DaTaReachControl import *
import cvxpy as cp

# Unknown function f
def f(x):
    return np.array([0])
lipF = np.array([[0.]])

# UNknown function G
def G(x):
    return np.array([[(x[0]+1)**2 / (1 + (x[0]+1)**2)]])
lipG = np.array([[0.7]])

# Control signal applied for the initial trajectory
uSeq = np.array([[0, 1.0, 0.5, 1.0, -0.75, 0.5, 0.75, 1.0, 0.5, 1.0, 1.0]])

# Initial state
xInit = np.array([0.1])

# Sampling time
deltaT = 0.4

# Generate the initial trajectory
xTraj, xDotTraj, xNext = generateTraj(f, G, xInit, uSeq, deltaT, 1)
time_meas = np.array([ i*deltaT for i in range(xTraj.shape[1])])
traj = {'x' : xTraj, 'xDot' : xDotTraj, 'u' : uSeq}

#Over-approximations functions
knownFunF = {0 : {-1 : lambda x : 0,
                   0 : lambda x : 0}}
fover = FOverApprox(lipF, traj=traj, knownFun=knownFunF)
gover = GOverApprox(lipG, fover, traj=traj)

# Compare the differential inclusion with the true but unknown xdot
pointLearning = 40 # number of point to plot between each deltaT of initial traj
tend = time_meas[-1] + deltaT # last time measured in the trajectory
xFinal = xNext # the value of the state at tend

# Compute a high resolution trajectory from the measurements
xTrajP, xTrajDotP,_ = generateTraj(f, G, xInit, uSeq, deltaT, pointLearning)
time_p = np.array([i*(deltaT/pointLearning) for i in range(xTrajP.shape[1])])

# Delta time for the control
dt = 0.04
# Get control range
uRange = np.array([[Interval(-1,1)]])
xG = -1.0

# Define the function that return the next state
next_state_aux = synthNextState(f, G, dt, atol=1e-10, rtol=1e-10)

def costFun(x, u):
    return (x[0]-xG)*(x[0]-xG)

# Stopping criteria
def stopCriteria(x):
    # print (np.abs(x[0]- xG))
    return np.abs(x[0,0]- xG) <= 1e-3

maxIter=1000

synth_control = DaTaControl(costFun, uRange, lipF, lipG, traj, knownFunF=knownFunF,
                    optVerb=False, solverVerb=True, probLearning=[0.1, 0.9],
                    threshUpdateApprox=0.1, thresMeanTraj=1e-3, coeffLearning=0.1,
                    minDataF=15 , maxDataF=25, minDataG=5, maxDataG=10, dt=dt)

currX = np.copy(xFinal).reshape(-1,1)
currXdot = np.copy(xDotTraj[:,-1]).reshape(-1,1)

for i in range(maxIter):
    xTrajP = np.concatenate((xTrajP, currX), axis=1)
    time_p = np.concatenate((time_p, np.array([tend + i*dt])))
    # print(currX.flatten())
    uVal = synth_control(currX, currXdot)
    # print(uVal.flatten())
    currX, currXdot = next_state_aux(currX.flatten(), uVal)
    currX = currX.reshape(-1,1)
    currXdot = currXdot.reshape(-1,1)

plt.figure()

plt.plot(time_meas, xTraj[0,:], "k*", markerSize=4, label="$\mathcal{T}$")
plt.plot(time_p, xTrajP[0,:],"tab:orange", label="$x$")
plt.plot(time_p[xTrajDotP.shape[1]:],np.full(xTrajP.shape[1]-xTrajDotP.shape[1], xG),
    "rs", markerSize=10, label="$x_\mathcal{G}$")
plt.legend(ncol=4, bbox_to_anchor=(0,1), loc='lower left')
plt.autoscale(enable=True, axis='y', tight=True)
plt.xlabel('$t$')
plt.grid(True)
plt.tight_layout()
plt.show()
