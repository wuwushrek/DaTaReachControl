import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tikzplotlib

from DaTaReachControl import *

def computeData(overApprox):
    x_lb = np.zeros(overApprox.shape)
    x_ub = np.zeros(overApprox.shape)
    for i in range(overApprox.shape[0]):
        for j in range(overApprox.shape[1]):
            x_lb[i,j] = overApprox[i,j].lb
            x_ub[i,j] = overApprox[i,j].ub
    return x_lb , x_ub

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
traj = {'x' : xTraj, 'xDot' : xDotTraj, 'u' : uSeq, 't' : time_meas}

#Over-approximations functions
knownFunF = {0 : {-1 : lambda x : Interval(0),
                   0 : lambda x : Interval(0)}}
fover = FOverApprox(lipF, traj=traj, knownFun=knownFunF)
gover = GOverApprox(lipG, fover, traj=traj)

# Compare the differential inclusion with the true but unknown xdot
pointLearning = 40 # number of point to plot between each deltaT of initial traj
tend = time_meas[-1] + deltaT # last time measured in the trajectory
xFinal = xNext # the value of the state at tend

# Compute a high resolution trajectory from the measurements
xTrajP, xTrajDotP,_ = generateTraj(f, G, xInit, uSeq, deltaT, pointLearning)
time_p = np.array([i*(deltaT/pointLearning) for i in range(xTrajP.shape[1])])

# Signal that's being applied after t = tend
def u_sig(t):
    return np.array([[0.5*np.cos(3*(t - tend))]])

def u_over(intVal):
    return 0.5* np.cos(np.array([[3*(intVal- tend)]]))

def u_der(intVal):
    return -1.5*np.sin(np.array([[3*(intVal - tend)]]))

# Generate a solution of the ODE ( a trajectory) stating at tend with
# control signal u_sig and for a duration of 2s
evalTime = np.linspace(tend,tend+2,101)
time_l, x_after, xdot_after = synthTraj(f, G, u_sig, xFinal, evalTime)

# Concatenate both trajectory
x_true = np.concatenate((xTrajP,x_after[:,1:]), axis=1)
xdot_true = np.concatenate((xTrajDotP, xdot_after[:,:]), axis=1)
full_time = np.concatenate((time_p,time_l[1:]), axis=None)

# COmpute the over-approx during the learning process
res_x = None
res_t1 = None
res_x_t = None
res_t2 = None
for i in range(uSeq.shape[1]):
    t0 = time_meas[i]
    x0 = np.array([[Interval(xTraj[j,i])] for j in range(xTraj.shape[0])])
    def temp_u (t):
        return np.array([[Interval(uSeq[j,i])] for j in range(uSeq.shape[0])])
    def temp_u_der (t):
        return np.array([[Interval(0)] for j in range(uSeq.shape[0])])
    tVal1, xVal = overApproxTube(x0, t0, 41, 0.01, fover, gover, temp_u,
                                    temp_u_der, useFast=True)
    if res_t1 is None:
        res_t1 = tVal1[:-1]
        res_x = xVal[:,:-1]
    else:
        res_t1 = np.concatenate((res_t1, tVal1[:-1]))
        res_x = np.concatenate((res_x, xVal[:,:-1]),axis=1)

    tVal2, xVal_t = overApproxTube(x0, t0, 41, 0.01, fover, gover, temp_u,
                                    temp_u_der, useFast=False)
    if res_t2 is None:
        res_t2 = tVal2[:-1]
        res_x_t = xVal_t[:,:-1]
    else:
        res_t2 = np.concatenate((res_t2, tVal2[:-1]))
        res_x_t = np.concatenate((res_x_t, xVal_t[:,:-1]),axis=1)

# Compute both tight over-approximation and loose over-approx
x0 = np.array([[Interval(xFinal[j])] for j in range(xFinal.shape[0])])
t1, xOver = overApproxTube(x0, tend, 201, 0.01, fover, gover, u_over,
                                    u_der, useFast=True)
t2, xOver_t = overApproxTube(x0, tend, 201, 0.01, fover, gover, u_over,
                                    u_der, useFast=False)
res_t1 = np.concatenate((res_t1, t1))
res_x = np.concatenate((res_x, xOver),axis=1)

res_t2 = np.concatenate((res_t2, t2))
res_x_t = np.concatenate((res_x_t, xOver_t),axis=1)

x_lb , x_ub = computeData(res_x)
x_lb_t , x_ub_t = computeData(res_x_t)

plt.figure()

plt.plot(time_meas, xTraj[0,:], "k*", markerSize=4, label="$\mathcal{T}$")
plt.plot(full_time, x_true[0,:], 'tab:orange', label="$x$")
plt.fill_between(res_t1, x_lb[0,:], x_ub[0,:], alpha=0.7, facecolor="tab:blue",\
    edgecolor= "darkcyan", label="Gronwall")
plt.fill_between(res_t2, x_lb_t[0,:], x_ub_t[0,:],  alpha=0.9, facecolor="tab:green",\
    edgecolor= "darkgreen", label="Recursive")

plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(ncol=4, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.5)
plt.grid(True)
plt.tight_layout()

plt.show()
