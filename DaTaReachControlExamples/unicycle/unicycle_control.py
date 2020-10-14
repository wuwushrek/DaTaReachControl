import numpy as np

from DaTaReachControlExamples.MyopicDataDrivenControlTrue \
    import MyopicDataDrivenControlTrue
from DaTaReachControlExamples.MyopicDataDrivenControlContextGP \
    import MyopicDataDrivenControlContextGP
from DaTaReachControlExamples.MyopicDataDrivenControlSINDYc \
    import MyopicDataDrivenControlSINDYc
from DaTaReachControlExamples.unicycle.MyopicDataDrivenControlC2Opt \
    import MyopicDataDrivenControlC2Opt
from DaTaReachControlExamples.unicycle.MyopicDataDrivenControlDaTaControl\
    import MyopicDataDrivenControlDaTaControl

from DaTaReachControlExamples.unicycle.unicycle_param import *

import matplotlib.pyplot as plt

if save_plot_tex:
    import tikzplotlib

if realtime_plot:
  m_run_time_plot = runtimePlot
else:
  m_run_time_plot = None

# Warm-up for Numba the compile the code for later execution
# Should not be includded in the computation time --> compilation time
training_data_DaTa = {'trajectory': rand_init_traj_vec,
                 'trajectory_der': rand_init_traj_der_vec,
                 'input_seq': rand_init_input_vec,
                 'cost_grad': rand_init_cost_val_vec}
DaTa_ddc = MyopicDataDrivenControlDaTaControl(training_data_DaTa,
                  input_lb,
                  input_ub,
                  DaTaControlObjective,
                  sampling_time,
                  one_step_dyn=true_next_step,
                  exit_condition=exitCondition,
                  useGronwall=useGronwall, maxData=maxData,
                  threshUpdateApprox=threshUpdateLearn, verbCtrl=False,
                  params=params_solver)
res_DaTa_ddc = DaTa_ddc.solve(5,
                        runtime_info=None, verbose=False)

# Draw the initial environment and the initial random trajectory
ax = drawInitialPlot(xlim_tup, ylim_tup, target_position, cost_thresh,
                       initial_state, rand_init_traj_vec)

# Construct the initial data package for true one-step optimal control ########
training_data = {'trajectory': rand_init_traj_vec,
                 'input_seq': rand_init_input_vec,
                 'cost_val': rand_init_cost_val_vec}


true_ddc = MyopicDataDrivenControlTrue(training_data,
                                       input_lb,
                                       input_ub,
                                       trueObjective,
                                       one_step_dyn=true_next_step,
                                       exit_condition=exitCondition)

res_true_ddc = true_ddc.solve(max_iteration,
                                runtime_info=m_run_time_plot, verbose=True)

# Solve the problem using SINDyc ##############################################
sindyc_ddc = MyopicDataDrivenControlSINDYc(training_data,
                                           input_lb,
                                           input_ub,
                                           sindycObjective,
                                           sampling_time,
                                           one_step_dyn=true_next_step,
                                           exit_condition=exitCondition,
                                           cvxpy_args=cvxpy_args_sindyc)
res_sindyc_ddc = sindyc_ddc.solve(max_iteration,
                            runtime_info=m_run_time_plot, verbose=True)

# Solve the problem using CGP-LCB #############################################
gp_ddc = MyopicDataDrivenControlContextGP(training_data,
                                         stateToContext,
                                         context_u_lb,
                                         context_u_ub,
                                         boObjective,
                                         one_step_dyn=true_next_step,
                                         exit_condition=exitCondition,
                                         solver_style=acquistion_type)
res_gp_ddc = gp_ddc.solve(max_iteration,
                            runtime_info=m_run_time_plot, verbose=True)

# Solve the problem using C2Opt #############################################
# Compute the context+input vector and the associated costs and the gradients
rand_init_context_input_vec = np.hstack(
    (stateToContext(rand_init_traj_vec[:-1, :]),
     rand_init_input_vec))
rand_init_cost_val_vec_, rand_init_cost_grad_vec = \
        firstOrderOracle(rand_init_context_input_vec)

# Training data packaged for C2Opt
training_data_c2opt = {'trajectory': rand_init_traj_vec,
                 'input_seq': rand_init_input_vec,
                 'cost_val': rand_init_cost_val_vec_,
                 'cost_grad': rand_init_cost_grad_vec}

# Provide `congol` all the information available about the problem for C2Opt
c2opt_ddc = MyopicDataDrivenControlC2Opt(training_data_c2opt,
                  stateToContext,
                  context_u_lb,
                  context_u_ub,
                  firstOrderOracle,
                  grad_lips_constant_c2opt,
                  one_step_dyn=true_next_step,
                  exit_condition=exitCondition,
                  solver=solver_str_c2opt)
res_c2opt_ddc = c2opt_ddc.solve(max_iteration,
                            runtime_info=m_run_time_plot, verbose=True)

# Solve the problem using DaTaControl ########################################
training_data_DaTa = {'trajectory': rand_init_traj_vec,
                 'trajectory_der': rand_init_traj_der_vec,
                 'input_seq': rand_init_input_vec,
                 'cost_grad': rand_init_cost_val_vec}
DaTa_ddc = MyopicDataDrivenControlDaTaControl(training_data_DaTa,
                  input_lb,
                  input_ub,
                  DaTaControlObjective,
                  sampling_time,
                  one_step_dyn=true_next_step,
                  exit_condition=exitCondition,
                  useGronwall=useGronwall, maxData=maxData,
                  threshUpdateApprox=threshUpdateLearn, verbCtrl=False,
                  params=params_solver)
res_DaTa_ddc = DaTa_ddc.solve(max_iteration,
                        runtime_info=m_run_time_plot, verbose=True)

# Retrieve the optimal trajectory
opt_traj_vec = true_ddc.trajectory
opt_traj_vec = np.vstack((opt_traj_vec, true_ddc.current_state))
opt_cost_vec = computeCost(opt_traj_vec)      #true_ddc.cost_val_vec

# Retrieve the trajectory for SINDYc
sindyc_traj_vec = sindyc_ddc.trajectory
sindyc_traj_vec = np.vstack((sindyc_traj_vec,sindyc_ddc.current_state))
opt_sindyc_vec = computeCost(sindyc_traj_vec)

# Retrieve the trajectory for CGP-LCB
context_vec = gp_ddc.bo_step.X[:,:gp_ddc.context_arg_dim]
gpyopt_x, gpyopt_y, gpyopt_sh, gpyopt_ch = context_vec.T
gpyopt_traj_vec = np.vstack((gpyopt_x, gpyopt_y,
                                np.arctan2(gpyopt_sh, gpyopt_ch))).T
gpyopt_traj_vec = np.vstack((gpyopt_traj_vec,gp_ddc.current_state))
# Retrieve the cost vector achieved by GPyOpt (CGP-LCB)
gpyopt_cost_vec = computeCost(gpyopt_traj_vec)    #gp_ddc.bo_step.Y[:, 0]

# Retrieve the trajectory for C2Opt
context_vec = c2opt_ddc.contextual_optimizer.objective.arg[:,:c2opt_ddc.context_arg_dim]
c2opt_x, c2opt_y, c2opt_sh, c2opt_ch = context_vec.T
c2opt_traj_vec = np.vstack((c2opt_x, c2opt_y, np.arctan2(c2opt_sh, c2opt_ch))).T
c2opt_traj_vec = np.vstack((c2opt_traj_vec, c2opt_ddc.current_state))
c2opt_cost_vec = computeCost(c2opt_traj_vec)    #c2opt_ddc.contextual_optimizer.objective.fun

# Retrieve the trajectory for DaTaControl
data_traj_vec = DaTa_ddc.trajectory
data_traj_vec = np.vstack((data_traj_vec,DaTa_ddc.current_state))
data_cost_vec = computeCost(data_traj_vec)

# Plot the trajectories
ax = plt.gca()
if not realtime_plot:
  ax.plot(opt_traj_vec[:,0], opt_traj_vec[:,1], linestyle='-',
    marker=true_ddc.marker_type, markerSize=true_ddc.marker_default_size,
    color=true_ddc.marker_color, label=true_ddc.marker_label)
  ax.plot(sindyc_traj_vec[:,0], sindyc_traj_vec[:,1], linestyle='-',
    marker=sindyc_ddc.marker_type, color=sindyc_ddc.marker_color,
    markerSize=sindyc_ddc.marker_default_size, label=sindyc_ddc.marker_label)
  ax.plot(gpyopt_traj_vec[:,0], gpyopt_traj_vec[:,1], linestyle='-',
    marker=gp_ddc.marker_type, color=gp_ddc.marker_color,
    markerSize=gp_ddc.marker_default_size, label=gp_ddc.marker_label)
  ax.plot(c2opt_traj_vec[:,0], c2opt_traj_vec[:,1], linestyle='-',
    marker=c2opt_ddc.marker_type, color=c2opt_ddc.marker_color,
    markerSize=c2opt_ddc.marker_default_size, label=c2opt_ddc.marker_label)
  ax.plot(data_traj_vec[:,0], data_traj_vec[:,1], linestyle='-',
    marker=DaTa_ddc.marker_type, color=DaTa_ddc.marker_color,
    markerSize=DaTa_ddc.marker_default_size, label=DaTa_ddc.marker_label)

ax.legend(loc='center left', ncol=1, labelspacing=0.25, framealpha=1,
              bbox_to_anchor=(1.,0.5))
plt.draw()
plt.pause(0.01)
plt.tight_layout()
plt.savefig(log_file+"_trajectory"+log_extension_file, transparent=True)
if save_plot_tex:
    tikzplotlib.save(log_file+"_trajectory.tex")

# Plot the cost function
fig = plt.figure()
plt.plot(opt_cost_vec, linestyle='-', marker=true_ddc.marker_type,
            markerSize=true_ddc.marker_default_size, color=true_ddc.marker_color,
            label=true_ddc.marker_label)
plt.plot(opt_sindyc_vec, linestyle='-', marker=sindyc_ddc.marker_type,
            markerSize=sindyc_ddc.marker_default_size, color=sindyc_ddc.marker_color,
            label=sindyc_ddc.marker_label)
plt.plot(gpyopt_cost_vec, linestyle='-', marker=gp_ddc.marker_type,
            markerSize=gp_ddc.marker_default_size, color=gp_ddc.marker_color,
            label=gp_ddc.marker_label)
plt.plot(c2opt_cost_vec, linestyle='-', marker=c2opt_ddc.marker_type,
            markerSize=c2opt_ddc.marker_default_size, color=c2opt_ddc.marker_color,
            label=c2opt_ddc.marker_label)
plt.plot(data_cost_vec, linestyle='-', marker=DaTa_ddc.marker_type,
            markerSize=DaTa_ddc.marker_default_size, color=DaTa_ddc.marker_color,
            label=DaTa_ddc.marker_label)
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.5)
plt.xlabel(r'$\mathrm{Time\ step} $')
plt.ylabel(r'$\mathrm{Cost\ function}$')
plt.grid(True)
plt.tight_layout()
plt.savefig(log_file+"_cost"+log_extension_file, transparent=True)
if save_plot_tex:
    tikzplotlib.save(log_file+"_cost.tex")

###############################################
# Obtain the computation time of each approaches
def compute_time(res):
  lenRes = len(res)
  computeTime = np.zeros(lenRes)
  for i,elem in enumerate(res):
    computeTime[i] = elem['query_time']
  return computeTime

opt_time_vect = compute_time(res_true_ddc)
sindyc_time_vec = compute_time(res_sindyc_ddc)
gpyopt_time_vec = compute_time(res_gp_ddc)
c2opt_time_vec = compute_time(res_c2opt_ddc)
datasyctime_vec = compute_time(res_DaTa_ddc)

fig = plt.figure()
plt.plot(opt_time_vect, linestyle='-', marker=true_ddc.marker_type,
            markerSize=true_ddc.marker_default_size, color=true_ddc.marker_color,
            label=true_ddc.marker_label)
plt.plot(sindyc_time_vec, linestyle='-', marker=sindyc_ddc.marker_type,
            markerSize=sindyc_ddc.marker_default_size, color=sindyc_ddc.marker_color,
            label=sindyc_ddc.marker_label)
plt.plot(gpyopt_time_vec, linestyle='-', marker=gp_ddc.marker_type,
            markerSize=gp_ddc.marker_default_size, color=gp_ddc.marker_color,
            label=gp_ddc.marker_label)
plt.plot(c2opt_time_vec, linestyle='-', marker=c2opt_ddc.marker_type,
            markerSize=c2opt_ddc.marker_default_size, color=c2opt_ddc.marker_color,
            label=c2opt_ddc.marker_label)
plt.plot(datasyctime_vec, linestyle='-', marker=DaTa_ddc.marker_type,
            markerSize=DaTa_ddc.marker_default_size, color=DaTa_ddc.marker_color,
            label=DaTa_ddc.marker_label)
plt.gca().set_yscale('log')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.5)
plt.xlabel(r'$\mathrm{Time\ step} $')
plt.ylabel(r'$\mathrm{Compute\ time\ in\ log\ scale\ (s)}$')
plt.grid(True)
plt.tight_layout()
plt.savefig(log_file+"_time"+log_extension_file,transparent=True)
if save_plot_tex:
    tikzplotlib.save(log_file+"_time.tex")

plt.show()
