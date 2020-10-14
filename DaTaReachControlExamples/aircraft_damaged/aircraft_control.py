import numpy as np

from DaTaReachControlExamples.MyopicDataDrivenControlTrue \
    import MyopicDataDrivenControlTrue
from DaTaReachControlExamples.MyopicDataDrivenControlContextGP \
    import MyopicDataDrivenControlContextGP
from DaTaReachControlExamples.MyopicDataDrivenControlSINDYc \
    import MyopicDataDrivenControlSINDYc
# from DaTaReachControlExamples.aircraft_damaged.MyopicDataDrivenControlC2Opt \
#     import MyopicDataDrivenControlC2Opt
from DaTaReachControlExamples.aircraft_damaged.MyopicDataDrivenControlDaTaControl\
    import MyopicDataDrivenControlDaTaControl

from DaTaReachControlExamples.aircraft_damaged.aircraft_param import *

import matplotlib.pyplot as plt

if save_plot_tex:
    import tikzplotlib

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
                  exit_condition=None,
                  useGronwall=useGronwall, maxData=maxData,
                  threshUpdateApprox=threshUpdateLearn, verbCtrl=False,
                  params=params_solver)
res_DaTa_ddc = DaTa_ddc.solve(5,
                        runtime_info=None, verbose=False)

# Draw the initial plot
save_figs = drawInitialPlot()

# Define the runtime info function
def runtimePlot(p_state, p_input, n_state, mt, mc, ms, ml, iterVal=0):
    return runtimePlot_aux(p_state, p_input, n_state, mt, mc, ms, ml, iterVal, save_figs)

if realtime_plot:
  m_run_time_plot = runtimePlot
else:
  m_run_time_plot = None

# Construct the initial data package for true one-step optimal control ########
training_data = {'trajectory': rand_init_traj_vec,
                 'input_seq': rand_init_input_vec,
                 'cost_val': rand_init_cost_val_vec}


true_ddc = MyopicDataDrivenControlTrue(training_data,
                                       input_lb,
                                       input_ub,
                                       trueObjective,
                                       one_step_dyn=true_next_step,
                                       exit_condition=None)

res_true_ddc = true_ddc.solve(max_iteration,
                                runtime_info=m_run_time_plot, verbose=True)

# # # Solve the problem using SINDyc ##############################################
# sindyc_ddc = MyopicDataDrivenControlSINDYc(training_data,
#                                            input_lb,
#                                            input_ub,
#                                            sindycObjective,
#                                            sampling_time,
#                                            one_step_dyn=true_next_step,
#                                            exit_condition=None,
#                                            cvxpy_args=cvxpy_args_sindyc,
#                                            sparse_thresh=sparse_thresh,
#                                            eps_thresh=eps_thresh,
#                                            scaling_sparse_min=scaling_sparse_min,
#                                            scaling_sparse_max=scaling_sparse_max,
#                                            libraryFun=libraryFun)
# stop_sindyc = 300
# res_sindyc_ddc = sindyc_ddc.solve(stop_sindyc,
#                             runtime_info=m_run_time_plot, verbose=True)

# # # Solve the problem using CGP-LCB #############################################
# gp_ddc = MyopicDataDrivenControlContextGP(training_data,
#                                          stateToContext,
#                                          state_lb,
#                                          state_ub,
#                                          boObjective,
#                                          one_step_dyn=true_next_step,
#                                          exit_condition=None,
#                                          solver_style=acquistion_type)
# res_gp_ddc = gp_ddc.solve(stop_sindyc,
#                             runtime_info=m_run_time_plot, verbose=True)

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
                  exit_condition=None,
                  useGronwall=useGronwall, maxData=maxData,
                  threshUpdateApprox=threshUpdateLearn, verbCtrl=False,
                  params=params_solver)
res_DaTa_ddc = DaTa_ddc.solve(max_iteration,
                        runtime_info=m_run_time_plot, verbose=True)


# Retrieve the optimal trajectory
opt_traj_vec = true_ddc.trajectory
opt_traj_vec = np.vstack((opt_traj_vec, true_ddc.current_state))
opt_cost_vec = computeCost(opt_traj_vec)      #true_ddc.cost_val_vec

# # Retrieve the trajectory for SINDYc
# sindyc_traj_vec = sindyc_ddc.trajectory
# sindyc_traj_vec = np.vstack((sindyc_traj_vec,sindyc_ddc.current_state))
# opt_sindyc_vec = computeCost(sindyc_traj_vec)

# # # Retrieve the trajectory for CGP-LCB
# context_vec = gp_ddc.bo_step.X[:,:gp_ddc.context_arg_dim]
# gpyopt_traj_vec = np.vstack((context_vec,gp_ddc.current_state))
# gpyopt_cost_vec = computeCost(gpyopt_traj_vec)


# Retrieve the trajectory for DaTaControl
data_traj_vec = DaTa_ddc.trajectory
data_traj_vec = np.vstack((data_traj_vec,DaTa_ddc.current_state))
data_cost_vec = computeCost(data_traj_vec)

# Plot the states evolution with time
time_vals = np.array([ i* sampling_time \
                        for i in range(n_data_max+max_iteration+1)])
for i, fig in enumerate(save_figs):
    ax = fig.gca()
    if not realtime_plot:
        ax.plot(time_vals, opt_traj_vec[:,i], linestyle='-',
            color=true_ddc.marker_color, label=true_ddc.marker_label)
        # ax.plot(time_vals[:(stop_sindyc+n_data_max+1)], sindyc_traj_vec[:,i], linestyle='-',
        #     color=sindyc_ddc.marker_color, label=sindyc_ddc.marker_label)
        # ax.plot(time_vals[:(stop_sindyc+n_data_max+1)], gpyopt_traj_vec[:,i], linestyle='-',
        #     color=gp_ddc.marker_color,label=gp_ddc.marker_label)
        ax.plot(time_vals, data_traj_vec[:,i], linestyle='-',
            color=DaTa_ddc.marker_color,label=DaTa_ddc.marker_label)
    ax.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left',
                columnspacing=1.5)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    fig.tight_layout()
    fig.savefig(log_file+"_x"+str(i)+log_extension_file, transparent=True)
    if save_plot_tex:
        tikzplotlib.save(log_file+"x_"+str(i)+".tex")


fig = plt.figure()
plt.plot(opt_cost_vec, linestyle='-', color=true_ddc.marker_color,
            label=true_ddc.marker_label)
# plt.plot(opt_sindyc_vec, linestyle='-', color=sindyc_ddc.marker_color,
#             label=sindyc_ddc.marker_label)
# plt.plot(gpyopt_cost_vec, linestyle='-', color=gp_ddc.marker_color,
#             label=gp_ddc.marker_label)
plt.plot(data_cost_vec, linestyle='-', color=DaTa_ddc.marker_color,
            label=DaTa_ddc.marker_label)
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(ncol=4, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.0)
plt.xlabel(r'$\mathrm{Time\ step} $')
plt.ylabel(r'$\mathrm{Cost\ function}$')
plt.grid(True)
plt.tight_layout()
plt.savefig(log_file+"_cost"+log_extension_file, transparent=True)
if save_plot_tex:
    tikzplotlib.save(log_file+"_cost.tex")

# Obtain the computation time of each approaches
def compute_time(res):
  lenRes = len(res)
  computeTime = np.zeros(lenRes)
  for i,elem in enumerate(res):
    computeTime[i] = elem['query_time']
  return computeTime

opt_time_vect = compute_time(res_true_ddc)
# sindyc_time_vec = compute_time(res_sindyc_ddc)
# gpyopt_time_vec = compute_time(res_gp_ddc)
# c2opt_time_vec = compute_time(res_c2opt_ddc)
datasyctime_vec = compute_time(res_DaTa_ddc)

fig = plt.figure()
plt.plot(opt_time_vect, linestyle='-', color=true_ddc.marker_color,
            label=true_ddc.marker_label)
# plt.plot(sindyc_time_vec, linestyle='-', color=sindyc_ddc.marker_color,
#             label=sindyc_ddc.marker_label)
# plt.plot(gpyopt_time_vec, linestyle='-', color=gp_ddc.marker_color,
#             label=gp_ddc.marker_label)
plt.plot(datasyctime_vec, linestyle='-', color=DaTa_ddc.marker_color,
            label=DaTa_ddc.marker_label)
plt.gca().set_yscale('log')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(ncol=4, bbox_to_anchor=(0,1), loc='lower left', columnspacing=2.0)
plt.xlabel(r'$\mathrm{Time\ step} $')
plt.ylabel(r'$\mathrm{Compute\ time\ in\ log\ scale\ (s)}$')
plt.grid(True)
plt.tight_layout()
plt.savefig(log_file+"_time"+log_extension_file,transparent=True)
if save_plot_tex:
    tikzplotlib.save(log_file+"_time.tex")

plt.show()
