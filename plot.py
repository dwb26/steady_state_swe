import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

hmm_data = open("hmm_data.txt", "r")
raw_times = open("raw_times.txt", "r")
raw_mse = open("raw_mse.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")
N1s_data = open("N1s_data.txt", "r")
raw_bpf_times = open("raw_bpf_times.txt", "r")
raw_bpf_mse = open("raw_bpf_mse.txt", "r")
alloc_counters_f = open("alloc_counters.txt", "r")
ref_stds_f = open("ref_stds.txt", "r")


# --------------------------------------------------- HMM data ------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))
length = int(hmm_data.readline())
sig_sd, obs_sd = list(map(float, hmm_data.readline().split()))
hmm_data.readline()
nx = int(hmm_data.readline())
for n in range(3):
    hmm_data.readline()
lag = int(hmm_data.readline())


# -------------------------------------------------- BPF results ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
bpf_times = []; bpf_ks = []; bpf_mse = []
for n_data in range(N_data):
	bpf_times.extend(list(map(float, raw_bpf_times.readline().split())))
	bpf_mse.extend(list(map(float, raw_bpf_mse.readline().split())))

bpf_mse_new = []
for x in bpf_mse:
	if not pd.isna(x):
		bpf_mse_new.append(x)
bpf_mse = bpf_mse_new
bpf_rmse = np.sqrt(bpf_mse)
bpf_mean_mse_log10 = np.mean(np.log10(bpf_mse))
bpf_median_mse_log10 = np.median(np.log10(bpf_mse))
bpf_mean_rmse_log10 = np.mean(np.log10(bpf_rmse))
bpf_median_rmse_log10 = np.median(np.log10(bpf_rmse))
ref_stds = np.array(list(map(float, ref_stds_f.readline().split())))



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
mse_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
ks_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
srs_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
glob_min_mse = 1
glob_max_mse = 0
eps = 1e-06
for i_mesh in range(N_MESHES):
	for n_alloc in range(N_ALLOCS):
		mse_arr[i_mesh, n_alloc, :] = list(map(float, raw_mse.readline().split()))
		for x in mse_arr[i_mesh, n_alloc, :]:
			if not np.abs(x - -1) < eps and not np.abs(x - -2) < eps:
				if x < glob_min_mse:
					glob_min_mse = x
				if x > glob_max_mse:
					glob_max_mse = x
glob_min_mse = np.log10(glob_min_mse)
glob_max_mse = np.log10(glob_max_mse)
N_NaNs = np.zeros((N_MESHES, N_ALLOCS))
for n_trial in range(N_trials * N_data):
	for i_mesh in range(N_MESHES):
		for n_alloc in range(N_ALLOCS):

			# These are NaNs because of the N1 allocation expiration
			if np.abs(mse_arr[i_mesh, n_alloc, n_trial] - -1) < eps:
				mse_arr[i_mesh, n_alloc, n_trial] = float("NaN")

			# These are NaNs from the MSE results
			if np.abs(mse_arr[i_mesh, n_alloc, n_trial] - -2) < eps:
				N_NaNs[i_mesh, n_alloc] += 1
				mse_arr[i_mesh, n_alloc, n_trial] = float("NaN")
rmse_arr = np.sqrt(mse_arr)



# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
colors = ["mediumpurple", "royalblue", "powderblue", "mediumseagreen", "greenyellow", "orange", "tomato", "firebrick"]
fig_width = 8; fig_height = 7
hspace = 0.9
fig1, axs = plt.subplots(nrows=N_MESHES, ncols=1, figsize=(fig_width, fig_height))
fig2, axs2 = plt.subplots(nrows=4, ncols=1, figsize=(fig_width, fig_height))
fig1.subplots_adjust(hspace=hspace)
fig1.suptitle(r"N_data = {}, N_trials = {}, nx = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}".format(N_data, N_trials, nx, N_bpf, "s", sig_sd, "o", obs_sd, length))
fig2.subplots_adjust(hspace=0.9)
fig2.suptitle(r"N_data = {}, N_trials = {}, nx = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}".format(N_data, N_trials, nx, N_bpf, "s", sig_sd, "o", obs_sd, length))



# ------------------------------------------------------------------------------------------------------------------- #
#
# Boxplots
#
# ------------------------------------------------------------------------------------------------------------------- #
if N_MESHES > 1:
	for i_mesh in range(N_MESHES):
		ax = sns.boxplot(data=pd.DataFrame(np.log10(mse_arr[i_mesh].T), columns=N1s), ax=axs[i_mesh], color=colors[i_mesh], whis=1000)
		ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
		ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
		ax.set(ylim=(glob_min_mse, glob_max_mse))
		if i_mesh == N_MESHES - 1:
			ax.set_xlabel(r"$N_1$")
else:
	for i_mesh in range(N_MESHES):
		ax = sns.boxplot(data=pd.DataFrame(np.log10(mse_arr[i_mesh].T), columns=N1s), ax=axs, color=colors[i_mesh])
		ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
		ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
		ax.set(ylim=(glob_min_mse, glob_max_mse))
		if i_mesh == N_MESHES - 1:
			ax.set_xlabel(r"$N_1$")



# ------------------------------------------------------------------------------------------------------------------- #
#
# Mean MSE from reference point estimates
#
# ------------------------------------------------------------------------------------------------------------------- #
# axs2[0].set_title("log10(Mean MSE)", fontsize=9)
# axs2[0].plot(N1s[:max_allocs], bpf_mean_mse_log10 * np.ones(max_allocs), color="black", label="BPF")
# for i_mesh in range(N_MESHES):
# 	mses = mse_arr[i_mesh].T
# 	means = []
# 	for n_alloc in range(alloc_counters[i_mesh]):
# 		mses_new = []
# 		for x in mses[:, n_alloc]:
# 			if not pd.isna(x):
# 				mses_new.append(x)
# 		means.append(np.mean(np.log10(mses_new)))
# 	# axs2[0].plot(N1s[:alloc_counters[i_mesh]], np.mean(np.log10(mse_arr[i_mesh].T), axis=0), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# 	axs2[0].plot(N1s[:alloc_counters[i_mesh]], means, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# axs2[0].set_xticks([])



# ------------------------------------------------------------------------------------------------------------------- #
#
# Median RMSE from reference point estimates
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[0].set_title("log10(Median RMSE)", fontsize=9)
axs2[0].plot(N1s[:max_allocs], bpf_median_rmse_log10 * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_MESHES):
	rmses = rmse_arr[i_mesh].T
	medians = []
	for n_alloc in range(alloc_counters[i_mesh]):
		rmses_new = []
		for x in rmses[:, n_alloc]:
			if not pd.isna(x):
				rmses_new.append(x)
		medians.append(np.median(np.log10(rmses_new)))
	axs2[0].plot(N1s[:alloc_counters[i_mesh]], medians, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
axs2[0].legend(loc=1, prop={'size': 7})
axs2[0].set_xlabel(r"$N_1$")



# ------------------------------------------------------------------------------------------------------------------- #
#
# KS statistics from the reference distribution
#
# ------------------------------------------------------------------------------------------------------------------- #
# axs2[1].set_title("log10(Mean KS statistics)", fontsize=9)
# axs2[1].plot(N1s[:max_allocs], bpf_mean_ks_log10 * np.ones(max_allocs), color="black", label="BPF")
# for i_mesh in range(N_MESHES):
# 	axs2[1].plot(N1s[:alloc_counters[i_mesh]], np.log10(np.mean(ks_arr[i_mesh, :alloc_counters[i_mesh], :].T, axis=0)), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# axs2[1].set_xticks([])



# ------------------------------------------------------------------------------------------------------------------- #
#
# NaN ratios
#
# ------------------------------------------------------------------------------------------------------------------- #
N_total = N_data * N_trials
axs2[1].set_title("Proportion of NaNs", fontsize=9)
for i_mesh in range(N_MESHES):
    axs2[1].plot(N1s[:alloc_counters[i_mesh]], N_NaNs[i_mesh, :alloc_counters[i_mesh]] / N_total, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
axs2[1].set_xlabel(r"$N_1$")
axs2[1].legend(loc=1)



# ------------------------------------------------------------------------------------------------------------------- #
#
# Reference standard deviations
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[2].set_title("Ref stds", fontsize=9)
for n in range(N_data):
	axs2[2].vlines(n * length, np.min(ref_stds), np.max(ref_stds), color="black")
axs2[2].scatter(range(N_data * length), ref_stds, s=2)
axs2[2].set_xlabel("Total iterate")



# ------------------------------------------------------------------------------------------------------------------- #
#
# Trial times
#
# ------------------------------------------------------------------------------------------------------------------- #
times_list = np.array(list(map(float, raw_times.readline().split())))
total_time_length = len(times_list)
axs2[3].set_title("Trial times", fontsize=9)
axs2[3].plot(range(total_time_length), np.array(times_list).flatten(), linewidth=0.5)
axs2[3].plot(range(total_time_length), np.mean(bpf_times) * np.ones(total_time_length), label="bpf mean", color="black")
axs2[3].set_xlabel("Trial")

plt.show()



