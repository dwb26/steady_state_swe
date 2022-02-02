import numpy as np
import matplotlib.pyplot as plt

curve_data_f = open("curve_data.txt", "r")
top_data_f = open("top_data.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")
length = int(hmm_data_f.readline())
hmm_data_f.readline()
space_left, space_right = list(map(float, hmm_data_f.readline().split()))
nx = int(hmm_data_f.readline())
k = float(hmm_data_f.readline())
h0, q0 = list(map(float, hmm_data_f.readline().split()))
thetas = np.empty(length)
obs = np.empty(length)
m = 0
for line in hmm_data_f:
	thetas[m], obs[m] = list(map(float, line.split()))
	m += 1
xs = np.linspace(space_left, space_right, nx)
curves = np.empty((length, nx))
Z = np.empty((length, nx))

n = 0
for line in curve_data_f:
	curves[n] = list(map(float, line.split()))
	n += 1

n = 0
for line in top_data_f:
	Z[n] = list(map(float, line.split()))
	n += 1

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 7))
fig.subplots_adjust(hspace=0.4)
m = 0
h_max = np.max(curves)
Z_min = np.min(Z)
for n in range(length):
	axs[m, n % 4].plot(xs, curves[n], label="h")
	axs[m, n % 4].plot(xs, Z[n], color="black", label="Z")
	axs[m, n % 4].plot(xs, curves[n] + Z[n], color="green", label="h + Z")
	axs[m, n % 4].set(ylim=(Z_min - 0.1, h_max + 0.1))
	axs[m, n % 4].set_title(r"$\theta_{} = {:.2f}$, $y_{} = {:.2f}$".format(n, thetas[n], n, obs[n]))
	axs[m, n % 4].set_xlabel("x")
	if (n + 1) %  4 == 0:
		m += 1
	if n == 0:
		axs[0, 0].legend()
plt.suptitle(r"Steady state ODE solutions with random walk scale parameter $\theta$ and shape k = {}".format(k))
plt.show()