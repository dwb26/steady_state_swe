import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

hmm_f = open("hmm_data.txt", "r")
data = open("curve_data.txt", "r")

length = int(hmm_f.readline())
signal = np.empty(length)
sig_sd, obs_sd = list(map(float, hmm_f.readline().split()))
space_left, space_right = list(map(float, hmm_f.readline().split()))
# for i in range(2):
	# hmm_f.readline().split()
nx = int(hmm_f.readline())
for i in range(4):
	hmm_f.readline().split()
for i in range(length):
	signal[i] = list(map(float, hmm_f.readline().split()))[0]

m = 0; n = 0
curves = []
counters = np.zeros(length, dtype=int)
for line in data:
	if m % 2 == 0:
		curves.extend(list(map(float, line.split())))
	else:
		counters[n] = list(map(int, line.split()))[0]
		n += 1
	m += 1
total_length = np.sum(counters)
curve_arr = np.array(curves).reshape((total_length, nx))

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
xs = np.linspace(space_left, space_right, nx)
# line, = ax.plot(xs, curve_arr[0])
line, = ax.plot(xs, np.log10(curve_arr[0]))

def update(n):
	line.set_data(xs, np.log10(curve_arr[n]))
	ax.set_title("iterate = {} / {}".format(n, total_length))
	ax.set(ylim=(0, np.max(np.log10(curve_arr))))
	return line,

ani = animation.FuncAnimation(fig, func=update, frames=range(1, total_length, 3))
plt.show()