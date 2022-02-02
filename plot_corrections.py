import numpy as np
import matplotlib.pyplot as plt

corrections_f = open("corrections.txt", "r")
regression_f = open("regression_curve.txt", "r")

length = 8; N1 = 50
mesh_size = int(regression_f.readline())
thetas = []; corrections = []
mesh = []; points = []
for line in corrections_f:
	theta, correction = list(map(float, line.split()))
	thetas.append(theta)
	corrections.append(correction)
thetas = np.array(thetas).reshape((length, N1))
corrections = np.array(corrections).reshape((length, N1))
for line in regression_f:
	m, p = list(map(float, line.split()))
	mesh.append(m)
	points.append(p)
theta_mesh = np.array(mesh).reshape((length, mesh_size))
regression_points = np.array(points).reshape((length, mesh_size))

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 7))
m = 0
for n in range(length):
	axs[m, n % 4].scatter(thetas[n], corrections[n], s=2, color="red")
	axs[m, n % 4].plot(theta_mesh[n], regression_points[n])
	if (n + 1) % 4 == 0:
		m += 1
plt.suptitle(r"Degree 2 correction regression fit for mesh configuration $(nx_0, nx_1) = ({}, {})$".format(25, 250))
plt.show()