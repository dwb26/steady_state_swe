#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "solvers.h"

void gen_Z_topography(double * xs, double * Z, int nx, double k, double theta) {
	for (int j = 0; j < nx; j++)
		Z[j] = -100.0 * pow(xs[j], k - 1) * exp(-xs[j] / theta) / (tgamma(k) * pow(theta, k));
}


void gen_Z_x_topography(double * xs, double * Z_x, int nx, double k, double theta) {
	for (int j = 0; j < nx; j++)
		Z_x[j] = -100.0 * exp(-xs[j] / theta) / (tgamma(k) * pow(theta, k)) * ((k - 1) * pow(xs[j], k - 2) - pow(xs[j], k - 1) / theta);
}


double target(double h, double Z_x, double q0) {
	double g = 9.81;
	return q0 * q0 / (g * h * h * h) - Z_x;
}


double RK4(double hn, double Z_xn, double dx, double q0) {

	double k1, k2, k3, k4;
	k1 = target(hn, Z_xn, q0);
	k2 = target(hn + 0.5 * dx * k1, Z_xn + 0.5 * dx * k1, q0);
	k3 = target(hn + 0.5 * dx * k2, Z_xn + 0.5 * dx * k2, q0);
	k4 = target(hn + dx * k3, Z_xn + dx * k3, q0);

	return hn + 1 / 6.0 * dx * (k1 + 2 * k2 + 2 * k3 + k4);
}


void solve(double k, double theta, int nx, double * xs,double * Z, double * Z_x, double * h, double dx, double q0) {
	gen_Z_topography(xs, Z, nx, k, theta);
	gen_Z_x_topography(xs, Z_x, nx, k, theta);
	for (int j = 1; j < nx; j++)
		h[j] = RK4(h[j - 1], Z_x[j - 1], dx, q0);
}