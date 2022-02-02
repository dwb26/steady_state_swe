#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <assert.h>
#include "solvers.h"
#include "particle_filters.h"

// This is the convection model

const double CFL_CONST = 0.8;

int weighted_double_cmp(const void * a, const void * b) {

	struct weighted_double d1 = * (struct weighted_double *) a;
	struct weighted_double d2 = * (struct weighted_double *) b;

	if (d1.x < d2.x)
		return -1;
	if (d2.x < d1.x)
		return 1;
	return 0;
}


void regression_fit(double * s, double * corrections, int N0, int N1, int M_poly, double * poly_weights, double * PHI, double * C, double * C_inv, double * MP, gsl_matrix * C_gsl, gsl_permutation * p, gsl_matrix * C_inv_gsl) {

	/* Set the values of the design matrix */
	int counter = 0, sg;
	for (int n = 0; n < N1; n++) {
		for (int m = 0; m < M_poly; m++)
			PHI[n * M_poly + m] = pow(s[N0 + counter], m);
		counter++;
	}

	/* Do C = PHI.T * PHI */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < M_poly; k++) {
			C[j * M_poly + k] = 0.0;
			for (int n = 0; n < N1; n++)
				C[j * M_poly + k] += PHI[n * M_poly + j] * PHI[n * M_poly + k];
		}
	}

	/* Invert C */
	for (int m = 0; m < M_poly * M_poly; m++)
		C_gsl->data[m] = C[m];
	gsl_linalg_LU_decomp(C_gsl, p, &sg);
	gsl_linalg_LU_invert(C_gsl, p, C_inv_gsl);
	counter = 0;
	for (int m = 0; m < M_poly; m++) {
		for (int n = 0; n < M_poly; n++) {
			C_inv[counter] = gsl_matrix_get(C_inv_gsl, m, n);
			counter++;
		}
	}

	/* Do C_inv * PHI.T */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < N1; k++) {
			MP[j * N1 + k] = 0.0;
			for (int n = 0; n < M_poly; n++)
				MP[j * N1 + k] += C_inv[j * M_poly + n] * PHI[k * M_poly + n];
		}
	}

	/* Compute the polynomial weights */
	for (int j = 0; j < M_poly; j++) {
		poly_weights[j] = 0.0;
		for (int n = 0; n < N1; n++)
			poly_weights[j] += MP[j * N1 + n] * corrections[n];
	}

}


double poly_eval(double x, double * poly_weights, int poly_degree) {

	double y_hat = 0.0;
	for (int m = 0; m < poly_degree + 1; m++)
		y_hat += poly_weights[m] * pow(x, m);
	return y_hat;

}


void resample(long size, double * w, long * ind, gsl_rng * r) {

	/* Generate the exponentials */
	double * e = (double *) malloc((size + 1) * sizeof(double));
	double g = 0;
	for (long i = 0; i <= size; i++) {
		e[i] = gsl_ran_exponential(r, 1.0);
		g += e[i];
	}
	/* Generate the uniform order statistics */
	double * u = (double *) malloc((size + 1) * sizeof(double));
	u[0] = 0;
	for (long i = 1; i <= size; i++)
		u[i] = u[i - 1] + e[i - 1] / g;

	/* Do the actual sampling with C_inv_gsl cdf */
	double cdf = w[0];
	long j = 0;
	for (long i = 0; i < size; i++) {
		while (cdf < u[i + 1]) {
			j++;
			cdf += w[j];
		}
		ind[i] = j;
	}

	free(e);
	free(u);
}


void mutate(int N_tot, double * theta, double * theta_res, double sig_sd, gsl_rng * rng, double theta_min, double theta_max) {
	for (int i = 0; i < N_tot; i++) {
		theta[i] = 0.9999 * theta_res[i] + gsl_ran_gaussian(rng, sig_sd);
		while ( (theta[i] < theta_min) || (theta[i] > theta_max) )
			theta[i] = 0.9999 * theta_res[i] + gsl_ran_gaussian(rng, sig_sd);
	}

}


void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx1 = hmm->nx;
	int nx0 = nxs[0];
	int obs_pos1 = nx1 - 1;
	int obs_pos0 = nx0 - 1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 1, M_poly = poly_degree + 1;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double dx1 = (space_right - space_left) / (double) (nx1 - 1);
	double dx0 = (space_right - space_left) / (double) (nx0 - 1);
	double obs, normaliser, abs_normaliser, x_hat, g0, g1;
	double theta_min = 0.5, theta_max = 7.5;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	double * thetas = (double *) malloc(N_tot * sizeof(double));
	double * res_thetas = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * solns0 = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc((poly_degree + 1) * sizeof(double));
	double * h1 = (double *) malloc(nx1 * sizeof(double));
	double * Z1 = (double *) malloc(nx1 * sizeof(double));
	double * Z_x1 = (double *) malloc(nx1 * sizeof(double));
	double * xs1 = (double *) malloc(nx1 * sizeof(double));
	double * h0 = (double *) malloc(nx0 * sizeof(double));
	double * Z0 = (double *) malloc(nx0 * sizeof(double));
	double * Z_x0 = (double *) malloc(nx0 * sizeof(double));
	double * xs0 = (double *) malloc(nx0 * sizeof(double));
	for (int j = 0; j < nx1; j++)
		xs1[j] = space_left + j * dx1;
	for (int j = 0; j < nx0; j++)
		xs0[j] = space_left + j * dx0;


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Initial conditions */
	/* ------------------ */
	double theta_init = hmm->signal[0];
	double h_init = hmm->h0, q_init = hmm->q0;
	h1[0] = h_init, h0[0] = h_init;
	for (int i = 0; i < N_tot; i++) {
		thetas[i] = gsl_ran_gaussian(rng, sig_sd) + theta_init;
		res_signs[i] = 1;
		while ( (thetas[i] < theta_min) || (thetas[i] > theta_max) )
			thetas[i] = gsl_ran_gaussian(rng, sig_sd) + theta_init;
	}


	/* Files */
	/* ----- */
	FILE * X_HATS = fopen("ml_xhats.txt", "w");



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0;



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 solutions																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			/* Fine solutions */
			solve(k, thetas[i], nx1, xs1, Z1, Z_x1, h1, dx1, q_init);
			solns1[i - N0] = h1[obs_pos1];

			/* Coarse solutions */
			solve(k, thetas[i], nx0, xs0, Z0, Z_x0, h0, dx0, q_init);
			solns0[i - N0] = h0[obs_pos0];

			/* Record the corrections samples for the regression approximation to the true correction curve */
			corrections[i - N0] = h1[obs_pos1] - h0[obs_pos0];

		}
		if (N1 > 0)
			regression_fit(thetas, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);




		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 weighting																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			g1 = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
			g0 = gsl_ran_gaussian_pdf(solns0[i - N0] + poly_eval(thetas[i], poly_weights, poly_degree) - obs, obs_sd);
			// g0 = gsl_ran_gaussian_pdf(solns0[i - N0] - obs, obs_sd);

			weights[i] = (g1 - g0) * (double) res_signs[i] / (double) N1;
			absolute_weights[i] = fabs(weights[i]);
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 0 weighting																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		double level0_dist;
		if (N1 > 0) {

			for (int i = 0; i < N0; i++) {

				/* Coarse solution */
				solve(k, thetas[i], nx0, xs0, Z0, Z_x0, h0, dx0, q_init);
				g0 = gsl_ran_gaussian_pdf(h0[obs_pos0] + poly_eval(thetas[i], poly_weights, poly_degree) - obs, obs_sd);
				// g0 = gsl_ran_gaussian_pdf(h0[obs_pos0] - obs, obs_sd);

				/* Weight computation */
				weights[i] = g0 * (double) res_signs[i] / (double) N0;
				absolute_weights[i] = fabs(weights[i]);
				normaliser += weights[i];
				abs_normaliser += absolute_weights[i];

			}

		}

		else {

			for (int i = 0; i < N0; i++) {

				/* Coarse solution */
				solve(k, thetas[i], nx0, xs0, Z0, Z_x0, h0, dx0, q_init);
				g0 = gsl_ran_gaussian_pdf(h0[obs_pos0] - obs, obs_sd);

				/* Weight computation */
				weights[i] = g0 * (double) res_signs[i] / (double) N0;
				absolute_weights[i] = fabs(weights[i]);
				normaliser += weights[i];
				abs_normaliser += absolute_weights[i];

			}

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			absolute_weights[i] /= abs_normaliser;
			weights[i] /= normaliser;
			signs[i] = weights[i] < 0 ? -1 : 1;
			ml_weighted[n][i].x = thetas[i];
			ml_weighted[n][i].w = weights[i];
			x_hat += thetas[i] * weights[i];
		}
		fprintf(X_HATS, "%e ", x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		for (int i = 0; i < N_tot; i++) {
			res_thetas[i] = thetas[ind[i]];
			res_signs[i] = signs[ind[i]];
		}
		mutate(N_tot, thetas, res_thetas, sig_sd, rng, theta_min, theta_max);

	}

	fclose(X_HATS);
	fclose(CORRECTIONS);
	fclose(REGRESSION_CURVE);

	free(signs);
	free(res_signs);
	free(ind);
	free(thetas);
	free(res_thetas);
	free(weights);
	free(absolute_weights);
	free(solns1);
	free(solns0);
	free(corrections);
	free(poly_weights);
	free(h1);
	free(Z1);
	free(Z_x1);
	free(xs1);
	free(h0);
	free(Z0);
	free(Z_x0);
	free(xs0);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);

}


void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx = hmm->nx;
	int obs_pos = nx - 1;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double obs, normaliser, x_hat;
	double theta_min = 0.5, theta_max = 7.5;
	long * ind = (long *) malloc(N * sizeof(long));
	double * thetas = (double *) malloc(N * sizeof(double));
	double * res_thetas = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * h = (double *) malloc(nx * sizeof(double));
	double * Z = (double *) malloc(nx * sizeof(double));
	double * Z_x = (double *) malloc(nx * sizeof(double));
	double * xs = (double *) malloc(nx * sizeof(double));
	for (int j = 0; j < nx; j++)
		xs[j] = space_left + j * dx;


	/* Initial conditions */
	/* ------------------ */
	double theta0 = hmm->signal[0];
	double h0 = hmm->h0, q0 = hmm->q0;
	h[0] = h0;
	for (int i = 0; i < N; i++) {
		thetas[i] = gsl_ran_gaussian(rng, sig_sd) + theta0;
		while ( (thetas[i] < theta_min) || (thetas[i] > theta_max) )
			thetas[i] = gsl_ran_gaussian(rng, sig_sd) + theta0;
	}

	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * X_HATS = fopen("x_hats.txt", "w");



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Weight generation																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		normaliser = 0.0;
		for (int i = 0; i < N; i++) {

			solve(k, thetas[i], nx, xs, Z, Z_x, h, dx, q0);
			weights[i] = gsl_ran_gaussian_pdf(h[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

		}


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = thetas[i];
			weighted[n][i].w = weights[i];
			x_hat += thetas[i] * weights[i];
		}
		fprintf(X_HATS, "%e ", x_hat);
		printf("n = %d, x_hat = %lf\n", n, x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			res_thetas[i] = thetas[ind[i]];
		mutate(N, thetas, res_thetas, sig_sd, rng, theta_min, theta_max);

	}

	fclose(CURVE_DATA);
	fclose(X_HATS);

	free(ind);
	free(thetas);
	free(res_thetas);
	free(weights);
	free(h);
	free(Z);
	free(Z_x);
	free(xs);

}


	// int mesh_size = 1000;
	// double mesh_incr;
	// double * theta_mesh = (double *) malloc(mesh_size * sizeof(double));

	// FILE * CORRECTIONS = fopen("corrections.txt", "w");
	// FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	// fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	// double theta_lo = 10.0, theta_hi = 0.0;


// fprintf(CORRECTIONS, "%e %e\n", thetas[i], corrections[i - N0]);

			// for (int l = 0; l < mesh_size; l++)
			// 	fprintf(REGRESSION_CURVE, "%e %e\n", theta_mesh[l], poly_eval(theta_mesh[l], poly_weights, poly_degree));


		// for (int i = 0; i < N_tot; i++) {
		// 	theta_lo = theta_lo < thetas[i] ? theta_lo : thetas[i];
		// 	theta_hi = theta_hi > thetas[i] ? theta_hi : thetas[i];
		// }
		// mesh_incr = (theta_hi - theta_lo) / (double) (mesh_size - 1);
		// for (int l = 0; l < mesh_size; l++)
		// 	theta_mesh[l] = theta_lo + l * mesh_incr;

