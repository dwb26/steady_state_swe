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
#include <assert.h>
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

const int N_TOTAL_MAX2 = 100000000;
const int N_LEVELS2 = 2;
const int N_MESHES2 = 9;
const int N_ALLOCS2 = 7;

/// This is the steady state ode model

void generate_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE)  {

	int run_ref = 1;		// REF ON
	// int run_ref = 0;		// REF OFF

	/* Reference distribution */
	/* ---------------------- */
	if (run_ref == 1)
		run_reference_filter(hmm, N_ref, rng, weighted_ref, n_data);
	else
		read_cdf(weighted_ref, hmm, n_data);


	/* Sample allocation */
	/* ----------------- */
	double T, T_temp;
	T = perform_BPF_trials(hmm, N_bpf, rng, N_trials, N_ref, weighted_ref, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE);
	if (n_data == 0)
		compute_sample_sizes(hmm, rng, level0_meshes, T, N0s, N1s, N_bpf, N_trials);
	else
		T_temp = read_sample_sizes(hmm, N0s, N1s, N_trials);

}


void generate_hmm(gsl_rng * rng, int n_data, int length, int nx) {

	/** 
	Generates the HMM data and outputs to file to be read in by read_hmm.
	*/

	int obs_pos = nx - 1;
	double sig_sd = 0.5;
	double obs_sd = 0.25;
	double space_left = 0.0, space_right = 25.0;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double k = 5.0, theta = 2.0, obs, g = 9.81;
	double theta_min = 0.5, theta_max = 7.5;
	double h0 = 5.0, q0 = 5.0;
	double * h = (double *) malloc(nx * sizeof(double));
	double * Z = (double *) malloc(nx * sizeof(double));
	double * Z_x = (double *) malloc(nx * sizeof(double));
	double * xs = (double *) malloc(nx * sizeof(double));
	for (int j = 0; j < nx; j++)
		xs[j] = space_left + j * dx;
	h[0] = h0;

	/* Write the available parameters */
	FILE * DATA = fopen("hmm_data.txt", "w");
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * TOP_DATA = fopen("top_data.txt", "w");
	fprintf(DATA, "%d\n", length);
	fprintf(DATA, "%lf %lf\n", sig_sd, obs_sd);
	fprintf(DATA, "%lf %lf\n", space_left, space_right);
	fprintf(DATA, "%d\n", nx);
	fprintf(DATA, "%lf\n", k);
	fprintf(DATA, "%lf %lf\n", h0, q0);

	/* Generate the data */
	for (int n = 0; n < length; n++) {

		solve(k, theta, nx, xs, Z, Z_x, h, dx, q0);
		obs = h[obs_pos] + gsl_ran_gaussian(rng, obs_sd);
		printf("true obs = %lf\n", h[obs_pos]);

		fprintf(DATA, "%e %e\n", theta, obs);
		for (int j = 0; j < nx; j++) {
			fprintf(CURVE_DATA, "%e ", h[j]);
			fprintf(TOP_DATA, "%e ", Z[j]);
		}
		fprintf(CURVE_DATA, "\n");
		fprintf(TOP_DATA, "\n");

		/* Evolve the signal with the mutation model */
		theta = 0.9999 * theta + gsl_ran_gaussian(rng, sig_sd);
		while ( (theta < theta_min) || (theta > theta_max) )
			theta = 0.9999 * theta + gsl_ran_gaussian(rng, sig_sd);

	}

	fclose(CURVE_DATA);
	fclose(DATA);
	fclose(TOP_DATA);

	free(h);
	free(Z);
	free(Z_x);
	free(xs);

}


void read_hmm(HMM * hmm, int n_data) {

	FILE * DATA = fopen("hmm_data.txt", "r");

	fscanf(DATA, "%d\n", &hmm->length);
	fscanf(DATA, "%lf %lf\n", &hmm->sig_sd, &hmm->obs_sd);
	fscanf(DATA, "%lf %lf\n", &hmm->space_left, &hmm->space_right);
	fscanf(DATA, "%d\n", &hmm->nx);
	fscanf(DATA, "%lf\n", &hmm->k);
	fscanf(DATA, "%lf %lf\n", &hmm->h0, &hmm->q0);

	hmm->signal = (double *) malloc(hmm->length * sizeof(double));
	hmm->observations = (double *) malloc(hmm->length * sizeof(double));
	for (int n = 0; n < hmm->length; n++)
		fscanf(DATA, "%lf %lf\n", &hmm->signal[n], &hmm->observations[n]);
	fclose(DATA);

	printf("DATA SET %d\n", n_data);
	printf("Data length 	             = %d\n", hmm->length);
	printf("sig_sd      	             = %lf\n", hmm->sig_sd);
	printf("obs_sd      	             = %lf\n", hmm->obs_sd);
	printf("nx          	             = %d\n", hmm->nx);
	printf("k           	             = %lf\n", hmm->k);
	printf("h0, q0         	             = %lf %lf\n", hmm->h0, hmm->q0);
	for (int n = 0; n < hmm->length; n++)
		printf("n = %d: signal = %lf, observation = %lf\n", n, hmm->signal[n], hmm->observations[n]);

}


void run_reference_filter(HMM * hmm, int N_ref, gsl_rng * rng, w_double ** weighted_ref, int n_data) {

	double ref_elapsed;
	char sig_sd_str[50], obs_sd_str[50], len_str[50], s0_str[50], n_data_str[50], ref_name[200];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(s0_str, 50, "%lf", hmm->signal[0]);
	snprintf(n_data_str, 50, "%d", n_data);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_len=%s_s0=%s_n_data=%s.txt", sig_sd_str, obs_sd_str, len_str, s0_str, n_data_str);
	puts(ref_name);

	/* Run the BPF with the reference number of particles */
	printf("Running reference BPF...\n");
	clock_t ref_timer = clock();
	bootstrap_particle_filter(hmm, N_ref, rng, weighted_ref);
	ref_elapsed = (double) (clock() - ref_timer) / (double) CLOCKS_PER_SEC;
	printf("Reference BPF for %d particles completed in %f seconds\n", N_ref, ref_elapsed);

	/* Sort and output the weighted particles for the KS tests */
	for (int n = 0; n < hmm->length; n++)
		qsort(weighted_ref[n], N_ref, sizeof(w_double), weighted_double_cmp);
	output_cdf(weighted_ref, hmm, N_ref, ref_name);

}


void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[200]) {

	FILE * data = fopen(file_name, "w");
	fprintf(data, "%d %d\n", N, hmm->length);

	for (int n = 0; n < hmm->length; n++) {
		for (int i = 0; i < N; i++)
			fprintf(data, "%e ", w_particles[n][i].x);
		fprintf(data, "\n");
		for (int i = 0; i < N; i++)
			fprintf(data, "%e ", w_particles[n][i].w);
		fprintf(data, "\n");
	}
	fclose(data);
}


void read_cdf(w_double ** w_particles, HMM * hmm, int n_data) {

	int N, length;
	char sig_sd_str[50], obs_sd_str[50], len_str[50], s0_str[50], n_data_str[50], ref_name[200];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(s0_str, 50, "%lf", hmm->signal[0]);
	snprintf(n_data_str, 50, "%d", n_data);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_len=%s_s0=%s_n_data=%s.txt", sig_sd_str, obs_sd_str, len_str, s0_str, n_data_str);
	FILE * data = fopen(ref_name, "r");
	fscanf(data, "%d %d\n", &N, &length);

	for (int n = 0; n < length; n++) {
		for (int i = 0; i < N; i++)
			fscanf(data, "%lf ", &w_particles[n][i].x);
		for (int i = 0; i < N; i++)
			fscanf(data, "%lf ", &w_particles[n][i].w);
	}
	fclose(data);
}


double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE) {

	int length = hmm->length;
	double ks = 0.0, elapsed = 0.0, mse = 0.0;
	w_double ** weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted[n] = (w_double *) malloc(N_bpf * sizeof(w_double));

	printf("Running BPF trials...\n");
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {

		printf("n_trial = %d\n", n_trial);

		/* Run the simulation for the BPF */
		clock_t bpf_timer = clock();
		bootstrap_particle_filter(hmm, N_bpf, rng, weighted);
		elapsed = (double) (clock() - bpf_timer) / (double) CLOCKS_PER_SEC;

		/* Compute the KS statistic for the run */
		ks = 0.0;
		for (int n = 0; n < length; n++) {
			qsort(weighted[n], N_bpf, sizeof(w_double), weighted_double_cmp);
			ks += ks_statistic(N_ref, weighted_ref[n], N_bpf, weighted[n]) / (double) length;
		}

		mse = compute_mse(weighted_ref, weighted, length, N_ref, N_bpf);
		fprintf(RAW_BPF_TIMES, "%e ", elapsed);
		fprintf(RAW_BPF_KS, "%e ", ks);
		fprintf(RAW_BPF_MSE, "%e ", mse);

	}

	fprintf(RAW_BPF_TIMES, "\n");
	fprintf(RAW_BPF_KS, "\n");
	fprintf(RAW_BPF_MSE, "\n");
	
	free(weighted);

	return elapsed;

}


void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials) {


	/* Variables to compute the sample sizes */
	/* ------------------------------------- */
	int N0, N0_lo, dist;
	double T_mlbpf, diff;
	clock_t timer;
	int N1_incr = (int) (N_bpf / (double) N_ALLOCS2);


	/* Variables to run the MLBPF */
	/* -------------------------- */
	int length = hmm->length;
	int nxs[N_LEVELS2] = { 0, hmm->nx };
	int * sample_sizes = (int *) malloc(N_LEVELS2 * sizeof(int));
	double * sign_ratios = (double *) malloc(length * sizeof(double));
	w_double ** ml_weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		ml_weighted[n] = (w_double *) malloc(N_TOTAL_MAX2 * sizeof(w_double));


	/* Variables for printing to file */
	/* ------------------------------ */
	FILE * N0s_f = fopen("N0s_data.txt", "w");
	fprintf(N0s_f, "%d %e\n", N_bpf, T);


	/* Compute the particle allocations */
	/* -------------------------------- */
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {

		nxs[0] = level0_meshes[i_mesh];
		printf("Computing the level 0 allocations for nx0 = %d\n", nxs[0]);

		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++) {

			sample_sizes[1] = N1s[n_alloc];
			printf("N1 = %d\n", N1s[n_alloc]);

			N0 = N_bpf;
			sample_sizes[0] = N0;
			N0_lo = N0;

			/* Find a value for N0_init that exceeds the required time */
			clock_t timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff < 0) {
				N0 *= 2;
				sample_sizes[0] = N0;

				timer = clock();
				ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;
			}

			/* Find a value for N0_lo that does not meet the required time */
			sample_sizes[0] = N0_lo;
			timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff > 0) {
				N0_lo = (int) (N0_lo / 2.0);
				sample_sizes[0] = N0_lo;

				timer = clock();
				ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;

				if (N0_lo == 0)
					diff = 0;
			}

			/* Run with the N0 we know exceeds the required time */
			sample_sizes[0] = N0;
			timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;

			if (N0_lo == 0)
				sample_sizes[0] = 0;

			else {

				/* Halve the interval until a sufficiently accurate root is found */
				while (fabs(diff) >= 0.01) {
					if (diff > 0)
						N0 = (int) (0.5 * (N0_lo + N0));
					else {
						dist = N0 - N0_lo;
						N0_lo = N0;
						N0 += dist;
					}
					sample_sizes[0] = N0;

					timer = clock();
					for (int i = 0; i < 1; i++)
						ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
					T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
					diff = (T_mlbpf - T) / T;

					if (N0_lo == N0)
						diff = 0.0;
				}
			}


			N0s[i_mesh][n_alloc] = sample_sizes[0];
			printf("N0 = %d for N1 = %d and nx0 = %d, timed diff = %.10lf\n", sample_sizes[0], N1s[n_alloc], nxs[0], diff);
			printf("\n");
			fprintf(N0s_f, "%d ", sample_sizes[0]);

		}

		fprintf(N0s_f, "\n");

	}

	fclose(N0s_f);

	free(sign_ratios);
	free(ml_weighted);
	free(sample_sizes);

}


double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials) {

	int N_bpf;
	double T;
	FILE * N0s_f = fopen("N0s_data.txt", "r");
	fscanf(N0s_f, "%d %lf\n", &N_bpf, &T);
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
			fscanf(N0s_f, "%d ", &N0s[i_mesh][n_alloc]);
	}

	for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
		printf("N1[%d] = %d ", n_alloc, N1s[n_alloc]);
	printf("\n");
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
			printf("N0[%d] = %d ", n_alloc, N0s[i_mesh][n_alloc]);
		printf("\n");
	}

	fclose(N0s_f);
	return T;

}


double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted) {

	double record, diff;
	double cum1 = 0, cum2 = 0;
	int j = 0, lim1, lim2;
	w_double * a1;
	w_double * a2;

	if (weighted_ref[0].x < weighted[0].x) {
		a1 = weighted_ref;
		a2 = weighted;
		lim1 = N_ref;
		lim2 = N;
	}
	else {
		a1 = weighted;
		a2 = weighted_ref;
		lim1 = N;
		lim2 = N_ref;
	}

	cum1 = a1[0].w;
	record = cum1;
	for (int i = 1; i < lim1; i++) {
		while (a2[j].x < a1[i].x && j < lim2) {
			cum2 += a2[j].w;
			diff = fabs(cum2 - cum1);
			record = diff > record ? diff : record;
			j++;
		}
		cum1 += a1[i].w;
		diff = fabs(cum2 - cum1);
		record = diff > record ? diff : record;
	}
	return record;
}


double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2) {

	double mse = 0.0, x1_hat, x2_hat, w1_sum, w2_sum;

	for (int n = 0; n < length; n++) {
		x1_hat = 0.0, x2_hat = 0.0, w1_sum = 0.0, w2_sum = 0.0;
		for (int i = 0; i < N1; i++) {
			x1_hat += weighted1[n][i].w * weighted1[n][i].x;
			w1_sum += weighted1[n][i].w;
		}
		for (int i = 0; i < N2; i++) {
			x2_hat += weighted2[n][i].w * weighted2[n][i].x;
			w2_sum += weighted2[n][i].w;
		}
		mse = mse + (x1_hat - x2_hat) * (x1_hat - x2_hat);
	}
	return mse / (double) length;
}





