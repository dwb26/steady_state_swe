// gcc -o equal_runtimes -lm -lgsl -lgslcblas equal_runtimes.c particle_filters.c solvers.c generate_model.c
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
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

const int N_TOTAL_MAX = 100000000;
const int N_LEVELS = 2;
const int N_MESHES = 9;
const int N_ALLOCS = 7;

void output_parameters(int N_trials, int * level0_meshes, int nx, int * N1s, int N_data, int N_bpf);
void record_reference_data(HMM * hmm, w_double ** weighted_ref, int N_ref, FILE * FULL_HMM_DATA, FILE * FULL_REF_DATA, FILE * REF_STDS);
void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, int * alloc_counters, FILE * ALLOC_COUNTERS, FILE * RAW_TIMES, FILE * RAW_KS, FILE * RAW_MSE, FILE * RAW_SRS, int N_data);

static int compare (const void * a, const void * b)
{
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0;  
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// This is the steady state ode model
int main(void) {

	clock_t timer = clock();
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);


	/* Main experiment parameters */
	/* -------------------------- */
	int N_data = 1;
	int N_trials = 100;
	int length = 30, nx = 250;
	int N_ref = 100000;
	int N_bpf = 250;
	int level0_meshes[N_MESHES] = { 225, 200, 175, 150, 125, 100, 75, 50, 25 };
	int N1s[N_ALLOCS] = { 0, 3, 50, 100, 150, 200, 249 };
	int nxs[N_LEVELS] = { 0, nx };
	int ** N0s = (int **) malloc(N_MESHES * sizeof(int *));	
	int * sample_sizes = (int *) malloc(N_LEVELS * sizeof(int));		
	int * alloc_counters = (int *) malloc(N_MESHES * sizeof(int));
	double *** raw_ks = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_mse = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_times = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_srs = (double ***) malloc(N_MESHES * sizeof(double **));
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		N0s[i_mesh] = (int *) malloc(N_ALLOCS * sizeof(int));
		raw_ks[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_mse[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_times[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_srs[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {
			raw_ks[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_mse[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_times[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_srs[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));	
		}
	}
	w_double ** weighted_ref = (w_double **) malloc(length * sizeof(w_double *));
	w_double ** ml_weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++) {
		weighted_ref[n] = (w_double *) malloc(N_ref * sizeof(w_double));
		ml_weighted[n] = (w_double *) malloc(N_TOTAL_MAX * sizeof(w_double));
	}
	FILE * RAW_BPF_TIMES = fopen("raw_bpf_times.txt", "w");
	FILE * RAW_BPF_KS = fopen("raw_bpf_ks.txt", "w");
	FILE * RAW_BPF_MSE = fopen("raw_bpf_mse.txt", "w");
	FILE * RAW_TIMES = fopen("raw_times.txt", "w");
	FILE * RAW_KS = fopen("raw_ks.txt", "w");
	FILE * RAW_MSE = fopen("raw_mse.txt", "w");
	FILE * RAW_SRS = fopen("raw_srs.txt", "w");
	FILE * ALLOC_COUNTERS = fopen("alloc_counters.txt", "w");
	FILE * L2_ERR_DATA = fopen("l2_err_data.txt", "w");
	FILE * REF_STDS = fopen("ref_stds.txt", "w");
	FILE * FULL_HMM_DATA = fopen("full_hmm_data.txt", "w");
	FILE * FULL_REF_DATA = fopen("full_ref_data.txt", "w");
	output_parameters(N_trials, level0_meshes, nx, N1s, N_data, N_bpf);
	


	/* ----------------------------------------------------------------------------------------------------- */
	/*																										 																									 */
	/* Accuracy trials 																				 																							 */
	/* 																																																			 */
	/* ----------------------------------------------------------------------------------------------------- */	
	HMM * hmm = (HMM *) malloc(sizeof(HMM));
	for (int n_data = 0; n_data < N_data; n_data++) {

		/* Generate the HMM data and run the BPF on it */
		/* ------------------------------------------- */
		generate_hmm(rng, n_data, length, nx);
		read_hmm(hmm, n_data);
		generate_model(rng, hmm, N0s, N1s, weighted_ref, N_ref, N_trials, N_bpf, level0_meshes, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE);
		record_reference_data(hmm, weighted_ref, N_ref, FULL_HMM_DATA, FULL_REF_DATA, REF_STDS);


		/* Run the MLBPF for each nx0/particle allocation for the same time as the BPF and test its accuracy */
		/* ------------------------------------------------------------------------------------------------- */
		int N0, N1, N_tot;
		double ks, sr;
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {

			printf("--------------\n");
			printf("|  N1 = %d  |\n", N1s[n_alloc]);
			printf("--------------\n");

			for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {

				printf("nx0 = %d\n", level0_meshes[i_mesh]);
				printf("**********************************************************\n");

				nxs[0] = level0_meshes[i_mesh];
				N1 = N1s[n_alloc], N0 = N0s[i_mesh][n_alloc], N_tot = N0 + N1;
				sample_sizes[0] = N0, sample_sizes[1] = N1;
				alloc_counters[i_mesh] = N_ALLOCS;

				if (N0 == 0) {
					alloc_counters[i_mesh] = n_alloc;
					for (int n_trial = 0; n_trial < N_trials; n_trial++)
						raw_mse[i_mesh][n_alloc][n_data * N_trials + n_trial] = -1;
				}
				else {
					for (int n_trial = 0; n_trial < N_trials; n_trial++) {
						clock_t trial_timer = clock();
						ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted);
						// ml_bootstrap_particle_filter_debug(hmm, sample_sizes, nxs, rng, ml_weighted, L2_ERR_DATA);
						double elapsed = (double) (clock() - trial_timer) / (double) CLOCKS_PER_SEC;

						ks = 0.0, sr = 0.0;
						for (int n = 0; n < length; n++) {
							qsort(ml_weighted[n], N_tot, sizeof(w_double), weighted_double_cmp);
							ks += ks_statistic(N_ref, weighted_ref[n], N_tot, ml_weighted[n]) / (double) length;
						}
						raw_mse[i_mesh][n_alloc][n_data * N_trials + n_trial] = compute_mse(weighted_ref, ml_weighted, length, N_ref, N_tot);
						raw_ks[i_mesh][n_alloc][n_data * N_trials + n_trial] = ks;
						raw_times[i_mesh][n_alloc][n_data * N_trials + n_trial] = elapsed;
						raw_srs[i_mesh][n_alloc][n_data * N_trials + n_trial] = sr;
					}
				}
				printf("\n");
			}
		}
	}

	output_ml_data(hmm, N_trials, raw_times, raw_ks, raw_mse, raw_srs, level0_meshes, N1s, alloc_counters, ALLOC_COUNTERS, RAW_TIMES, RAW_KS, RAW_MSE, RAW_SRS, N_data);

	fclose(RAW_BPF_TIMES);
	fclose(RAW_BPF_KS);
	fclose(RAW_BPF_MSE);
	fclose(RAW_TIMES);
	fclose(RAW_KS);
	fclose(RAW_MSE);
	fclose(RAW_SRS);
	fclose(ALLOC_COUNTERS);
	fclose(L2_ERR_DATA);
	fclose(REF_STDS);
	fclose(FULL_HMM_DATA);
	fclose(FULL_REF_DATA);

	double total_elapsed = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	int hours = (int) floor(total_elapsed / 3600.0);
	int minutes = (int) floor((total_elapsed - hours * 3600) / 60.0);
	int seconds = (int) (total_elapsed - hours * 3600 - minutes * 60);
	printf("Total time for experiment = %d hours, %d minutes and %d seconds\n", hours, minutes, seconds);

	return 0;
}


/* --------------------------------------------------------------------------------------------------------------------
 *
 * Functions
 *
 * ----------------------------------------------------------------------------------------------------------------- */
void output_parameters(int N_trials, int * level0_meshes, int nx, int * N1s, int N_data, int N_bpf) {

	FILE * ML_PARAMETERS = fopen("ml_parameters.txt", "w");
	FILE * N1s_DATA = fopen("N1s_data.txt", "w");

	fprintf(ML_PARAMETERS, "%d %d %d %d %d \n", N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf);
	for (int m = 0; m < N_MESHES; m++)
		fprintf(ML_PARAMETERS, "%d ", level0_meshes[m]);
	fprintf(ML_PARAMETERS, "\n");
	fprintf(ML_PARAMETERS, "%d\n", nx);
	for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
		fprintf(N1s_DATA, "%d ", N1s[n_alloc]);
	fprintf(N1s_DATA, "\n");

	fclose(ML_PARAMETERS);
	fclose(N1s_DATA);

}


void record_reference_data(HMM * hmm, w_double ** weighted_ref, int N_ref, FILE * FULL_HMM_DATA, FILE * FULL_REF_DATA, FILE * REF_STDS) {

	int length = hmm->length;
	double std = 0.0, EX = 0.0, EX2 = 0.0;

	for (int n = 0; n < length; n++) {
		fprintf(FULL_HMM_DATA, "%e %e\n", hmm->signal[n], hmm->observations[n]);
			EX = 0.0, EX2 = 0.0;
		for (int i = 0; i < N_ref; i++) {
			fprintf(FULL_REF_DATA, "%e %e\n", weighted_ref[n][i].x, weighted_ref[n][i].w);
			EX += weighted_ref[n][i].x * weighted_ref[n][i].w;
			EX2 += weighted_ref[n][i].x * weighted_ref[n][i].x * weighted_ref[n][i].w;
		}
		std = sqrt(EX2 - EX * EX);
		fprintf(REF_STDS, "%e ", std);
	}

}


void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, int * alloc_counters, FILE * ALLOC_COUNTERS, FILE * RAW_TIMES, FILE * RAW_KS, FILE * RAW_MSE, FILE * RAW_SRS, int N_data) {

	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		fprintf(ALLOC_COUNTERS, "%d ", alloc_counters[i_mesh]);
		printf("Alloc counters for nx0 = %d = %d\n", level0_meshes[i_mesh], alloc_counters[i_mesh]);
	}
	fprintf(ALLOC_COUNTERS, "\n");

	/* Work horizontally from top left to bottom right, writing the result from each trial, new line when finished */
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {
			for (int n_trial = 0; n_trial < N_data * N_trials; n_trial++) {
				if (raw_times[i_mesh][n_alloc][n_trial] == 0)
					;
				else
					fprintf(RAW_TIMES, "%e ", raw_times[i_mesh][n_alloc][n_trial]);
				if (isnan(raw_mse[i_mesh][n_alloc][n_trial]))
					fprintf(RAW_MSE, "%d ", -2);
				else
					fprintf(RAW_MSE, "%e ", raw_mse[i_mesh][n_alloc][n_trial]);
				fprintf(RAW_KS, "%e ", raw_ks[i_mesh][n_alloc][n_trial]);
				fprintf(RAW_SRS, "%e ", raw_srs[i_mesh][n_alloc][n_trial]);
			}

			fprintf(RAW_KS, "\n");
			fprintf(RAW_MSE, "\n");
			fprintf(RAW_SRS, "\n");

		}
	}
}


