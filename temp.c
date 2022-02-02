

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
	double * thetas_res = (double *) malloc(N * sizeof(double));
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
	for (int i = 0; i < N; i++)
		thetas[i] = gsl_ran_gaussian(rng, sig_sd) + theta0;

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
		double w_sum = 0.0;
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = thetas[i];
			weighted[n][i].w = weights[i];
			x_hat += thetas[i] * weights[i];
			w_sum += weights[i];
		}
		assert(fabs(w_sum - 1.0) < 1e-06);
		fprintf(X_HATS, "%e ", x_hat);
		printf("n = %d, x_hat = %lf\n", n, x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			thetas_res[i] = thetas[ind[i]];
		mutate(N, thetas, thetas_res, sig_sd, rng, theta_min, theta_max);

	}

	fclose(CURVE_DATA);
	fclose(X_HATS);

	free(ind);
	free(thetas);
	free(thetas_res);
	free(weights);
	free(h);
	free(Z);
	free(Z_x);
	free(xs);

}







void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted) {
	

	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx0 = nxs[0], nx1 = nxs[1];
	int obs_pos0 = nx0 + 1;
	int obs_pos1 = nx1 + 1;
	int lag = hmm->lag, start_point = 0, counter0, counter1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 2, M_poly = poly_degree + 1;
	double sign_rat = 0.0, coarse_scaler = 0.5;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double v = hmm->v, mu = hmm->mu;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double space_length = space_right - space_left;
	double T_stop = hmm->T_stop;
	double dx0 = space_length / (double) (nx0 - 1);
	double dx1 = space_length / (double) (nx1 - 1);
	double dt0 = CFL_CONST * dx0 / v;
	double dt1 = CFL_CONST * dx1 / v;
	double r0 = mu * dt0 / (2.0 * dx0 * dx0), r1 = mu * dt1 / (2.0 * dx1 * dx1);
	double obs, normaliser, abs_normaliser, g0, g1, ml_xhat;
	size_t size0 = (nx0 + 4) * sizeof(double), size1 = (nx1 + 4) * sizeof(double);
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	double * s = (double *) malloc(N_tot * sizeof(double));
	double * s_res = (double *) malloc(N_tot * sizeof(double));
	double * sign_ratios = (double *) calloc(length, sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * h1s = (double *) malloc(N1 * sizeof(double));
	double * h0s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc((poly_degree + 1) * sizeof(double));
	double * theta = (double *) calloc(2, sizeof(double));
	double * ml_xhats = (double *) malloc(length * sizeof(double));
	double * rho0 = (double *) calloc(nx0 + 4, sizeof(double));
	double * rho1 = (double *) calloc(nx1 + 4, sizeof(double));
	double * ics0 = (double *) calloc(nx0 + 4, sizeof(double));
	double * ics1 = (double *) calloc(nx1 + 4, sizeof(double));
	double * slopes0 =  (double *) malloc((nx0 + 1) * sizeof(double));
	double * Q_star0 = (double *) malloc(nx0 * sizeof(double));
	double * slopes1 =  (double *) malloc((nx1 + 1) * sizeof(double));
	double * Q_star1 = (double *) malloc(nx1 * sizeof(double));
	double * xs0 = construct_space_mesh((nx0 + 4) * sizeof(double), space_left, dx0, nx0);
	double * xs1 = construct_space_mesh((nx1 + 4) * sizeof(double), space_left, dx1, nx1);
	double ** X = (double **) malloc(N_tot * sizeof(double *));
	for (int i = 0; i < N_tot; i++)
		X[i] = (double *) malloc((lag + 1) * sizeof(double));


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Level 0 Crank-Nicolson matrices */
	/* ------------------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower0 = gsl_vector_alloc(nx0 + 1);
	gsl_vector * main0 = gsl_vector_alloc(nx0 + 2);
	gsl_vector * upper0 = gsl_vector_alloc(nx0 + 1);
	construct_forward(r0, nx0 + 2, lower0, main0, upper0);

	/* Construct the present time matrix */
	gsl_matrix * B0 = gsl_matrix_calloc(nx0 + 2, nx0 + 2);
	gsl_vector * u0 = gsl_vector_calloc(nx0 + 2);
	gsl_vector * Bu0 = gsl_vector_alloc(nx0 + 2);
	construct_present(r0, nx0 + 2, B0);


	/* Level 1 Crank-Nicolson matrices */
	/* ------------------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower1 = gsl_vector_alloc(nx1 + 1);
	gsl_vector * main1 = gsl_vector_alloc(nx1 + 2);
	gsl_vector * upper1 = gsl_vector_alloc(nx1 + 1);
	construct_forward(r1, nx1 + 2, lower1, main1, upper1);
	
	/* Construct the present time matrix */
	gsl_matrix * B1 = gsl_matrix_calloc(nx1 + 2, nx1 + 2);
	gsl_vector * u1 = gsl_vector_calloc(nx1 + 2);
	gsl_vector * Bu1 = gsl_vector_alloc(nx1 + 2);
	construct_present(r1, nx1 + 2, B1);


	/* Initial conditions */
	/* ------------------ */
	double s0 = hmm->signal[0], l2_err = 0.0;
	generate_ics(ics0, dx0, nx0, space_left);
	generate_ics(ics1, dx1, nx1, space_left);
	memcpy(rho0, ics0, size0);
	memcpy(rho1, ics1, size1);
	for (int i = 0; i < N_tot; i++) {
		s[i] = gsl_ran_gaussian(rng, sig_sd) + s0;
		X[i][0] = s[i];
		res_signs[i] = 1;	
	}
	gsl_interp * ics_interp = gsl_interp_alloc(gsl_interp_linear, nx1 + 4);
	gsl_interp_accel * acc = gsl_interp_accel_alloc();

	/* Output files */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	// FILE * ML_XHATS = fopen("ml_xhats.txt", "w");



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		obs = hmm->observations[n];


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 solutions																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		normaliser = 0.0, abs_normaliser = 0.0;
		for (int i = N0; i < N_tot; i++) {


			/* Fine solution */
			/* ------------- */
			/* Fine solve with respect to the historical particles */
			minmod_convection_diffusion_solver(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, s[i], r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1);
			h1s[i - N0] = rho1[obs_pos1];
			

			/* Coarse solution */
			/* --------------- */
			/* Coarse solve with respect to the historical particles */
			minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
			h0s[i - N0] = rho0[obs_pos0];
			
			/* Reset the initial conditions to the current time level for the next particle weighting */
			corrections[i - N0] = h1s[i - N0] - h0s[i - N0];
			memcpy(rho1, ics1, size1);
			memcpy(rho0, ics0, size0);			


		}

		/* Output the level 1 particles and their corresponding correction value */
		if (N1 > 0)
			regression_fit(s, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 weight generation																				 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			g1 = gsl_ran_gaussian_pdf(h1s[i - N0] - obs, obs_sd);
			g0 = gsl_ran_gaussian_pdf(h0s[i - N0] + poly_eval(s[i], poly_weights, poly_degree) - obs, obs_sd);

			weights[i] = (g1 - g0) * (double) res_signs[i] / (double) N1;
			absolute_weights[i] = fabs(weights[i]);
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 0 weight generation																				 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {

			for (int i = 0; i < N0; i++) {


				/* Coarse solution */
				/* --------------- */
				/* Coarse solve with respect to the historical particles */
				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
				g0 = gsl_ran_gaussian_pdf(rho0[obs_pos0] + poly_eval(s[i], poly_weights, poly_degree) - obs, obs_sd);


				/* Weight computation */
				/* ------------------ */
				weights[i] = g0 * (double) res_signs[i] / (double) N0;
				absolute_weights[i] = fabs(weights[i]);
				normaliser += weights[i];
				abs_normaliser += absolute_weights[i];

				/* Reset the initial conditions to the current time level for the next particle weighting */
				memcpy(rho0, ics0, size0);

			}

		}

		else {

			for (int i = 0; i < N0; i++) {


				/* Coarse solution */
				/* --------------- */
				/* Coarse solve with respect to the historical particles */
				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
				g0 = gsl_ran_gaussian_pdf(rho0[obs_pos0] - obs, obs_sd);


				/* Weight computation */
				/* ------------------ */
				weights[i] = g0 * (double) res_signs[i] / (double) N0;
				absolute_weights[i] = fabs(weights[i]);
				normaliser += weights[i];
				abs_normaliser += absolute_weights[i];

				/* Reset the initial conditions to the current time level for the next particle weighting */
				memcpy(rho0, ics0, size0);

			}

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		ml_xhat = 0.0, sign_rat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			absolute_weights[i] /= abs_normaliser;
			weights[i] /= normaliser;
			signs[i] = weights[i] < 0 ? -1 : 1;
			ml_weighted[n][i].x = s[i];
			ml_weighted[n][i].w = weights[i];
			ml_xhat += s[i] * weights[i];
		}
		ml_xhats[n] = ml_xhat;
		// fprintf(ML_XHATS, "%e ", ml_xhat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		for (int i = 0; i < N_tot; i++) {
			s_res[i] = s[ind[i]];
			res_signs[i] = signs[ind[i]];
		}
		mutate(N_tot, s, s_res, sig_sd, rng, n);
		evolve_initial_conditions(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, ml_xhats[n - lag], r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1, rho0, nx0, size0, size1, ics0, ics1, ics_interp, acc, xs0, xs1);

	}

	fclose(CURVE_DATA);
	// fclose(ML_XHATS);

	free(signs);
	free(res_signs);
	free(ind);
	free(s);
	free(s_res);
	free(sign_ratios);
	free(weights);
	free(absolute_weights);
	free(h1s);
	free(h0s);
	free(corrections);
	free(poly_weights);
	free(theta);
	free(ml_xhats);
	free(rho0);
	free(rho1);
	free(ics0);
	free(ics1);
	free(slopes0);
	free(Q_star0);
	free(slopes1);
	free(Q_star1);
	free(xs0);
	free(xs1);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	free(X);

	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);

	gsl_interp_free(ics_interp);
	gsl_interp_accel_free(acc);

	gsl_vector_free(lower0);
	gsl_vector_free(main0);
	gsl_vector_free(upper0);
	gsl_matrix_free(B0);
	gsl_vector_free(u0);
	gsl_vector_free(Bu0);
	gsl_vector_free(lower1);
	gsl_vector_free(main1);
	gsl_vector_free(upper1);	
	gsl_matrix_free(B1);
	gsl_vector_free(u1);
	gsl_vector_free(Bu1);

}