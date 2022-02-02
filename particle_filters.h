	typedef struct {
	double * signal;
	double * observations;
	int length;
	int nx;
	double sig_sd;
	double obs_sd;
	double space_left;
	double space_right;
	double k;
	double h0;
	double q0;
} HMM;


typedef struct weighted_double {
	double x;
	double w;
} w_double;

int weighted_double_cmp(const void * a, const void * b);

void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted);

void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted);

void ml_bootstrap_particle_filter_debug(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, FILE * L2_ERR_DATA);