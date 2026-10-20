// FvFm model driven by an antecedent temperature index.
//
// Variant of FvFm_antecedent.stan: residual diagnostics showed strongly
// left-skewed residuals that aren't explained by tree, date, or any of the
// antecedent weather covariates, so the likelihood here is a skew-normal
// instead of a normal -- same mean structure (mu), with one added shape
// parameter (skew_shape) to let the error distribution itself be
// asymmetric. skew_shape < 0 => left skew (occasional large negative
// excursions), matching the residual pattern observed.
//
// Stan has no NA, so missing FvFm entries are passed in as index pairs
// indicating missing days; see the data block below for exactly what
// needs to be precomputed in R before calling this model.

data {

  int<lower=1> Ndays;                // number days for which the mean can be calculated (Ndays_total - Nlag)
  int<lower=1> Ntrees;               // Number of trees in data
  int<lower=1> Nlag;                 // number of lag days included in the antecedent window
  int<lower=Ndays*Ntrees> N;         // total number of rows in the data, should be at least as many as days * trees
  int<lower=1> K;                    // columns in design matrix. Should at least be one for the intercept
  int<lower=1> Npanels;              // Number of unique panels/branches tracked

  // Fv/Fm is modeled as log(Fv) with log(Fm) as an offset: since
  // log(Fv) = log(Fm) + log(Fv/Fm), giving log(Fv) mean log(Fm) + mu is
  // equivalent to modeling the log ratio directly with mean mu.
  vector[N] l_Fv;                                   // variable fluorescence; placeholder (e.g. 1) where missing
  vector[N] l_Fm;                                   // log(maximal fluorescence), used as an offset; placeholder (e.g. 0) where missing
  array[N] int<lower=0, upper=1> pa_FvFm;           // presence/absence of a paired Fv/Fm measurement
  array[N] int<lower=1, upper=Ndays> date_index;    // date index lookup

  // --- lagged environmental variables --- //
  matrix[Ndays, Nlag] Tmax;
  matrix[Ndays, Nlag] Tmin;
  matrix[Ndays, Nlag] VPD;
  // vector[Ndays + Nlag] pre_dawn_psi;                 // missingness filled with placeholder, like 0
  // int<lower=0> Nmiss_psi;
  // array[Nmiss_psi] int<lower=1, upper=Ndays + Nlag> miss_psi;  // missingness indexes

  // --- Experimental design  --- //
  matrix[N, K] X;                                       // The 'fixed effect' design matrix
  array[N] int<lower=1, upper=Ntrees + 1> tree_id;      // index lookup for the tree random effect (plus a level for NA)
  array[N] int<lower=1, upper=Npanels + 1> panel_id;    // index lookup for the panel random effect

}


parameters {

  vector[K] beta;                       // regression coefficients: intercept, plus design covariates
  vector[3] alpha;                      // regression coefs for the antecedent variables
  real<lower=0> sigma;                  // residual scale
  vector[Ntrees] u_tree_std;            // non-scaled tree random effects
  vector[Npanels] u_panel_std;          // non-scaled panel effects
  real<lower=0> sigma_tree;             // sd for tree effects
  real<lower=0> sigma_panel;            // sd for panel effects

  // unnormalized Dirichlet weights (gamma trick) for each lagged covariate //
  vector<lower=0>[Nlag] gamma_Tmax;
  vector<lower=0>[Nlag] gamma_Tmin;
  vector<lower=0>[Nlag] gamma_VPD;
  // vector<lower=0>[Nlag] gamma_psi;

  // threshold effects //
  real tau_Tmax;
  real delta_Tmax;

  // skew-normal shape parameter: negative => left-skewed residuals //
  real skew_shape;

  // real mu_psi;                           // Psi missing-data model mean
  // real<lower=0> sigma_psi;               // Psi missing-data model SD
  // real<lower=0, upper=1> phi;            // AR(1) missing data model
  //
  // vector[Nmiss_psi] psi_impute;          // imputed values for missing Event entries
}

transformed parameters {
  // vector[Ndays + Nlag] psi;          // observed + imputed Psi (Nlag days of pre-season padding + Ndays of study)
  // vector[N] mu;                      // mean log(Fv/Fm)



  // // --- splice imputed values into the Psi series --- //
  // psi = pre_dawn_psi;
  // for (i in 1:Nmiss_psi) {
  //   psi[miss_psi[i]] = psi_impute[i];
  // }

  // Constructing normalized weights //
  vector[Nlag] w_Tmax = gamma_Tmax ./ sum(gamma_Tmax);
  vector[Nlag] w_Tmin = gamma_Tmin ./ sum(gamma_Tmin);
  vector[Nlag] w_VPD = gamma_VPD ./ sum(gamma_VPD);
  // vector[Nlag] w_psi = gamma_psi ./ sum(gamma_psi);

  // // construct antPsi //
  // vector[Ndays] antPsi;
  // for(d in 1:Ndays) {
  //   antPsi[d] = w_psi[1] * psi[d + Nlag];
  //   for(l in 2:Nlag){
  //     antPsi[d] += w_psi[l] * psi[d + Nlag - l + 1];
  //   }
  // }


  // --- Construct cumulative effects ---
  vector[Ndays] ant_Tmax = Tmax * w_Tmax;
  vector[Ndays] ant_Tmin = Tmin * w_Tmin;
  vector[Ndays] ant_VPD = VPD * w_VPD;

  // --- mean log(Fv/Fm) for every row (needed for both the likelihood,
  // restricted to observed rows below, and for generated quantities) ---
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = X[n, ] * beta +
        alpha[1] * (ant_Tmax[date_index[n]] - tau_Tmax) +
        delta_Tmax * fmax(0.0, ant_Tmax[date_index[n]] - tau_Tmax) +
        alpha[2] * ant_Tmin[date_index[n]] +
        alpha[3] * ant_VPD[date_index[n]] +
        // delta_vpd * fmin(0.0, tau_vpd - ant_VPD[date_index[n]]) +
        u_tree_std[tree_id[n]] * sigma_tree +
        u_panel_std[panel_id[n]] * sigma_panel +
        l_Fm[n];
  }

}

model {

  // --- priors ---
  beta ~ normal(0, 5);
  alpha ~ normal(0, 5);
  sigma ~ student_t(10, 0, 1);
  u_tree_std ~ std_normal();
  u_panel_std ~ std_normal();
  sigma_tree ~ student_t(10, 0, 1);
  sigma_panel ~ student_t(10, 0, 1);

  gamma_Tmax ~ gamma(1, 1);
  gamma_Tmin ~ gamma(1, 1);
  gamma_VPD ~ gamma(1, 1);
  // gamma_psi ~ gamma(1, 1);

  // mu_psi ~ normal(0, 5);
  // sigma_psi ~ student_t(10, 0, 1);

  delta_Tmax ~ normal(0, 1);
  tau_Tmax ~ normal(0, 10);
  // delta_vpd ~ normal(0, 1);
  // tau_vpd ~ normal(0, 10);

  skew_shape ~ normal(0, 5);

  // // --- psi missing-data model (AR(1) over the full padded + observed series) ---
  // psi[1] ~ normal(mu_psi, sqrt(pow(sigma_psi, 2) / (1 - pow(phi, 2))));
  // for(d in 2:(Ndays + Nlag)){
  //   psi[d] ~ normal(mu_psi + phi * (psi[d - 1] - mu_psi), sigma_psi);
  // }

  // --- likelihood ---
  // log(Fv) ~ SkewNormal(log(Fm) + mu, sigma, skew_shape), i.e.
  // log(Fv/Fm) ~ SkewNormal(mu, sigma, skew_shape)
  for(n in 1:N) {
    if(pa_FvFm[n]) {
      target += skew_normal_lpdf(l_Fv[n] | mu[n], sigma, skew_shape);
    }
  }

}

generated quantities {
  vector[N] l_Fv_rep;               // posterior-predictive replicate for log(Fv)

  for (n in 1:N) {
    l_Fv_rep[n] = skew_normal_rng(mu[n], sigma, skew_shape);
  }

}
