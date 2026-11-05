// Two-component mixture for Fv/Fm: irreparable damage vs. recovery, with
// damage probability driven by the current value of the recovery trajectory
// (a joint longitudinal/hazard structure) rather than its own antecedent
// submodel.
//
//
// Here the antecedent structure goes back on the RECOVERY MEAN ONLY (as in
// FvFm_antecedent.stan), where it is identified by the full dataset
// (~3870+ rows), not the damaged tail. The probability of tipping into the
// damaged mode is then a function of how far that already-estimated mean
// has fallen below the healthy baseline -- a shared-parameter structure
// like a joint longitudinal/survival model, where the hazard is driven by
// the current value of a fitted trajectory. This needs only an intercept
// and a slope (kappa0, kappa1) to be estimated from the sparse tail,
// instead of a full antecedent submodel.
//
// This file also fixes the known limitation flagged in
// FvFm_antecedent_mixture.stan's header: the mixture now lives on the
// FvFm scale rather than log(Fv). The recovery component is
// beta_proportion_lpdf(FvFm | inv_logit(mu_rec), phi). The damaged component
// is uniform severity plus instrument noise: a damaged needle's true FvFm is
// anywhere between dead (0) and a fixed cap, U_dmg = 0.65, and is read with
// Normal(0, sigma_dmg) error. The cap reflects the expectation that needles
// reading above about 0.65 can still recover, so the shoulder of readings
// between about 0.65 and 0.75 is left to the recovery component. The density
// has full support on the real line and is nearly flat between 0 and U_dmg,
// matching the shelf of low readings (FvFm below 0.5 is close to uniform,
// down to about -0.1). This lets the 43 rows
// with FvFm <= 0 -- previously dropped in R because log(Fv) is undefined --
// re-enter the likelihood. They fall out almost entirely into the damaged
// component, which is scientifically correct: they are the most severely
// damaged needles, not missing-at-random noise.
//
// On the FvFm scale the components are not separated by orders of magnitude,
// and an unconstrained normal damaged mean drifted up to ~0.8 (the healthy
// mode). Here no location is estimated: the damaged mean is fixed at about
// U_dmg / 2 = 0.325, which identifies the labels without an ordering parameter
// (no delta_dmg). sigma_dmg is the component's only parameter.
//
// mu_rec also includes a date random effect. Fv/Fm was measured on only 23 of
// the Ndays days, and much of the day-to-day variation is not explained by
// weather. u_date_std is indexed by date_index, so the unmeasured days simply
// draw from the prior and carry that uncertainty into their predictions.
//
// Stan has no NA, so missing FvFm entries are passed in as a presence/
// absence indicator; see the data block below for exactly what needs to be
// precomputed in R before calling this model.
//
// Antecedent slopes (alpha) are currently shared across tree_treatment x heat_treatment
// groups here, not group-specific -- a deliberate step back from a richer
// version of this model, since even on the well-identified recovery mean the
// group-specific alphas were still showing mixing problems. One shared
// Tmax/Tmin/VPD sensitivity is a simpler, more stable starting point; splitting
// it back out by group is a natural next step once this version is solid.
// The lag weights (w) are likewise shared across Tmax, Tmin and VPD, assuming
// one memory timescale: weather varies only by date, and separate weights for
// each covariate were mixing poorly.
// Ngroups/group_id are no longer read by this file -- harmless to keep
// passing them in stan_data (rstan ignores unmatched data-list entries), no
// R-side changes required.

functions {

  // Density of y = s + e, with s ~ Uniform(0, u) and e ~ Normal(0, sigma):
  // (Phi(y / sigma) - Phi((y - u) / sigma)) / u
  // Readings above u / 2 use the equivalent upper-tail difference, since both
  // lower-tail log CDFs round to 0 when y is far above u and sigma is small.
  real uniform_noise_lpdf(real y, real u, real sigma) {
    real z_hi = y / sigma;
    real z_lo = (y - u) / sigma;
    real log_diff = (y > u / 2)
        ? log_diff_exp(std_normal_lcdf(-z_lo | ), std_normal_lcdf(-z_hi | ))
        : log_diff_exp(std_normal_lcdf(z_hi | ), std_normal_lcdf(z_lo | ));
    return log_diff - log(u);
  }

}

data {

  int<lower=1> Ndays;                // number days for which the mean can be calculated (Ndays_total - Nlag)
  int<lower=1> Ntrees;               // Number of trees in data
  int<lower=1> Nlag;                 // number of lag days included in the antecedent window
  int<lower=1> N;                    // total number of rows in the data
  int<lower=1> K;                    // columns in the mean design matrix. Intercept must be column 1
  int<lower=1> Npanels;              // Number of unique panels/branches tracked

  // Fv/Fm is modeled on the Fv/Fm scale (not log(Fv/Fm)): the recovery
  // component below is beta and the damaged component is uniform severity
  // plus normal noise. Working on the raw scale means Fv/Fm can legitimately
  // be <= 0.
  vector[N] FvFm;                                   // placeholder (e.g. -99) where missing
  array[N] int<lower=0, upper=1> pa_FvFm;           // presence/absence of Fv/Fm measurement
  array[N] int<lower=1, upper=Ndays> date_index;    // date index lookup

  // --- lagged environmental variables --- //
  matrix[Ndays, Nlag] Tmax;
  matrix[Ndays, Nlag] Tmin;
  matrix[Ndays, Nlag] VPD;

  // --- Experimental design  --- //
  matrix[N, K] X;                                   // design matrix for the recovery-component mean
  array[N] int<lower=1, upper=Ntrees> tree_id;      // index lookup for the tree random effect
  array[N] int<lower=1, upper=Npanels> panel_id;    // index lookup for the panel random effect

}

transformed data {

  real U_dmg = 0.65;                    // upper limit of true FvFm for a damaged needle; above this it can still recover

}

parameters {

  // --- recovery component mean, with the antecedent (SAM) structure
  
  vector[K] beta;                       // regression coefficients: intercept, plus design covariates
  vector[3] alpha;                      // antecedent coefficients (Tmax, Tmin, VPD), shared across all tree_treatment x heat_treatment groups
  // real tau_Tmax;                        // hinge location on standardized ant_Tmax
  // real delta_Tmax;                      // hinge slope change above tau_Tmax
  real l_phi;                    // recovery-component dispersion
  vector[Ntrees] u_tree_std;            // non-scaled tree random effects
  vector[Npanels] u_panel_std;          // non-scaled panel effects
  vector[Ndays] u_date_std;             // non-scaled date effects
  real<lower=0> sigma_tree;             // sd for tree effects
  real<lower=0> sigma_panel;            // sd for panel effects
  real<lower=0> sigma_date;             // sd for date effects

  simplex[Nlag] w;                      // lag weights, shared across Tmax, Tmin and VPD

  // --- damaged component --- //
  real<lower=0> sigma_dmg;              // measurement-noise SD of damaged readings, on the FvFm scale

  // --- tipping/hazard model: P(damaged) as a function of how far mu_rec
  // has fallen below the healthy baseline --- //
  real kappa0;                          // logit intercept at zero depression
  real<lower=0> kappa1;                 // slope on depression below baseline; >0 since more depressed must mean more likely damaged
  // vector[Npanels] u_panel_p_std;        // non-scaled panel death-propensity effects, beyond what depression already explains
  // real<lower=0> sigma_panel_p;          // sd for panel death-propensity effects

}

transformed parameters {

  // --- Construct cumulative effects ---
  vector[Ndays] ant_Tmax = Tmax * w;
  vector[Ndays] ant_Tmin = Tmin * w;
  vector[Ndays] ant_VPD = VPD * w;

  vector[N] mu_rec;    // logit of recovery-component mean
  vector[N] eta;       // logit of P(damaged)

  for (n in 1:N) {
    mu_rec[n] = X[n, ] * beta
        + alpha[1] * ant_Tmax[date_index[n]]
        // + delta_Tmax * fmax(0.0, ant_Tmax[date_index[n]] - tau_Tmax)
        + alpha[2] * ant_Tmin[date_index[n]]
        + alpha[3] * ant_VPD[date_index[n]]
        + u_tree_std[tree_id[n]] * sigma_tree
        + u_panel_std[panel_id[n]] * sigma_panel
        + u_date_std[date_index[n]] * sigma_date;

    // Depression below the healthy baseline, on the log(Fv/Fm) scale (strip
    // the l_Fm offset back out so this is comparable to beta[1], the
    // group-level healthy intercept). fmax(.,0) means nothing happens until
    // the fitted trajectory actually sinks below baseline -- the tipping
    // point -- after which the odds of damage rise with depression. This
    // responds to depression from ANY source (antecedent weather, needle
    // age, panel idiosyncrasies, ...): a branch that is already running low
    // for whatever reason is closer to collapse.
    real d_n = beta[1] - mu_rec[n];
    eta[n] = kappa0 + kappa1 * fmax(0.0, d_n);
        //+ u_panel_p_std[panel_id[n]] * sigma_panel_p;
  }

}

model {

  // --- priors: recovery-component mean / antecedent structure, copied
  // verbatim from FvFm_antecedent.stan -- these are back on the full
  // dataset and don't need the tail-insurance tightening used when this
  // structure lived in the damage logit --- //
  beta ~ normal(0, 2);
  alpha ~ normal(0, 2);
  l_phi ~ normal(4, 5);
  u_tree_std ~ std_normal();
  u_panel_std ~ std_normal();
  u_date_std ~ std_normal();
  sigma_tree ~ student_t(10, 0, 1);
  sigma_panel ~ student_t(10, 0, 1);
  sigma_date ~ student_t(10, 0, 1);

  w ~ dirichlet(rep_vector(2, Nlag));

  // delta_Tmax ~ normal(0, 1);
  // tau_Tmax ~ normal(0, 10);

  // --- priors: damaged component ---
  // sigma_dmg is now instrument noise only, not the spread of damage severity:
  // the lowest readings reach about -0.1, and a pooled fit gives ~0.055.
  sigma_dmg ~ normal(0, 0.1);

  // --- priors: tipping/hazard model ---
  // (95 + 43 restored rows) / (3870 + 43 total) =~ 3.5% base rate =>
  // logit(0.035) =~ -3.3 at zero depression.
  kappa0 ~ normal(-3, 1.5);
  // Healthy/damaged separation is ~1.6 log-units; wanting P(damaged)
  // noticeably elevated (~0.3) by d =~ 1.6 implies kappa1 =~ 1.5, well
  // inside this prior.
  kappa1 ~ normal(0, 2);
  // u_panel_p_std ~ std_normal();
  // sigma_panel_p ~ student_t(10, 0, 0.5);

  // --- likelihood ---
  // FvFm ~ pi * (Uniform(0, U_dmg) + Normal(0, sigma_dmg))
  //        + (1 - pi) * BetaProportion(inv_logit(mu_rec), phi),
  // with logit(pi) = eta. Explicitly guard FvFm[n] > 0 for the recovery term
  // rather than relying on beta_proportion_lpdf's built-in domain check --
  // FvFm[n] <= 0 rows then get all their mixture mass, correctly, from the
  // damaged component.
  for (n in 1:N) {
    if (pa_FvFm[n]) {
      real lp_d = log_inv_logit(eta[n]) + uniform_noise_lpdf(FvFm[n] | U_dmg, sigma_dmg);
      real lp_r = (FvFm[n] > 0)
          ? log1m_inv_logit(eta[n]) + beta_proportion_lpdf(FvFm[n] | inv_logit(mu_rec[n]), exp(l_phi))
          : negative_infinity();
      target += log_sum_exp(lp_d, lp_r);
    }
  }

}

generated quantities {
  vector[N] pi_dmg;        // covariate-driven P(damaged), i.e. inv_logit(eta)
  vector[N] p_dmg;         // posterior responsibility P(damaged | Fv) -- the classifier
  vector[N] log_lik;       // per-row mixture log-density, 0 where unobserved; for loo
  vector[N] FvFm_rep;      // posterior-predictive replicate for raw Fv

  for (n in 1:N) {
    real lp_d = log_inv_logit(eta[n]) + uniform_noise_lpdf(FvFm[n] | U_dmg, sigma_dmg);
    real lp_r = (FvFm[n] > 0)
        ? log1m_inv_logit(eta[n]) + beta_proportion_lpdf(FvFm[n] | inv_logit(mu_rec[n]), exp(l_phi))
        : negative_infinity();

    pi_dmg[n] = inv_logit(eta[n]);

    // pi_dmg uses the fitted trajectory only; p_dmg additionally conditions
    // on the observed value. Both are wanted -- the gap between them is
    // where the depression-based hazard fails to anticipate damage that
    // actually happened.
    log_lik[n] = pa_FvFm[n] ? log_sum_exp(lp_d, lp_r) : 0;
    p_dmg[n] = pa_FvFm[n] ? exp(lp_d - log_sum_exp(lp_d, lp_r)) : pi_dmg[n];

    FvFm_rep[n] = bernoulli_rng(pi_dmg[n])
                    ? uniform_rng(0, U_dmg) + normal_rng(0, sigma_dmg)
                    : beta_proportion_rng(inv_logit(mu_rec[n]), exp(l_phi));
  }

}
