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
// beta_proportion_lpdf(FvFm | inv_logit(mu_rec), phi), truncated to
// (0, U_rec = 0.9) and renormalized -- see the "recovery component" note
// below. The damaged component is uniform severity plus instrument noise: a
// damaged needle's true FvFm is anywhere between dead (0) and a fixed cap,
// U_dmg = 0.65, and is read with Normal(0, sigma_dmg) error. The cap reflects
// the expectation that needles reading above about 0.65 can still recover,
// so the shoulder of readings between about 0.65 and 0.75 is left to the
// recovery component. The density has full support on the real line and is
// nearly flat between 0 and U_dmg, matching the shelf of low readings (FvFm
// below 0.5 is close to uniform, down to about -0.1). This lets the 43 rows
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
// --- Recovery component: capped at U_rec = 0.9 ---
// beta_proportion_lpdf has full support on (0, 1), but no observed FvFm goes
// above ~0.89 -- the right tail out to 1 is physically impossible here, and
// letting the Beta spread probability mass across it was distorting the
// fitted shape (phi had to account for a stretch of support the data never
// uses). This is implemented as a RESCALING, not a truncation: FvFm / U_rec
// is modeled directly as the Beta-proportion variate, i.e. U_rec is treated
// as the asymptote and inv_logit(mu_rec) is the mean fraction of that
// asymptote (not the mean of raw FvFm -- beta/alpha are on the
// logit(FvFm / U_rec) scale now). An earlier version instead truncated the
// untruncated-(0,1) Beta to (0, U_rec) and renormalized by its CDF there
// (beta_proportion_lcdf); that is more faithful to "cap the density's
// support" literally, but autodiffing the incomplete-beta CDF turned out to
// be ~220x more expensive per row than the plain lpdf, which dominated the
// whole model's gradient cost. The rescaling gives the same practical
// result -- the recovery component can never generate FvFm >= U_rec -- at
// the original speed, since it only adds a constant -log(U_rec) Jacobian
// term instead of a differentiated CDF call.
//
// --- Antecedent structure: Tmax only, for now ---
// Earlier versions carried all three weather series (Tmax, Tmin, VPD) as
// antecedent covariates, with various attempts at sharing their lag-weight
// vectors to control mixing (one Dirichlet simplex shared across all three;
// three independent simplices; a shared exponential-decay vector for
// Tmax/VPD plus a separate one for Tmin). All of these still showed mixing
// problems. Tmax is the variable of actual scientific interest here, and
// VPD is mathematically derived from temperature (it's a function of
// saturation vapor pressure, which depends on Tmax/Tmin), so VPD and Tmax
// were never really independent signals to begin with -- carrying both,
// however their lag weights were parameterized, was asking the antecedent
// structure to separate two things that aren't separable from this data.
// This version drops Tmin and VPD entirely and carries only Tmax, as a
// simpler, better-identified starting point. Reintroducing VPD/Tmin later
// (if at all) should probably be as a derived/residual quantity relative to
// Tmax rather than as independent antecedent series.
//
// --- Antecedent lag weights: back to a free Dirichlet simplex ---
// The exponential-decay parameterization (lag weight = softmax(-rho * lag),
// one scalar rate instead of a free simplex) was adopted specifically to
// cut the redundant freedom of multiple correlated, independently-weighted
// covariates. With only one weather covariate left, that redundancy is
// gone, so there's no longer a clear reason to prefer the more restrictive
// decay shape over the original simplex[Nlag] w_Tmax ~ dirichlet(...), which
// can fit any lag-weight shape the data support, not just a monotonic decay.
//
// Antecedent slope (alpha) now varies by the full tree_treatment x
// heat_treatment group (alpha[Ngroups], indexed by group_id) -- up from the
// heat_treatment-only split that was the first step back out of the
// fully-shared slope used while this version was being stabilized against
// mixing problems. group_id/Ngroups are the same interaction-of-factors
// construction already sitting as residual code in sam_analysis.qmd (built
// for the original FvFm_antecedent.stan group-specific alphas); Nheat/
// heat_id are no longer read by this file for the same reason Ngroups/
// group_id weren't before -- harmless to keep passing them in stan_data, no
// R-side changes required. The lag weighting (w_Tmax) and everything else
// in the antecedent structure remain shared across groups; only the
// sensitivity itself (alpha) is group-specific.
//
// --- No date effect ---
// Earlier versions included a date random effect (u_date_std) in mu_rec to
// absorb day-to-day variation not explained by weather. That's removed here
// so day-level variation is left entirely to the weather antecedent term
// (ant_Tmax) to explain -- the point of this version is to see how much of
// it it can actually account for, rather than quietly soaking up the rest
// in a per-day intercept.
//
// Stan has no NA, so missing FvFm entries are passed in as a presence/
// absence indicator; see the data block below for exactly what needs to be
// precomputed in R before calling this model.

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

  // beta_proportion_lpdf of y rescaled onto (0, 1) by a fixed asymptote
  // `cap`: y / cap is the Beta-proportion variate, with a constant
  // -log(cap) Jacobian term for the change of variables back to the FvFm
  // scale. Cheap (no CDF calls), unlike a true truncation-by-renormalization.
  real beta_proportion_scaled_lpdf(real y, real mu, real phi, real cap) {
    return beta_proportion_lpdf(y / cap | mu, phi) - log(cap);
  }

  // Matching posterior-predictive sampler: draw on (0, 1) and rescale back.
  real beta_proportion_scaled_rng(real mu, real phi, real cap) {
    return cap * beta_proportion_rng(mu, phi);
  }

}

data {

  int<lower=1> Ndays;                // number days for which the mean can be calculated (Ndays_total - Nlag)
  int<lower=1> Ntrees;               // Number of trees in data
  int<lower=1> Nlag;                 // number of lag days included in the antecedent window
  int<lower=1> N;                    // total number of rows in the data
  int<lower=1> K;                    // columns in the mean design matrix. Intercept must be column 1
  int<lower=1> Npanels;              // Number of unique panels/branches tracked
  int<lower=1> Ngroups;              // number of tree_treatment x heat_treatment groups (alpha varies by this)

  // Fv/Fm is modeled on the Fv/Fm scale (not log(Fv/Fm)): the recovery
  // component below is beta and the damaged component is uniform severity
  // plus normal noise. Working on the raw scale means Fv/Fm can legitimately
  // be <= 0.
  vector[N] FvFm;                                   // placeholder (e.g. -99) where missing
  array[N] int<lower=0, upper=1> pa_FvFm;           // presence/absence of Fv/Fm measurement
  array[N] int<lower=1, upper=Ndays> date_index;    // date index lookup

  // --- lagged environmental variables --- //
  // Tmin and VPD are no longer read by this file -- harmless to keep
  // passing them in stan_data (rstan ignores unmatched data-list entries),
  // no R-side changes required (see header note on dropping them).
  matrix[Ndays, Nlag] Tmax;

  // --- Experimental design  --- //
  matrix[N, K] X;                                   // design matrix for the recovery-component mean
  array[N] int<lower=1, upper=Ntrees> tree_id;      // index lookup for the tree random effect
  array[N] int<lower=1, upper=Npanels> panel_id;    // index lookup for the panel random effect
  array[N] int<lower=1, upper=Ngroups> group_id;    // index lookup for the group-specific antecedent slope

}

transformed data {

  real U_dmg = 0.45;                    // upper limit of true FvFm for a damaged needle; above this it can still recover
  real U_rec = 0.9;                     // upper limit of the recovery component's support; no observed FvFm exceeds ~0.89

}

parameters {

  // --- recovery component mean, with the antecedent (SAM) structure

  vector[K] beta;                       // regression coefficients: intercept, plus design covariates
  vector[Ngroups] alpha;                // antecedent coefficient (Tmax only, see header note), one per tree_treatment x heat_treatment group
  // real tau_Tmax;                        // hinge location on standardized ant_Tmax
  // real delta_Tmax;                      // hinge slope change above tau_Tmax
  real l_phi;                    // recovery-component dispersion
  vector[Ntrees] u_tree_std;            // non-scaled tree random effects
  vector[Npanels] u_panel_std;          // non-scaled panel effects
  real<lower=0> sigma_tree;             // sd for tree effects
  real<lower=0> sigma_panel;            // sd for panel effects

  simplex[Nlag] w_Tmax;                 // Tmax lag weights, freely estimated (see header note)

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

  // --- Construct cumulative effect ---
  vector[Ndays] ant_Tmax = Tmax * w_Tmax;

  vector[N] mu_rec;    // logit of recovery-component mean
  vector[N] eta;       // logit of P(damaged)

  for (n in 1:N) {
    mu_rec[n] = X[n, ] * beta
        + alpha[group_id[n]] * ant_Tmax[date_index[n]]
        // + delta_Tmax * fmax(0.0, ant_Tmax[date_index[n]] - tau_Tmax)
        + u_tree_std[tree_id[n]] * sigma_tree
        + u_panel_std[panel_id[n]] * sigma_panel;

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
  // Kept tight (sd = 0.5, tightened down from an earlier normal(0, 2)) now
  // that alpha is a vector[Ngroups] (expected Ngroups = 4: tree_treatment x
  // heat_treatment) -- same reasoning as when it was tightened for the
  // (now-removed) Tmax/Tmin/VPD vector: independent per-group slopes have
  // more freedom to trade off against each other than a single shared
  // scalar did, and this is the richer group-specific version that earlier
  // showed mixing problems on its own, so keep the tight prior as insurance.
  alpha ~ normal(0, 0.5);
  l_phi ~ normal(4, 5);
  u_tree_std ~ std_normal();
  u_panel_std ~ std_normal();
  sigma_tree ~ student_t(10, 0, 1);
  sigma_panel ~ student_t(10, 0, 1);

  w_Tmax ~ dirichlet(rep_vector(1, Nlag));

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
  //        + (1 - pi) * U_rec * BetaProportion(inv_logit(mu_rec), phi),
  // i.e. FvFm / U_rec is the recovery component's Beta-proportion variate
  // (U_rec treated as the asymptote; see header note). logit(pi) = eta.
  // Explicitly guard 0 < FvFm[n] < U_rec for the recovery term rather than
  // relying on beta_proportion_lpdf's built-in domain check on FvFm[n] /
  // U_rec -- FvFm[n] outside (0, U_rec) then gets all its mixture mass,
  // correctly, from the damaged component. (No observed FvFm reaches U_rec,
  // so this guard currently never removes rows from the recovery term in
  // practice; it's there for correctness if it ever does.)
  for (n in 1:N) {
    if (pa_FvFm[n]) {
      real lp_d = log_inv_logit(eta[n]) + uniform_noise_lpdf(FvFm[n] | U_dmg, sigma_dmg);
      real lp_r = (FvFm[n] > 0 && FvFm[n] < U_rec)
          ? log1m_inv_logit(eta[n]) + beta_proportion_scaled_lpdf(FvFm[n] | inv_logit(mu_rec[n]), exp(l_phi), U_rec)
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
    real lp_r = (FvFm[n] > 0 && FvFm[n] < U_rec)
        ? log1m_inv_logit(eta[n]) + beta_proportion_scaled_lpdf(FvFm[n] | inv_logit(mu_rec[n]), exp(l_phi), U_rec)
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
                    : beta_proportion_scaled_rng(inv_logit(mu_rec[n]), exp(l_phi), U_rec);
  }

}
