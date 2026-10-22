// Two-component mixture for Fv/Fm: irreparable damage vs. recovery.
//
// Variant of FvFm_antecedent.stan. The earlier models treat the long left tail
// of log(Fv/Fm) as an error-distribution problem -- FvFm_antecedent.stan ignores
// it, FvFm_antecedent_skewnormal.stan absorbs it into a skew_shape parameter.
// It may instead be due to a separate process/threshold. log(Fv/Fm) is bimodal:
//
//   healthy mode         n = 3870   mean -0.232, sd 0.063
//   damaged tail (<-1)   n =   95   median -1.83, min -6.86
//   Fv <= 0 (log undef)  n =   43   dropped in R, see "Known limitation"
//
// Needles in the damaged group have Fm collapsed to ~24 (vs ~200 healthy) and Fv
// scattered in (-1.9, 1.79) -- instrument noise around a true zero. The group
// skews heated (47) over control (15) and clusters at measurements 0/1/2, right
// after the heat treatment.
//
// So the likelihood here is a mixture of two Normals on l_Fv:
//
//   component D ("damaged"):  Fv/Fm has collapsed toward zero, irreversibly.
//   component R ("recovery"): the mean structure from FvFm_antecedent.stan.
//
// The key change relative to FvFm_antecedent.stan is WHERE the antecedent (SAM)
// terms act. They no longer shift the mean; they form a logit model for the
// probability that an observation belongs to component D. Cumulative heat load
// predicts *whether the needle died*, not *how depressed* its Fv/Fm is. The
// tau_Tmax/delta_Tmax hinge is MOVED into that logit, where it parameterizes a
// thermal threshold for irreversible damage. A panel-level random effect in the
// logit gives each branch a persistent death propensity across its dates.
//
// As in FvFm_antecedent.stan, the antecedent slopes (alpha_p) are
// group-specific: each tree_treatment x heat_treatment combination (Ngroups =
// 4) gets its own Tmax/Tmin/VPD sensitivity in the damage logit, i.e. heat and
// irrigation can change how strongly antecedent weather predicts the ODDS OF
// DAMAGE, not just the baseline odds (which Xp*beta_p already captures via its
// own tree_treatment x heat_treatment interaction). The lag-weight profiles
// and the Tmax hinge (tau_Tmax/delta_Tmax) stay shared across groups.
//
// Label switching is impossible here: delta_dmg > 0 forces component D's level
// strictly below the healthy baseline beta[1], so the components can't swap.
//
//
// Known limitation: the 43 rows with Fv <= 0 are still dropped in R by
// filter(!is.nan(l_Fv)). Those rows are NOT missing at random -- they are the
// most severely damaged needles, excluded precisely because the damage is total.
// So the fitted damage frequency is biased low and tau_Tmax will read as higher
// (more heat required) than reality. Fixing it properly means either a
// left-censored contribution at a detection floor, or moving the mixture onto
// the raw Fv scale, where a Normal(0, sigma_dmg) damaged component has support
// at and below zero and those 43 rows can be retained.
//
// Output size: mu_rec, eta and the four generated quantities are each length N
// (~4080), so this model saves ~6 N-vectors per draw against 2 in
// FvFm_antecedent.stan. If memory is tight under rstan, drop pi_dmg (it is just
// inv_logit(eta), recoverable in R as plogis(eta)) and l_Fv_rep first.
//
// Stan has no NA, so missing FvFm entries are passed in as a presence/absence
// indicator; see the data block below for exactly what needs to be precomputed
// in R before calling this model.

data {

  int<lower=1> Ndays;                // number days for which the mean can be calculated (Ndays_total - Nlag)
  int<lower=1> Ntrees;               // Number of trees in data
  int<lower=1> Nlag;                 // number of lag days included in the antecedent window
  int<lower=1> N;                    // total number of rows in the data
  int<lower=1> K;                    // columns in the mean design matrix. Intercept must be column 1
  // int<lower=1> Kp;                   // columns in the damage-probability design matrix. Intercept must be column 1
  int<lower=1> Npanels;              // Number of unique panels/branches tracked
  int<lower=1> Ngroups;              // number of tree_treatment x heat_treatment groups, each with its own damage-logit antecedent slopes

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

  // --- Experimental design  --- //
  matrix[N, K] X;                                   // design matrix for the recovery-component mean
  // matrix[N, Kp] Xp;                                 // design matrix for the damage-probability logit
  array[N] int<lower=1, upper=Ntrees> tree_id;      // index lookup for the tree random effect
  array[N] int<lower=1, upper=Npanels> panel_id;    // index lookup for the panel random effect
  array[N] int<lower=1, upper=Ngroups> group_id;    // index lookup for the row's tree_treatment x heat_treatment group

}


parameters {

  // --- recovery component mean --- //
  vector[K] beta;                       // regression coefficients: intercept, plus design covariates
  real<lower=0> sigma;                  // recovery-component residual SD
  vector[Ntrees] u_tree_std;            // non-scaled tree random effects
  vector[Npanels] u_panel_std;          // non-scaled panel effects
  real<lower=0> sigma_tree;             // sd for tree effects
  real<lower=0> sigma_panel;            // sd for panel effects

  // --- damaged component --- //
  real<lower=0> delta_dmg;              // drop below the healthy baseline; > 0 identifies the labels
  real<lower=0> sigma_dmg;              // damaged-component SD (expect ~1.3, much wider than sigma)

  // --- damage-probability (logit) model --- //
  real beta_p;                          // intercept on the logit scale

  // alpha_p is now partially pooled across groups rather than fit
  // independently (normal(0,1) per group): the damaged tail is tiny and
  // split very unevenly across the 4 tree_treatment x heat_treatment groups
  // (as few as ~2-3 events in the thinnest one), so an independent fit lets
  // the thinnest group's slopes wander onto whatever handful of points it
  // has, which is a likely source of chain-to-chain disagreement. Each
  // group's vector is now a non-centered deviation from a shared mean, so
  // a data-starved group borrows strength from the better-informed ones.
  vector[3] mu_alpha_p;                 // population-average antecedent slopes (Tmax, Tmin, VPD) across groups
  vector<lower=0>[3] tau_alpha_p;       // between-group SD of each slope -- the amount of pooling
  array[Ngroups] vector[3] alpha_p_std; // non-centered group deviations (std_normal); alpha_p[g] built in transformed parameters
  //real tau_Tmax;                        // hinge location on standardized ant_Tmax
  //real delta_Tmax;                      // hinge slope change above tau_Tmax
  vector[Npanels] u_panel_p_std;        // non-scaled panel death-propensity effects
  real<lower=0> sigma_panel_p;          // sd for panel death-propensity effects

  // unnormalized Dirichlet weights (gamma trick) for each lagged covariate //
  vector<lower=0>[Nlag] gamma_Tmax;
  vector<lower=0>[Nlag] gamma_Tmin;
  vector<lower=0>[Nlag] gamma_VPD;

}

transformed parameters {

  // Constructing normalized weights //
  vector[Nlag] w_Tmax = gamma_Tmax ./ sum(gamma_Tmax);
  vector[Nlag] w_Tmin = gamma_Tmin ./ sum(gamma_Tmin);
  vector[Nlag] w_VPD = gamma_VPD ./ sum(gamma_VPD);

  // Non-centered reconstruction of the group-specific antecedent slopes;
  // see the note on alpha_p_std in the parameters block.
  array[Ngroups] vector[3] alpha_p;
  for (g in 1:Ngroups) {
    alpha_p[g] = mu_alpha_p + tau_alpha_p .* alpha_p_std[g];
  }

  // --- Construct cumulative effects ---
  vector[Ndays] ant_Tmax = Tmax * w_Tmax;
  vector[Ndays] ant_Tmin = Tmin * w_Tmin;
  vector[Ndays] ant_VPD = VPD * w_VPD;

  // Damaged needles sit delta_dmg log-units below the HEALTHY BASELINE beta[1],
  // not below their own row's fitted mean: a dead needle's Fv/Fm is ~0
  // regardless of treatment, needle age, or which tree it sits on, so no
  // covariates and no random effects enter the damaged component. Only l_Fm
  // enters per row (below), because the response is l_Fv rather than the log
  // ratio itself. This is a scalar rather than a length-N vector to keep the
  // saved output small; mu_dmg[n] is l_Fm[n] + mu_dmg_level.
  real mu_dmg_level = beta[1] - delta_dmg;

  vector[N] mu_rec;   // recovery-component mean for l_Fv
  vector[N] eta;      // logit of P(damaged)

  for (n in 1:N) {
    mu_rec[n] = X[n, ] * beta
        + u_tree_std[tree_id[n]] * sigma_tree
        + u_panel_std[panel_id[n]] * sigma_panel
        + l_Fm[n];

    eta[n] = alpha_p[group_id[n]][1] * (ant_Tmax[date_index[n]])
        // + delta_Tmax * fmax(0.0, ant_Tmax[date_index[n]] - tau_Tmax)
        + alpha_p[group_id[n]][2] * ant_Tmin[date_index[n]]
        + alpha_p[group_id[n]][3] * ant_VPD[date_index[n]]
        + u_panel_p_std[panel_id[n]] * sigma_panel_p;
  }

}

model {

  // --- priors: recovery component ---
  beta ~ normal(0, 5);
  sigma ~ student_t(10, 0, 1);
  u_tree_std ~ std_normal();
  u_panel_std ~ std_normal();
  sigma_tree ~ student_t(10, 0, 1);
  sigma_panel ~ student_t(10, 0, 1);

  // --- priors: damaged component ---
  // observed separation is ~1.6 log-units (healthy -0.232 vs damaged median -1.83)
  delta_dmg ~ lognormal(0.5, 0.75);     // median ~1.65
  sigma_dmg ~ student_t(10, 0, 2);      // damaged tail spans -1 to -6.9

  // --- priors: damage-probability model ---
  // ~2.3% of retained rows are damaged, so the logit intercept sits near -3.7.
  // A diffuse prior on a rare-event logit invites separation, so these are kept
  // tighter than the normal(0, 5) used on the mean model's coefficients.
  beta_p ~ normal(-3.5, 1.5);
  // if (Kp > 1) {
  //   beta_p[2:Kp] ~ normal(0, 2.5);
  // }

  // Partial pooling across groups (see parameters block). mu_alpha_p gets
  // the same weakly-informative prior the independent alpha_p[g] used to
  // have; tau_alpha_p controls how much each group is allowed to deviate
  // from that shared mean. With only 4 groups, tau_alpha_p's own posterior
  // will stay wide, but it still shrinks the data-starved groups toward the
  // better-informed ones instead of letting them fit on 2-3 points alone.
  for (g in 1:Ngroups) {
    alpha_p_std[g] ~ std_normal();
  }
  mu_alpha_p ~ normal(0, 1);
  tau_alpha_p ~ student_t(10, 0, 0.5);
  // delta_Tmax ~ normal(0, 2.5);
  // ant_Tmax is standardized, so it spans roughly [-2.5, 2.5]
  // tau_Tmax ~ normal(0, 1.5);
  u_panel_p_std ~ std_normal();
  sigma_panel_p ~ student_t(10, 0, 1);

  // Dirichlet(2,2,2,2,2) via the gamma trick, up from Dirichlet(1,...,1)
  // (uniform on the simplex). Uniform gives equal prior mass to a degenerate
  // solution that dumps all weight on one lag day and to a smooth profile;
  // with the antecedent terms now identified by only ~2-8 events per group
  // (not the full ~3870-row dataset, as in FvFm_antecedent.stan), the
  // data alone can't reliably rule out a spurious single-day optimum, which
  // is a likely source of the extra mode. This pulls the prior mildly
  // toward a smoother window without forcing it; raise further (e.g. 3-5)
  // if chains still disagree on w_Tmax/w_Tmin/w_VPD.
  gamma_Tmax ~ gamma(2, 1);
  gamma_Tmin ~ gamma(2, 1);
  gamma_VPD ~ gamma(2, 1);

  // --- likelihood ---
  // l_Fv ~ pi * Normal(mu_dmg, sigma_dmg) + (1 - pi) * Normal(mu_rec, sigma),
  // with logit(pi) = eta. Written with log_inv_logit/log1m_inv_logit rather than
  // log_mix(inv_logit(eta), ...) because eta sits around -3.5 for most rows,
  // where inv_logit loses precision but the log forms stay stable.
  for (n in 1:N) {
    if (pa_FvFm[n]) {
      target += log_sum_exp(
        log_inv_logit(eta[n])   + normal_lpdf(l_Fv[n] | l_Fm[n] + mu_dmg_level, sigma_dmg),
        log1m_inv_logit(eta[n]) + normal_lpdf(l_Fv[n] | mu_rec[n], sigma)
      );
    }
  }

}

generated quantities {
  vector[N] pi_dmg;        // covariate-driven P(damaged), i.e. inv_logit(eta)
  vector[N] p_dmg;         // posterior responsibility P(damaged | l_Fv) -- the classifier
  vector[N] log_lik;       // per-row mixture log-density, 0 where unobserved; for loo
  vector[N] l_Fv_rep;      // posterior-predictive replicate for log(Fv)

  for (n in 1:N) {
    real lp_d = log_inv_logit(eta[n])
                  + normal_lpdf(l_Fv[n] | l_Fm[n] + mu_dmg_level, sigma_dmg);
    real lp_r = log1m_inv_logit(eta[n])
                  + normal_lpdf(l_Fv[n] | mu_rec[n], sigma);

    pi_dmg[n] = inv_logit(eta[n]);

    // pi_dmg uses covariates only; p_dmg additionally conditions on the observed
    // value. Both are wanted -- the gap between them is where the weather model
    // fails to anticipate damage that actually happened.
    log_lik[n] = pa_FvFm[n] ? log_sum_exp(lp_d, lp_r) : 0;
    p_dmg[n] = pa_FvFm[n] ? exp(lp_d - log_sum_exp(lp_d, lp_r)) : pi_dmg[n];

    l_Fv_rep[n] = bernoulli_rng(pi_dmg[n])
                    ? normal_rng(l_Fm[n] + mu_dmg_level, sigma_dmg)
                    : normal_rng(mu_rec[n], sigma);
  }

}
