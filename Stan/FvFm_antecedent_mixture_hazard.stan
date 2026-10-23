// Two-component mixture for Fv/Fm: irreparable damage vs. recovery, with
// damage probability driven by the current value of the recovery trajectory
// (a joint longitudinal/hazard structure) rather than its own antecedent
// submodel.
//
// Variant of FvFm_antecedent_mixture.stan. That model put the antecedent
// (SAM) structure directly on the damage-probability logit: group-specific
// alpha_p slopes plus three Dirichlet lag-weights,
// estimated almost entirely from the rare damaged tail (as few as 2-3
// events in the thinnest tree_treatment x heat_treatment group). That is a
// lot of structure asked of very little data, and is the suspected cause of
// the chain-mixing problems diagnosed in that model (3 of 4 chains agreeing,
// 1 landing on a different answer for everything logit-related) -- either
// the shared lag-window placement or a thin group's slopes (or both) have a
// second, comparably-good solution that 19-95 rows can't rule out.
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
// FvFm_antecedent_mixture.stan's header: the mixture now lives on the RAW
// Fv scale rather than log(Fv). The recovery component is
// lognormal_lpdf(Fv | mu_rec, sigma) -- the identical distributional
// assumption as log(Fv) ~ normal(mu_rec, sigma), just reparameterized with
// the Jacobian folded in by Stan's built-in lognormal functions -- which
// has support only on Fv > 0. The damaged component is
// normal_lpdf(Fv | 0, sigma_dmg), with full support on the real line,
// matching the observed instrument noise around a true zero (damaged
// needles have Fv scattered in roughly (-1.9, 1.79)). This lets the 43 rows
// with Fv <= 0 -- previously dropped in R because log(Fv) is undefined --
// re-enter the likelihood. They fall out almost entirely into the damaged
// component, which is scientifically correct: they are the most severely
// damaged needles, not missing-at-random noise.
//
// Because the damaged mean is now the fixed physical constant 0 (raw Fv),
// not an estimated quantity, there is no delta_dmg/label-switching
// parameter here: the components are separated by orders of magnitude on
// the raw scale (healthy Fv ~ exp(-0.232)*Fm ~ 0.79*200 ~ 158, versus
// damaged-component values near 0), not the ~1.6 log-unit gap that made an
// explicit ordering constraint necessary in the log-scale version.
//
// Stan has no NA, so missing FvFm entries are passed in as a presence/
// absence indicator; see the data block below for exactly what needs to be
// precomputed in R before calling this model.
//
// Antecedent slopes (alpha) are SHARED across tree_treatment x heat_treatment
// groups here, not group-specific -- a deliberate step back from a richer
// version of this model, since even on the well-identified recovery mean the
// group-specific alphas were still showing mixing problems. One shared
// Tmax/Tmin/VPD sensitivity is a simpler, more stable starting point; splitting
// it back out by group is a natural next step once this version is solid.
// Ngroups/group_id are no longer read by this file -- harmless to keep
// passing them in stan_data (rstan ignores unmatched data-list entries), no
// R-side changes required.

data {

  int<lower=1> Ndays;                // number days for which the mean can be calculated (Ndays_total - Nlag)
  int<lower=1> Ntrees;               // Number of trees in data
  int<lower=1> Nlag;                 // number of lag days included in the antecedent window
  int<lower=1> N;                    // total number of rows in the data
  int<lower=1> K;                    // columns in the mean design matrix. Intercept must be column 1
  int<lower=1> Npanels;              // Number of unique panels/branches tracked

  // Fv/Fm is modeled on the raw Fv scale (not log(Fv)): the recovery
  // component below is lognormal (equivalent to log(Fv) ~ normal(mu, sig)),
  // and the damaged component is normal with a mean fixed at zero. Working
  // on the raw scale means Fv can legitimately be <= 0 -- those rows just
  // get all their mixture-likelihood mass from the damaged component.
  vector[N] Fv;                                     // raw variable fluorescence (Fm - Fo); placeholder (e.g. 0) where missing
  vector[N] l_Fm;                                   // log(maximal fluorescence), used as an offset; placeholder (e.g. 0) where missing
  array[N] int<lower=0, upper=1> pa_FvFm;           // presence/absence of a paired Fv/Fm measurement
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


parameters {

  // --- recovery component mean, with the antecedent (SAM) structure
  // identical in form to FvFm_antecedent.stan --- //
  vector[K] beta;                       // regression coefficients: intercept, plus design covariates
  vector[3] alpha;                      // antecedent coefficients (Tmax, Tmin, VPD), shared across all tree_treatment x heat_treatment groups
  // real tau_Tmax;                        // hinge location on standardized ant_Tmax
  // real delta_Tmax;                      // hinge slope change above tau_Tmax
  real<lower=0> sigma;                  // recovery-component residual SD
  vector[Ntrees] u_tree_std;            // non-scaled tree random effects
  vector[Npanels] u_panel_std;          // non-scaled panel effects
  real<lower=0> sigma_tree;             // sd for tree effects
  real<lower=0> sigma_panel;            // sd for panel effects

  // unnormalized Dirichlet weights (gamma trick) for each lagged covariate //
  vector<lower=0>[Nlag] gamma_Tmax;
  vector<lower=0>[Nlag] gamma_Tmin;
  vector<lower=0>[Nlag] gamma_VPD;

  // --- damaged component: fixed at raw Fv = 0, only a scale to estimate --- //
  real<lower=0> sigma_dmg;              // damaged-component SD on the raw Fv scale (expect ~0.6-0.9; see prior note)

  // --- tipping/hazard model: P(damaged) as a function of how far mu_rec
  // has fallen below the healthy baseline --- //
  real kappa0;                          // logit intercept at zero depression
  real<lower=0> kappa1;                 // slope on depression below baseline; >0 since more depressed must mean more likely damaged
  // vector[Npanels] u_panel_p_std;        // non-scaled panel death-propensity effects, beyond what depression already explains
  // real<lower=0> sigma_panel_p;          // sd for panel death-propensity effects

}

transformed parameters {

  // Constructing normalized weights //
  vector[Nlag] w_Tmax = gamma_Tmax ./ sum(gamma_Tmax);
  vector[Nlag] w_Tmin = gamma_Tmin ./ sum(gamma_Tmin);
  vector[Nlag] w_VPD = gamma_VPD ./ sum(gamma_VPD);

  // --- Construct cumulative effects ---
  vector[Ndays] ant_Tmax = Tmax * w_Tmax;
  vector[Ndays] ant_Tmin = Tmin * w_Tmin;
  vector[Ndays] ant_VPD = VPD * w_VPD;

  vector[N] mu_rec;    // recovery-component mean for Fv on the log scale (as in FvFm_antecedent.stan's mu)
  vector[N] eta;       // logit of P(damaged)

  for (n in 1:N) {
    mu_rec[n] = X[n, ] * beta
        + alpha[1] * ant_Tmax[date_index[n]]
        // + delta_Tmax * fmax(0.0, ant_Tmax[date_index[n]] - tau_Tmax)
        + alpha[2] * ant_Tmin[date_index[n]]
        + alpha[3] * ant_VPD[date_index[n]]
        + u_tree_std[tree_id[n]] * sigma_tree
        + u_panel_std[panel_id[n]] * sigma_panel
        + l_Fm[n];

    // Depression below the healthy baseline, on the log(Fv/Fm) scale (strip
    // the l_Fm offset back out so this is comparable to beta[1], the
    // group-level healthy intercept). fmax(.,0) means nothing happens until
    // the fitted trajectory actually sinks below baseline -- the tipping
    // point -- after which the odds of damage rise with depression. This
    // responds to depression from ANY source (antecedent weather, needle
    // age, panel idiosyncrasies, ...): a branch that is already running low
    // for whatever reason is closer to collapse.
    real d_n = beta[1] - (mu_rec[n] - l_Fm[n]);
    eta[n] = kappa0 + kappa1 * fmax(0.0, d_n);
        //+ u_panel_p_std[panel_id[n]] * sigma_panel_p;
  }

}

model {

  // --- priors: recovery-component mean / antecedent structure, copied
  // verbatim from FvFm_antecedent.stan -- these are back on the full
  // dataset and don't need the tail-insurance tightening used when this
  // structure lived in the damage logit --- //
  beta ~ normal(0, 5);
  alpha ~ normal(0, 5);
  sigma ~ student_t(10, 0, 1);
  u_tree_std ~ std_normal();
  u_panel_std ~ std_normal();
  sigma_tree ~ student_t(10, 0, 1);
  sigma_panel ~ student_t(10, 0, 1);

  gamma_Tmax ~ gamma(2, 1);
  gamma_Tmin ~ gamma(2, 1);
  gamma_VPD ~ gamma(2, 1);

  // delta_Tmax ~ normal(0, 1);
  // tau_Tmax ~ normal(0, 10);

  // --- priors: damaged component ---
  // Raw-Fv scale: healthy Fv ~ exp(-0.232)*Fm ~ 158, damaged-Fm-row
  // recovery-predicted Fv ~ exp(-0.232)*24 ~ 19, versus actual damaged
  // values scattered roughly in (-1.9, 1.79) around a true zero -- implying
  // an SD on the order of 0.6-0.9. sigma_dmg would need to blow out past
  // ~5-10 before meaningfully overlapping recovery territory, so no
  // explicit ordering constraint (unlike the log-scale delta_dmg) is needed
  // to keep the components apart; confirm post-fit rather than assuming it.
  sigma_dmg ~ student_t(10, 0, 1);

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
  // Fv ~ pi * Normal(0, sigma_dmg) + (1 - pi) * LogNormal(mu_rec, sigma),
  // with logit(pi) = eta. lognormal_lpdf(Fv | mu_rec, sigma) is exactly
  // normal_lpdf(log(Fv) | mu_rec, sigma) with the Jacobian folded in, so
  // this is the same recovery-component assumption as the log-scale models,
  // just expressed in Fv units so it can share a mixture with a component
  // that has support at and below zero. Explicitly guard Fv[n] > 0 for the
  // recovery term rather than relying on lognormal_lpdf's built-in domain
  // check -- Fv[n] <= 0 rows then get all their mixture mass, correctly,
  // from the damaged component.
  for (n in 1:N) {
    if (pa_FvFm[n]) {
      real lp_d = log_inv_logit(eta[n]) + normal_lpdf(Fv[n] | 0, sigma_dmg);
      real lp_r = (Fv[n] > 0)
          ? log1m_inv_logit(eta[n]) + lognormal_lpdf(Fv[n] | mu_rec[n], sigma)
          : negative_infinity();
      target += log_sum_exp(lp_d, lp_r);
    }
  }

}

generated quantities {
  vector[N] pi_dmg;        // covariate-driven P(damaged), i.e. inv_logit(eta)
  vector[N] p_dmg;         // posterior responsibility P(damaged | Fv) -- the classifier
  vector[N] log_lik;       // per-row mixture log-density, 0 where unobserved; for loo
  vector[N] Fv_rep;        // posterior-predictive replicate for raw Fv

  for (n in 1:N) {
    real lp_d = log_inv_logit(eta[n]) + normal_lpdf(Fv[n] | 0, sigma_dmg);
    real lp_r = (Fv[n] > 0)
        ? log1m_inv_logit(eta[n]) + lognormal_lpdf(Fv[n] | mu_rec[n], sigma)
        : negative_infinity();

    pi_dmg[n] = inv_logit(eta[n]);

    // pi_dmg uses the fitted trajectory only; p_dmg additionally conditions
    // on the observed value. Both are wanted -- the gap between them is
    // where the depression-based hazard fails to anticipate damage that
    // actually happened.
    log_lik[n] = pa_FvFm[n] ? log_sum_exp(lp_d, lp_r) : 0;
    p_dmg[n] = pa_FvFm[n] ? exp(lp_d - log_sum_exp(lp_d, lp_r)) : pi_dmg[n];

    Fv_rep[n] = bernoulli_rng(pi_dmg[n])
                    ? normal_rng(0, sigma_dmg)
                    : lognormal_rng(mu_rec[n], sigma);
  }

}
