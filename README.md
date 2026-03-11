# insurance-experience

Individual policy-level Bayesian posterior experience rating for insurance pricing.

The problem this solves: your GLM gives every "standard 35-year-old with a clean licence" the same premium. But a 35-year-old who has made three claims in the past four years is a different risk from one who has made none. NCD handles this crudely. This library does it properly.

`insurance-experience` computes a posteriori (experience-adjusted) premiums using the full claims history of each individual policy. The output is a multiplicative credibility factor that slots directly into your existing GLM rating engine:

```
posterior_premium = prior_premium × credibility_factor
```

Your GLM does the covariate work. This library does the longitudinal experience work. Neither replaces the other.

## Who this is for

Pricing actuaries and data scientists working on:
- Motor fleet experience rating (replacing static mod factors with time-decayed Bayesian updates)
- UK personal lines renewal pricing under FCA PS21/5 (individual experience as a defensible pricing signal)
- Commercial liability experience modification
- Home insurance repeat-claimant pricing

## The four model tiers

### 1. StaticCredibilityModel
Bühlmann-Straub applied at individual policy level. Estimates a single kappa parameter (within/between variance ratio) from the portfolio. Simple, interpretable, fast.

Best for: portfolios with 2–4 years of history, limited computational budget, or actuaries who need to explain the formula to a committee.

### 2. DynamicPoissonGammaModel
State-space Poisson-gamma with seniority weighting. Parameters p and q control how quickly older claims are discounted. Fitted by empirical Bayes MLE on the portfolio log-likelihood (tractable via negative binomial marginals, no MCMC required).

Reference: Ahn, Jeong, Lu, Wüthrich, "Dynamic Bayesian Credibility", arXiv:2308.16058.

Best for: fleets with 3+ years of history where risk quality genuinely changes over time, or situations where a policy's accident record from 5 years ago should count for less than last year's.

### 3. SurrogateModel
For non-conjugate Bayesian models where the true posterior requires MCMC. Computes exact Bayesian premiums via importance sampling on a representative sub-portfolio (1–10%), then fits a WLS function g(.) that approximates log(CF) from a sufficient statistic. Scales to large portfolios without per-policy MCMC.

Reference: Calcetero Vanegas, Badescu, Lin, Insurance: Mathematics and Economics 118 (2024).

Best for: when you have a complex severity model (e.g., lognormal with covariates) and the posterior is intractable, but you need to score 100k+ policies.

### 4. DeepAttentionModel
Linear attention over the claims sequence. Replaces fixed Bühlmann credibility weights with learned weights that can depend on covariates (exposure level, period, claims in that period). Distribution-free fitting via Poisson deviance.

Reference: Wüthrich, "Experience Rating in Insurance Pricing", SSRN 4726206 (2024).

Requires: `pip install insurance-experience[deep]`

Best for: large portfolios (10k+ policies, 5+ years of history) where the relationship between history and risk is non-linear, and where you have resource to train and validate a neural model.

## Installation

```bash
pip install insurance-experience

# With deep attention model support (torch required):
pip install insurance-experience[deep]
```

## Quick start

```python
from insurance_experience import ClaimsHistory, StaticCredibilityModel, balance_calibrate

# Describe each policy's claims history
histories = [
    ClaimsHistory(
        policy_id="FLT001",
        periods=[1, 2, 3, 4],
        claim_counts=[0, 1, 0, 0],
        exposures=[1.0, 1.0, 1.0, 0.75],  # years on risk
        prior_premium=12_500.0,             # GLM output
    ),
    ClaimsHistory(
        policy_id="FLT002",
        periods=[1, 2, 3, 4],
        claim_counts=[3, 2, 1, 3],
        exposures=[1.0, 1.0, 1.0, 1.0],
        prior_premium=12_500.0,
    ),
    # ... rest of portfolio
]

# Fit on portfolio (estimates kappa from cross-sectional spread)
model = StaticCredibilityModel()
model.fit(histories)

# Predict credibility factors
cf_good = model.predict(histories[0])   # < 1.0 (better than prior)
cf_bad  = model.predict(histories[1])   # > 1.0 (worse than prior)

# Apply to get posterior premiums
posterior_good = histories[0].prior_premium * cf_good
posterior_bad  = histories[1].prior_premium * cf_bad

# Enforce the balance property: sum(posterior) ≈ sum(actual)
cal = balance_calibrate(model.predict, histories)
print(f"Calibration factor: {cal.calibration_factor:.4f}")

# Production scoring
df = model.predict_batch(histories)
# Returns Polars DataFrame: policy_id, prior_premium, credibility_factor, posterior_premium
```

## Dynamic model with seniority weighting

```python
from insurance_experience import DynamicPoissonGammaModel

model = DynamicPoissonGammaModel()
model.fit(histories)

print(f"Fitted p={model.p_:.3f}, q={model.q_:.3f}")
# p close to 1: strong mean-reversion each period
# q close to 1: older claims retain more weight

# Get uncertainty alongside the point estimate
alpha, beta = model.predict_posterior_params(histories[0])
posterior_mean = alpha / beta
posterior_variance = alpha / beta**2
```

## Surrogate model for large portfolios

```python
from insurance_experience import SurrogateModel

# Sub-portfolio fitting: 5% of policies used for IS calibration
model = SurrogateModel(n_is_samples=2000, subsample_frac=0.05, random_state=42)
model.fit(histories)

# Scoring the full portfolio is instant (analytical g(.) evaluation)
df = model.predict_batch(all_histories)
```

## Balance property

The balance property (Wüthrich 2020, Lindholm & Wüthrich 2025) requires that the sum of posterior premiums equals the sum of observed claims at portfolio level. This is the self-financing constraint: experience rating redistributes the total premium, but does not inflate it.

All models support post-fit balance calibration:

```python
from insurance_experience import balance_calibrate, calibrated_predict_fn, balance_report

cal = balance_calibrate(model.predict, histories)
calibrated = calibrated_predict_fn(model.predict, cal)

# Diagnostic report
report = balance_report(model.predict, histories, by_n_periods=True)
print(report)
```

## Data schema

```python
ClaimsHistory(
    policy_id    = "FLT001",      # str, unique
    periods      = [1, 2, 3],     # list[int], year indices (ascending, unique)
    claim_counts = [0, 1, 0],     # list[int], non-negative
    claim_amounts= [0, 1500, 0],  # list[float] | None (for severity models)
    exposures    = [1.0, 1.0, 0.75],  # list[float], years on risk
    prior_premium= 12_500.0,      # float > 0, GLM output for next period
)
```

## Design choices

**Multiplicative credibility factor.** All models output CF = posterior/prior rather than the posterior itself. This keeps the library composable: you can replace the GLM, swap the experience model, or apply multiple experience adjustments independently.

**sklearn-style interface.** `fit(histories)` then `predict(history)`. No hidden state, no global configuration. Fitted parameters are plain attributes (kappa_, p_, q_) — readable in an audit trail.

**Polars-native batch output.** `predict_batch()` returns a Polars DataFrame. At portfolio scale (100k+ policies), Polars is materially faster than pandas for the column operations pricing teams actually do (group-bys by segment, quantile distributions of CF, join to rating factors).

**Balance calibration separate from fitting.** Some libraries bake calibration into training. We don't. The calibration factor is explicit and auditable. You can see exactly how much the model was rescaled, and in which direction.

**torch as optional dependency.** The deep attention model requires torch. Requiring torch as a mandatory dependency would be unreasonable for teams who only want Bühlmann-Straub. The `[deep]` extra makes the dependency explicit.

## Relationship to other Burning Cost libraries

- **insurance-credibility**: Bühlmann-Straub and Hachemeister at **group** level (fleet, territory, scheme). Use that for pooling across groups. Use this for updating individual policy premiums.
- **experience-rating**: NCD/bonus-malus Markov chains. That library implements the contractual NCD structure. This library implements the actuarially optimal Bayesian replacement for NCD.

## References

- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory*. Springer.
- Ahn, J., Jeong, H., Lu, Y. & Wüthrich, M.V. (2023). Dynamic Bayesian credibility. arXiv:2308.16058.
- Calcetero Vanegas, S., Badescu, A. & Lin, X.S. (2024). Effective experience rating via surrogate modelling. *Insurance: Mathematics and Economics* 118, 25–43. arXiv:2211.06568.
- Wüthrich, M.V. (2024). Experience rating in insurance pricing. SSRN 4726206.
- Wüthrich, M.V. (2020). Bias regularization in neural network models. *European Actuarial Journal* 10, 179–202.
- Lindholm, M. & Wüthrich, M.V. (2025). The balance property in insurance pricing. *Scandinavian Actuarial Journal*.

## Licence

MIT. See [LICENSE](LICENSE).
