import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
sample_ref = rng.normal(loc=0, scale=1, size=200)
sample_new = rng.normal(loc=0.5, scale=1.2, size=200)

# 1) Perform KS test
ks_stat, p_value = stats.ks_2samp(sample_ref, sample_new)

# 2) Significance level
alpha = 0.05

# 3) Drift detection
drift_detected = p_value < alpha

print("KS Statistic:", ks_stat)
print("P-value:", p_value)
print("Drift detected:", drift_detected)