import numpy as np

rng = np.random.default_rng(42)
ref = rng.normal(loc=0, scale=1, size=500)
new = rng.normal(loc=0.4, scale=1.3, size=500)

# 1) Bins
bins = np.linspace(min(ref.min(), new.min()), max(ref.max(), new.max()), 11)

# 2) Histogram proportions
ref_counts, _ = np.histogram(ref, bins=bins)
new_counts, _ = np.histogram(new, bins=bins)
ref_prop = ref_counts / len(ref)
new_prop = new_counts / len(new)

# 3) PSI calculation
eps = 1e-6
psi_values = (ref_prop - new_prop) * np.log((ref_prop + eps) / (new_prop + eps))

# 4) Total PSI
psi_total = np.sum(psi_values)

print("PSI per bin:", psi_values)
print("Total PSI:", psi_total)