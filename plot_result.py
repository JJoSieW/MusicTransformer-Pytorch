import pandas as pd
import matplotlib.pyplot as plt
import os
# File paths
# root_dir = "final_results/metrics_generated/new_fixed_files/v3_new_metrics"
# contour_csv = os.path.join(root_dir, "con_results.csv")
# ablation_csv = os.path.join(root_dir, "ablation_results.csv")

root_dir = "final_results/metrics_generated/new_fixed_files/v4"
contour_csv = os.path.join(root_dir, "con_evaluation_results.csv")
ablation_csv = os.path.join(root_dir, "ablation_evaluation_results.csv")

# Define common bins for consistent comparison
import numpy as np
bins = np.linspace(-50, 50, 30)

# Load data
df_contour = pd.read_csv(contour_csv)
df_ablation = pd.read_csv(ablation_csv)

# Extract slope_ratio column
if "slope_ratio" in df_contour.columns:
    contour_slope = df_contour['slope_ratio'].dropna()
    ablation_slope = df_ablation['slope_ratio'].dropna()
else:
    contour_slope = df_contour['slope_score'].dropna()
    ablation_slope = df_ablation['slope_score'].dropna()

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(contour_slope, bins=bins, alpha=0.6, density=True, label='Contour-conditioned Model', color='blue', edgecolor='black')
plt.hist(ablation_slope, bins=bins, alpha=0.6, density=True, label='Ablation Model', color='orange', edgecolor='black')

# Labels and title
plt.xlabel('Slope Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Slope Ratios: Contour-conditioned vs Ablation Models')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Save or show plot
plt.tight_layout()
plt.savefig("figures/slope_ratio_distribution_comparison.png", dpi=300)
plt.show()
