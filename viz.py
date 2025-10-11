import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("./comparaison.csv", sep=";")

# Pivot to have algorithms as columns
pivot_df = df.pivot(index="metric", columns="algorithm", values="score")

# Sort metrics alphabetically for consistency (optional)
pivot_df = pivot_df.sort_index()

# Plot horizontal bars
ax = pivot_df.plot(kind="barh", figsize=(10, 7))
plt.xlabel("Score")
plt.ylabel("Benchmark / Metric")
plt.title("Comparison of PPO vs Offline IQL across Benchmarks")
plt.legend(title="Algorithm")
plt.tight_layout()

# Save as PNG
plt.savefig("ppo_vs_iql_comparison.png")

print("Plot saved as ppo_vs_iql_comparison.png")
