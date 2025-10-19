import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define which benchmarks belong to which dataset
online_data = {
    "offline": [
        "matmul_256_768_3072",
        "matmul_256_2048_2048",
        "matmul_256_256_512",
        "matmul_256_512_1024",
    ],
    "online": [
        "matmul_1024_128_768",
        "matmul_1024_128_512",
        "matmul_1024_1024_128",
        "matmul_1024_128_1024",
        "matmul_1024_128_2048",
    ],
}

# Load CSV
df = pd.read_csv("online_iql.csv", sep=";")

# Filter out "average_speedup" rows
df = df[df["metric"] != "average_speedup"]

# Pivot table for easier comparison
pivot_df = df.pivot(index="metric", columns="algorithm", values="score")

# Determine dataset (online/offline) for each metric
def get_dataset(metric):
    if metric in online_data["online"]:
        return "online"
    elif metric in online_data["offline"]:
        return "offline"
    else:
        return "unknown"

pivot_df["dataset"] = pivot_df.index.map(get_dataset)

# Assign colors based on dataset
color_map = {"online": "blue", "offline": "red", "unknown": "gray"}
colors = pivot_df["dataset"].map(color_map)

# Plot grouped horizontal bars
ax = pivot_df[["Online Finetuned IQL", "PPO"]].plot.barh(
    figsize=(10, 6),
    color=["#1f77b4", "#ff7f0e"],
    edgecolor="black"
)

# Apply y-labels and color backgrounds per dataset type
for i, (dataset, metric) in enumerate(zip(pivot_df["dataset"], pivot_df.index)):
    ax.get_yticklabels()[i].set_color(color_map[dataset])

plt.xlabel("Speedup")
plt.ylabel("Benchmark")
plt.title("Online Finetuned IQL vs PPO — Benchmark Speedup Comparison")

# Create legend
blue_patch = mpatches.Patch(color="blue", label="Online data")
red_patch = mpatches.Patch(color="red", label="Offline data")
orange_patch = mpatches.Patch(color="#ff7f0e", label="PPO")
blue_bar_patch = mpatches.Patch(color="#1f77b4", label="IQL")

plt.legend(handles=[blue_bar_patch, orange_patch, blue_patch, red_patch], loc="best")

plt.tight_layout()
plt.savefig("online_offline_iql_vs_ppo.png", dpi=300)
plt.show()

print("✅ Plot saved as online_offline_iql_vs_ppo.png")
