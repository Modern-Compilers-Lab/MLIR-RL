import pandas as pd
import matplotlib.pyplot as plt

# Your online dataset benchmarks
online_data = [
    "matmul_256_768_2",
    "matmul_256_768_3072",
    "matmul_256_2048_2048",
    "matmul_256_256_512",
    "matmul_256_1024_1024",
    "matmul_256_1536_1000",
    "matmul_256_256_128",
    "matmul_256_512_1024",
    "matmul_256_1536_4096",
    "matmul_256_1408_1000",
    "matmul_256_1280_1000",
    "matmul_256_768_768",
    "matmul_256_2048_1000",
    "matmul_256_4096_1024",
    "matmul_256_128_256",
    "matmul_1024_128_768",
    "matmul_1024_2048_128",
    "matmul_1024_128_256",
    "matmul_1024_128_512",
    "matmul_1024_1024_128",
    "matmul_1024_1024_256",
    "matmul_1024_128_1024",
    "matmul_1024_1536_128",
    "matmul_1024_128_128",
    "matmul_1024_128_2048",
]

# Load your CSV file
df = pd.read_csv("online_iql.csv", sep=";")

# Exclude "average_speedup" from bar chart (optional, keep only benchmarks)
benchmarks_df = df[df["metric"] != "average_speedup"]

# Assign colors depending on online/offline
colors = [
    "blue" if metric in online_data else "red"
    for metric in benchmarks_df["metric"]
]

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(benchmarks_df["metric"], benchmarks_df["score"], color=colors)

plt.xlabel("Score")
plt.ylabel("Benchmark")
plt.title("Online finetuning of IQL")

# Add legend manually
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color="blue", label="Online data")
red_patch = mpatches.Patch(color="red", label="Offline data")
plt.legend(handles=[blue_patch, red_patch])

plt.tight_layout()
plt.savefig("online_offline_iql.png")

print("Plot saved as online_offline_iql.png")
