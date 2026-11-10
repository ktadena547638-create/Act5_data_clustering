import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Tadena_student_data.csv")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df)

cluster_colors = {0: "red", 1: "green", 2: "blue"}
cluster_labels = {
    0: "Struggling Students",
    1: "Average Students",
    2: "Top Students",
}

plt.figure(figsize=(10, 6))

for cluster_id in sorted(df["cluster"].unique()):
    group = df[df["cluster"] == cluster_id]
    plt.scatter(
        group["study_hours_per_day"],
        group["exam_score"],
        color=cluster_colors[cluster_id],
        label=cluster_labels[cluster_id],
        alpha=0.7,
        edgecolors="black",
        linewidths=0.4,
        s=70,
    )

plt.title("Student Clusters: Study Hours vs Exam Score", fontsize=14, fontweight="bold")
plt.xlabel("Study Hours Per Day (standardized)", fontsize=12)
plt.ylabel("Exam Score (standardized)", fontsize=12)
plt.legend(title="Cluster Group", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plot_path = "Tadena_cluster_plot.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print("Cluster plot saved to:", plot_path)
print()
print("Cluster summary (number of students per group):")
print(df["cluster"].value_counts().sort_index())
print()
print("Cluster assignments:")
for cid, label in cluster_labels.items():
    count = (df["cluster"] == cid).sum()
    print(f"  Cluster {cid} = {label}: {count} students")
