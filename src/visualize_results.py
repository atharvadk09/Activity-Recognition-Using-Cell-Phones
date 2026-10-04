import pandas as pd
import matplotlib

# Use a non-GUI backend so Tkinter is not required
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Load model comparison results
results = pd.read_csv("model_comparison_results.csv")

print("Model Results:")
print(results)


# Create accuracy comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"] * 100
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)


# Display accuracy values above each bar
for i, accuracy in enumerate(results["Accuracy"]):
    plt.text(
        i,
        accuracy * 100 + 1,
        f"{accuracy * 100:.2f}%",
        ha="center"
    )


plt.tight_layout()


# Save graph inside results folder
plt.savefig(
    "results/model_accuracy_comparison.png",
    dpi=300
)


print("\nAccuracy graph saved to:")
print("results/model_accuracy_comparison.png")