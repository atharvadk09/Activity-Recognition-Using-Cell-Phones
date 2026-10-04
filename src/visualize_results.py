import pandas as pd
import matplotlib.pyplot as plt

# Load model results
results = pd.read_csv("model_comparison_results.csv")

print("Model Results:")
print(results.to_string(index=False))

# ==========================================
# Accuracy comparison
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Display accuracy values above bars
for i, value in enumerate(results["Accuracy"]):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

# Save figure
plt.savefig("model_accuracy_comparison.png", dpi=300)

plt.show()

print("\nAccuracy graph saved as model_accuracy_comparison.png")