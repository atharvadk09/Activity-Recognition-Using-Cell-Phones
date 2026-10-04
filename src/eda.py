import pandas as pd
import matplotlib.pyplot as plt

# Dataset path
DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"

# -----------------------------
# Load labels
# -----------------------------
y_train = pd.read_csv(
    f"{DATA_PATH}/train/y_train.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id"]
)

y_test = pd.read_csv(
    f"{DATA_PATH}/test/y_test.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id"]
)

# Activity names
activity_labels = pd.read_csv(
    f"{DATA_PATH}/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id", "activity"]
)

# Convert IDs to activity names
activity_map = dict(
    zip(activity_labels["activity_id"], activity_labels["activity"])
)

y_train["activity"] = y_train["activity_id"].map(activity_map)
y_test["activity"] = y_test["activity_id"].map(activity_map)

# -----------------------------
# Print class distribution
# -----------------------------
print("Training activity distribution:")
print(y_train["activity"].value_counts())

print("\nTesting activity distribution:")
print(y_test["activity"].value_counts())

# -----------------------------
# Plot training distribution
# -----------------------------
plt.figure(figsize=(9, 5))

y_train["activity"].value_counts().plot(kind="bar")

plt.title("Training Dataset - Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()