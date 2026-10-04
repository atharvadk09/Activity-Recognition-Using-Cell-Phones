import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. Dataset path
# ==========================================

DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"

# ==========================================
# 2. Load data
# ==========================================

X_train = pd.read_csv(
    f"{DATA_PATH}/train/X_train.txt",
    sep=r"\s+",
    header=None
)

y_train = pd.read_csv(
    f"{DATA_PATH}/train/y_train.txt",
    sep=r"\s+",
    header=None
).values.ravel()

X_test = pd.read_csv(
    f"{DATA_PATH}/test/X_test.txt",
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    f"{DATA_PATH}/test/y_test.txt",
    sep=r"\s+",
    header=None
).values.ravel()

# ==========================================
# 3. Activity names
# ==========================================

activity_labels = pd.read_csv(
    f"{DATA_PATH}/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_names = activity_labels["activity"].tolist()

# ==========================================
# 4. Train SVM
# ==========================================

print("Training SVM...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("SVM training completed!")

# ==========================================
# 5. Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 6. Confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 7. Display confusion matrix
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_names
)

display.plot(
    xticks_rotation=45
)

plt.title("SVM Confusion Matrix")
plt.tight_layout()

# Save figure
plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved as svm_confusion_matrix.png")