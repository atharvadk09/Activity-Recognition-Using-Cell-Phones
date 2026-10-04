import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC

DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"

# ==========================================
# Load data
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

# Activity names
activity_labels = pd.read_csv(
    f"{DATA_PATH}/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_names = activity_labels["activity"].tolist()

# ==========================================
# Train final SVM
# ==========================================

print("Training final SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("Training completed.")

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_names
)

display.plot(xticks_rotation=45)

plt.title("Final SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/final_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: results/final_confusion_matrix.png")
