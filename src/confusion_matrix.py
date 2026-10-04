import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from load_data import load_data


# Load dataset
X_train, y_train, X_test, y_test, _, activity_labels = load_data()

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# Train SVM
print("Training SVM...")

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

svm.fit(X_train, y_train)

print("SVM training completed!")


# Predictions
y_pred = svm.predict(X_test)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Activity names
activity_names = activity_labels["activity"].tolist()


# Create confusion matrix plot
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_names
)

fig, ax = plt.subplots(figsize=(9, 7))

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.title("SVM Confusion Matrix")
plt.tight_layout()

# Save figure
plt.savefig(
    "results/svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print("results/svm_confusion_matrix.png")