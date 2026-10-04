import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from load_data import load_data


# Load dataset
X_train, y_train, X_test, y_test, _, activity_labels = load_data()

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# Train final SVM
print("Training final SVM model...")

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

svm.fit(X_train, y_train)

print("Training completed.")


# Predictions
y_pred = svm.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nFinal SVM Accuracy:")
print(f"{accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=activity_labels["activity"]
))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
activity_names = activity_labels["activity"].tolist()

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

plt.title("Final SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/final_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFinal confusion matrix saved to:")
print("results/final_confusion_matrix.png")