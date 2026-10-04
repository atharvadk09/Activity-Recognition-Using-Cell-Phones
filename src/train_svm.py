import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Dataset path
# ==========================================

DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"

# ==========================================
# 2. Load training data
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

# ==========================================
# 3. Load testing data
# ==========================================

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
# 4. Load activity names
# ==========================================

activity_labels = pd.read_csv(
    f"{DATA_PATH}/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_names = activity_labels["activity"].tolist()

# ==========================================
# 5. Create SVM model
# ==========================================

print("Training SVM...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

# ==========================================
# 6. Train
# ==========================================

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 7. Make predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 8. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

# ==========================================
# 9. Classification report
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=activity_names
    )
)

# ==========================================
# 10. Confusion matrix
# ==========================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))