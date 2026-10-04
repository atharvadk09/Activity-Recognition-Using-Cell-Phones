import pandas as pd
import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. Dataset path
# ==========================================

DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"

# ==========================================
# 2. Load dataset
# ==========================================

print("Loading dataset...")

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

print("Dataset loaded!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Features:", X_train.shape[1])

# ==========================================
# 3. Define models
# ==========================================

models = {
    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 4. Train and evaluate
# ==========================================

results = []

for name, model in models.items():

    print("\n--------------------------------")
    print("Training:", name)
    print("--------------------------------")

    # Training time
    start_train = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_train

    # Prediction time
    start_prediction = time.time()

    y_pred = model.predict(X_test)

    prediction_time = time.time() - start_prediction

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (s)": train_time,
        "Prediction Time (s)": prediction_time
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Training Time:", round(train_time, 2), "seconds")
    print("Prediction Time:", round(prediction_time, 2), "seconds")


# ==========================================
# 5. Create comparison table
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(results_df.to_string(index=False))

# ==========================================
# 6. Save results
# ==========================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to model_comparison_results.csv")