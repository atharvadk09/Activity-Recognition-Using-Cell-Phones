import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_data


# Load dataset
X_train, y_train, X_test, y_test, _, _ = load_data()
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Starting SVM hyperparameter tuning...")

# Parameters to test
param_grid = {
    "C": [1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

# Base SVM
svm = SVC()

# 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Train
grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(f"{grid_search.best_score_ * 100:.2f}%")

# Evaluate best model on test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(f"{test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save tuning results
results = pd.DataFrame({
    "Best C": [grid_search.best_params_["C"]],
    "Best Gamma": [grid_search.best_params_["gamma"]],
    "Kernel": [grid_search.best_params_["kernel"]],
    "CV Accuracy": [grid_search.best_score_],
    "Test Accuracy": [test_accuracy]
})

results.to_csv("results/svm_tuning_results.csv", index=False)

print("\nTuning results saved to:")
print("results/svm_tuning_results.csv")