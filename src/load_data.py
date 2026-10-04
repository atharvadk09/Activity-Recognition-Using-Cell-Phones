import pandas as pd

# Dataset path
DATA_PATH = "data/UCI HAR Dataset/UCI HAR Dataset"


def load_data():
    # -----------------------------
    # Load feature names
    # -----------------------------
    features = pd.read_csv(
        f"{DATA_PATH}/features.txt",
        sep=r"\s+",
        header=None,
        names=["id", "feature"]
    )

    feature_names = features["feature"].tolist()

    # -----------------------------
    # Load training data
    # -----------------------------
    X_train = pd.read_csv(
        f"{DATA_PATH}/train/X_train.txt",
        sep=r"\s+",
        header=None
    )

    X_train.columns = feature_names

    # -----------------------------
    # Load testing data
    # -----------------------------
    X_test = pd.read_csv(
        f"{DATA_PATH}/test/X_test.txt",
        sep=r"\s+",
        header=None
    )

    X_test.columns = feature_names

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

    # -----------------------------
    # Activity names
    # -----------------------------
    activity_labels = pd.read_csv(
        f"{DATA_PATH}/activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity"]
    )

    activity_map = dict(
        zip(activity_labels["activity_id"], activity_labels["activity"])
    )

    y_train["activity"] = y_train["activity_id"].map(activity_map)
    y_test["activity"] = y_test["activity_id"].map(activity_map)

    return X_train, y_train["activity_id"], X_test, y_test["activity_id"], feature_names, activity_labels


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names, activity_labels = load_data()

    print("Dataset loaded successfully!\n")

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\nNumber of features:", len(feature_names))

    print("\nFirst 10 feature names:")
    for feature in feature_names[:10]:
        print(feature)

    print("\nFirst 5 rows of training data:")
    print(X_train.iloc[:5, :5])

    print("\nActivity labels:")
    print(activity_labels)