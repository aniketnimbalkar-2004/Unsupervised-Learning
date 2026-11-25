"""Compare five supervised classifiers on the Wisconsin breast-cancer dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str], scale: bool
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", Pipeline(steps=numeric_steps), numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0)


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), True),
        "Support Vector Machine": (
            SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
            True,
        ),
        "Gaussian Naive Bayes": (GaussianNB(), False),
    }

    rows = []
    for model_name, (estimator, scale) in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features, scale),
                ),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "malignant_precision": precision_score(
                    y_test, predictions, pos_label=0, zero_division=0
                ),
                "malignant_recall": recall_score(
                    y_test, predictions, pos_label=0, zero_division=0
                ),
                "malignant_f1": f1_score(
                    y_test, predictions, pos_label=0, zero_division=0
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "malignant_f1", ascending=False, ignore_index=True
    )
    results.insert(0, "rank", range(1, len(results) + 1))

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(results["model"][::-1], results["malignant_f1"][::-1])
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 score (malignant class)")
    axis.set_title("Classifier comparison - Wisconsin breast-cancer dataset")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
