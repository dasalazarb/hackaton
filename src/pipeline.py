"""End-to-end pipeline for fluid type prediction and RF estimation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

TRAIN_FEATURES_FLUID = [
    "PRESS",
    "TEMP",
    "OILGRAV",
    "SOLGOR",
    "Visco",
    "Psat",
    "Bo",
]

RECOVERY_FEATURES = [
    "Depth",
    "Area",
    "RESTHICK",
    "PRESS",
    "TEMP",
    "OILGRAV",
    "SOLGOR",
    "Visco",
    "Psat",
    "Bo",
    "POROSITY",
    "NTG",
    "PERM",
    "CONWATER",
    "PERM.AQUIFER",
    "OOIP",
    "RF",
    "RFmax",
    "GOR",
    "RELPERM.RESSAT.Kro",
    "RELPERM.ENDPOINT.Kro",
    "RELPERM.ENDPOINT.Krw",
    "Field.OILRATE",
    "Field.WATRATE",
    "Field.GASRATE",
]

DROP_COLUMNS_BY_INDEX = [31, 32]


def load_datasets(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test datasets from disk."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns by positional index if they exist in the DataFrame."""
    columns_to_drop: Iterable[str] = [
        df.columns[idx] for idx in DROP_COLUMNS_BY_INDEX if idx < len(df.columns)
    ]
    return df.drop(columns=columns_to_drop)


def prepare_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the training data and remove rows without a target."""
    cleaned = drop_unused_columns(train_df)
    cleaned = cleaned.dropna(axis=0, subset=["FLUIDTYPE"]).reset_index(drop=True)
    return cleaned


def build_fluidtype_model(random_state: int = 42) -> ImbPipeline:
    """Create the pipeline used to classify the fluid type."""
    model = ImbPipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=random_state)),
            ("sampler", SMOTE(random_state=random_state)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300, max_depth=20, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )
    return model


def train_fluidtype_classifier(train_df: pd.DataFrame) -> Tuple[ImbPipeline, pd.DataFrame]:
    """Train the fluid type classifier and return the trained model and report."""
    feature_df = train_df[TRAIN_FEATURES_FLUID]
    target = train_df["FLUIDTYPE"].astype(str)

    X_train, X_valid, y_train, y_valid = train_test_split(
        feature_df, target, test_size=0.2, random_state=42, stratify=target
    )

    model = build_fluidtype_model()
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_valid)
    report = classification_report(y_valid, val_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return model, report_df


def predict_fluidtype(model: ImbPipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """Generate fluid type predictions using a trained model."""
    fluid_features = test_df[TRAIN_FEATURES_FLUID]
    predictions = model.predict(fluid_features)
    return pd.DataFrame({"CASENAME": test_df["CASENAME"], "FLUIDTYPE": predictions})


def impute_with_kmeans(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Estimate the recovery factor (RF) for the test set using KMeans clustering."""
    train_features = train_df[RECOVERY_FEATURES].copy()
    test_features = test_df[RECOVERY_FEATURES].copy()

    imputer = IterativeImputer(max_iter=500, random_state=random_state)
    train_imputed = imputer.fit_transform(train_features)
    test_imputed = imputer.transform(test_features)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(train_scaled)

    train_labels = kmeans.labels_
    test_labels = kmeans.predict(test_scaled)

    train_with_labels = pd.DataFrame(train_features)
    train_with_labels["cluster"] = train_labels

    cluster_mean_rfmax = train_with_labels.groupby("cluster")["RFmax"].mean()
    estimated_rf = cluster_mean_rfmax.loc[test_labels].reset_index(drop=True)

    return pd.DataFrame({"CASENAME": test_df["CASENAME"], "RF": estimated_rf})


def build_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    """Run the full pipeline and persist prediction artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model, report_df = train_fluidtype_classifier(train_df)
    fluid_predictions = predict_fluidtype(model, test_df)

    rf_estimates = impute_with_kmeans(train_df, test_df)
    combined = fluid_predictions.merge(rf_estimates, on="CASENAME")

    fluid_predictions.to_csv(output_dir / "fluidtype_predictions.csv", index=False)
    rf_estimates.to_csv(output_dir / "rf_estimates.csv", index=False)
    combined.to_csv(output_dir / "submission.csv", index=False)
    report_df.to_csv(output_dir / "validation_report.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("adv_train_data.csv"), help="Path to the training CSV file.")
    parser.add_argument("--test", type=Path, default=Path("adv_test_data.csv"), help="Path to the test CSV file.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory where prediction files will be stored."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = load_datasets(args.train, args.test)
    train_df = prepare_training_data(train_df)
    test_df = drop_unused_columns(test_df)

    build_outputs(train_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()
