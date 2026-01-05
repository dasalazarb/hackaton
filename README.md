# Reservoir Analytics Pipeline

This repository contains a cleaned-up, end-to-end workflow for predicting reservoir `FLUIDTYPE` and estimating recovery factor (`RF`) values from the provided training and test CSV files.

## Repository layout

- `src/pipeline.py` – main entry point that prepares data, trains the classifier, estimates RF with clustering, and writes prediction artifacts.
- `func_modelClass.py` – reusable model definitions (stacking, voting, and grid-search helper) retained from the original exploratory work.
- `adv_train_data.csv`, `adv_test_data.csv` – primary training and inference datasets (smaller competition split).
- `adv_train_prod_data.csv`, `adv_test_prod_data.csv` – production-scale datasets with the same schema as above.
- `variables.txt`, `web.txt` – supporting reference material from the original project.

Legacy exploratory notebooks/scripts (`Diego.py`, `Felipe.py`, `Xiomara.py`) remain for historical context but are superseded by the streamlined pipeline in `src/pipeline.py`.

## End-to-end pipeline

1. **Load data** – read the training and test CSV files (defaults are the smaller `adv_*` datasets).
2. **Clean columns** – drop the two unused columns in positions 31 and 32, and remove rows without a `FLUIDTYPE` label in the training data.
3. **Train classifier** – build an imputation + SMOTE-balanced `RandomForestClassifier` pipeline on the fluid-property feature subset, reserving a validation split to produce a classification report.
4. **Predict fluid type** – run the trained classifier on the cleaned test set to generate `FLUIDTYPE` predictions per `CASENAME`.
5. **Estimate RF** – impute missing values on a wider set of reservoir features, scale them, and assign clusters via KMeans to transfer cluster-average `RFmax` values into an estimated `RF` column for the test set.
6. **Persist artifacts** – write `fluidtype_predictions.csv`, `rf_estimates.csv`, a merged `submission.csv`, and the validation metrics (`validation_report.csv`) into the chosen output directory.

## Quick start

```bash
# Activate the pipeline on the default data split
python -m src.pipeline --train adv_train_data.csv --test adv_test_data.csv --output-dir outputs
```

Outputs will be created under `outputs/` by default. Swap `adv_train_prod_data.csv` and `adv_test_prod_data.csv` into the `--train`/`--test` flags when you are ready to run on the production-sized files.

## Development notes

- The pipeline relies on scikit-learn and imbalanced-learn. Install dependencies with `pip install -r requirements.txt` if you add one, or manually install `scikit-learn` and `imbalanced-learn` in your environment.
- The helper functions in `func_modelClass.py` remain available if you wish to experiment with stacking/voting ensembles and grid search beyond the simplified RandomForest baseline used in `src/pipeline.py`.
