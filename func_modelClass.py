"""Model composition utilities for hackathon pipelines."""

from typing import Dict, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_stacking(random_state: int = 42) -> Tuple[StackingClassifier, Dict[str, list]]:
    """Return a stacking classifier and a parameter grid for tuning."""
    base_learners = [
        ("mlp", MLPClassifier(max_iter=1000, learning_rate="adaptive", random_state=random_state, early_stopping=True)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(random_state=random_state)),
        ("dt", DecisionTreeClassifier(max_depth=30, random_state=random_state)),
        ("svc", SVC(gamma=0.01, kernel="rbf", probability=True, decision_function_shape="ovo")),
        ("bayes", GaussianNB()),
    ]
    meta_learner = MLPClassifier(max_iter=1000, learning_rate="adaptive", random_state=random_state, early_stopping=True)
    model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=-1)

    params_2grid = {
        "classification__svc__C": [0.001, 0.1, 10],
        "classification__svc__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "classification__svc__degree": [3, 4, 6],
        "classification__svc__gamma": [0.001, 0.01, 5],
        "classification__rf__n_estimators": [5, 15, 20],
        "classification__rf__max_depth": [20, 30],
        "classification__rf__max_features": ["auto", "sqrt", "log2"],
    }

    return model, params_2grid


def get_voting(random_state: int = 42) -> Tuple[VotingClassifier, Dict[str, list]]:
    """Return a voting classifier and a parameter grid for tuning."""
    clf1 = DecisionTreeClassifier(max_depth=30, random_state=random_state)
    forest = RandomForestClassifier(random_state=random_state)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = SVC(gamma=0.01, kernel="rbf", probability=True, decision_function_shape="ovo")
    clf4 = GaussianNB()
    clf5 = MLPClassifier(max_iter=1000, learning_rate="adaptive", random_state=random_state, early_stopping=True)

    model = VotingClassifier(
        estimators=[("svc", clf3), ("rf", forest), ("dt", clf1), ("gauss", clf4), ("knn", clf2), ("mlp", clf5)],
        voting="hard",
        weights=None,
        n_jobs=-1,
    )

    params_2grid = {
        "classification__svc__C": [0.001, 0.1, 10],
        "classification__svc__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "classification__svc__degree": [3, 4, 6],
        "classification__svc__gamma": [0.001, 0.01, 5],
        "classification__rf__n_estimators": [5, 15, 20],
        "classification__rf__max_depth": [20, 50],
        "classification__rf__max_features": ["auto", "sqrt", "log2"],
    }
    return model, params_2grid


def evaluate_model(model, params_2grid, X, y, xtest: pd.DataFrame, data_test: pd.DataFrame):
    """Fit a model using GridSearchCV with SMOTE balancing and return predictions."""
    model = Pipeline(
        [
            ("undersampling", SMOTE()),
            ("classification", model),
        ]
    )

    grid = GridSearchCV(estimator=model, param_grid=params_2grid, cv=5)
    grid.fit(X, y)

    pred1 = grid.predict(xtest)
    pred1 = pd.concat([data_test.CASENAME, pd.DataFrame(pred1)], axis=1)
    pred1.columns = ["CASENAME", "FLUIDTYPE"]

    return grid.best_estimator_, pred1, grid.best_params_
