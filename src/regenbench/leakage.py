"""Random-split versus group-split diagnostic. Not a biological result."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regenbench.benchmark import evaluate
from regenbench.data import Dataset


def donor_tag_fixture(*, n_donors=20, replicates=8, seed=0) -> Dataset:
    """Label is constant per donor. The only useful feature is that donor's own tag.

    A random row split sees the same tag in train and test. A donor holdout never
    does. The tag is an arbitrary id, not a measurement.
    """
    if n_donors < 10 or n_donors % 2 or replicates < 2:
        raise ValueError("use an even donor count of at least 10 and at least 2 replicates")
    rng = np.random.default_rng(seed)
    tags = rng.permutation(n_donors)
    rows, xs, ys, groups = [], [], [], []
    for donor in range(n_donors):
        label = "1" if donor < n_donors // 2 else "0"
        one_hot = np.zeros(n_donors)
        one_hot[int(tags[donor])] = 1.0
        for replicate in range(replicates):
            rows.append({
                "sample_id": f"d{donor}-r{replicate}",
                "label": label,
                "donor_id": f"d{donor}",
            })
            xs.append([*one_hot, float(rng.normal())])
            ys.append(label)
            groups.append(f"d{donor}")
    features = [f"f_tag_{i}" for i in range(n_donors)] + ["f_noise"]
    return Dataset(
        rows=rows,
        features=features,
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys),
        groups=np.asarray(groups),
        sha256="synthetic-donor-tag-fixture",
        group_columns=["donor_id"],
    )


def _logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )


def random_row_accuracy(data: Dataset, *, folds=5, seed=0) -> float:
    """Stratified over rows. This is the leaky comparison, not the headline split."""
    pred = np.empty(len(data.y), dtype=object)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(data.x, data.y):
        model = _logistic().fit(data.x[train], data.y[train])
        pred[test] = model.predict(data.x[test])
    return float(accuracy_score(data.y, pred))


def leakage_gap(data: Dataset | None = None, *, folds=5, seed=0) -> dict:
    data = donor_tag_fixture(seed=seed) if data is None else data
    grouped = evaluate(data, folds=folds, seed=seed, bootstrap_draws=100)
    group_accuracy = grouped[0]["models"]["logistic"]["accuracy"]
    leaked = random_row_accuracy(data, folds=folds, seed=seed)
    return {
        "kind": "leakage diagnostic on a synthetic donor-tag fixture",
        "not_a_biological_result": True,
        "random_row_logistic_accuracy": leaked,
        "group_holdout_logistic_accuracy": group_accuracy,
        "gap_random_minus_group": leaked - group_accuracy,
        "warning": "Use the group holdout as the estimand. The random-row number is only a leak detector.",
    }
