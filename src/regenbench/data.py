"""Strict CSV loading and connected grouping across experimental units."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    rows: list[dict[str, str]]
    features: list[str]
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sha256: str
    group_columns: list[str]
    task: str = "classification"
    target_column: str = "label"


def connected_groups(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    """Rows sharing ANY selected unit belong to the same connected component.

    Tuple concatenation is insufficient: (donor A, plate 1) and (donor A,
    plate 2) must stay together when blocking both donor and plate.
    """
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("provide distinct grouping columns")
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for column in columns:
            value = row.get(column, "").strip()
            if not value:
                raise ValueError(f"row {i + 2}: missing grouping value {column}")
            key = (column, value)
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    return np.array([f"group-{root(i):06d}" for i in range(len(rows))])


def load_table(path: str | Path, group_columns: list[str], *, task="classification", target_column=None) -> Dataset:
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    target_column = target_column or ("label" if task == "classification" else "target")
    if target_column.startswith("f_") or target_column in group_columns:
        raise ValueError("target must be separate from features and grouping columns")
    if any(column.startswith("f_") for column in group_columns):
        raise ValueError("grouping columns cannot also be features")
    path = Path(path)
    raw = path.read_bytes()
    # Parse the same snapshot recorded by the report's input hash.
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("duplicate CSV columns")
        required = {"sample_id", target_column, *group_columns}
        if not required.issubset(header):
            raise ValueError(f"missing columns: {sorted(required - set(header))}")
        features = [c for c in header if c.startswith("f_")]
        if not features:
            raise ValueError("features must have an f_ prefix; metadata is never inferred as a feature")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"row {line}: ragged CSV")
            rows.append({k: v.strip() for k, v in row.items()})
    if not rows:
        raise ValueError("empty dataset")
    for column in ["sample_id", target_column, *group_columns]:
        if any(not r[column] for r in rows):
            raise ValueError(f"blank {column}")
    sample_ids = [r["sample_id"] for r in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id must be unique (duplicate samples leak)")
    if "image_sha256" in header:
        checksums = [r["image_sha256"] for r in rows]
        if any(re.fullmatch(r"[0-9a-fA-F]{64}", value) is None for value in checksums):
            raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
        if len({value.lower() for value in checksums}) != len(checksums):
            raise ValueError("image_sha256 must be unique (duplicate source images leak)")
    try:
        x = np.array([[float(row[c]) for c in features] for row in rows])
    except ValueError as exc:
        raise ValueError("all f_ columns must be numeric") from exc
    if not np.isfinite(x).all():
        raise ValueError("features must be finite; handle missing values explicitly upstream")
    if task == "regression":
        try:
            y = np.array([float(r[target_column]) for r in rows])
        except ValueError as exc:
            raise ValueError("regression targets must be numeric") from exc
        if not np.isfinite(y).all():
            raise ValueError("regression targets must be finite")
    else:
        y = np.array([r[target_column] for r in rows])
    if task == "classification" and len(set(y)) < 2:
        raise ValueError("classification requires at least two labels")
    return Dataset(rows, features, x, y, connected_groups(rows, group_columns),
                   hashlib.sha256(raw).hexdigest(), group_columns, task, target_column)


def extract_feature_importances(model, feature_names: list[str]) -> dict[str, float] | None:
    """Extract per-feature importance or absolute standardized coefficients from a fitted model."""
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            importances = np.abs(coef)
        elif coef.ndim == 2:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            raise ValueError("model coefficients must be a one- or two-dimensional array")
    if importances is None:
        return None
    if (importances.shape != (len(feature_names),) or not np.isfinite(importances).all()
            or len(set(feature_names)) != len(feature_names)):
        raise ValueError("feature contributions must be finite and aligned with distinct feature names")
    return dict(zip(feature_names, map(float, importances)))


def feature_importance_method(model) -> str | None:
    """Describe the quantity; different estimators' values are not comparable."""
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        return "training_impurity_decrease"
    if hasattr(estimator, "coef_"):
        return "absolute_standardized_coefficient_mean_across_classes"
    return None


def unblocked_overlaps(data: Dataset, partitions) -> dict[str, list[int]]:
    """Report known experimental units omitted from the selected holdout."""
    result = {}
    for column in ["donor_id", "batch_id", "plate_id", "group_id", "acquisition_day", "source_well"]:
        if column not in data.rows[0] or column in data.group_columns:
            continue
        result[column] = []
        for train, test in partitions:
            left = {data.rows[i][column] for i in train} - {""}
            right = {data.rows[i][column] for i in test} - {""}
            result[column].append(len(left & right))
    return result
