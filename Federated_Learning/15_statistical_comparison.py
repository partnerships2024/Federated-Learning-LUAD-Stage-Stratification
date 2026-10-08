"""
Paired statistical comparison of matched centralized and federated BiLSTM models.

Inputs
------
federated_results/matched_centralized_test_predictions_for_stats.csv
federated_results/final_global_test_predictions_for_stats.csv

Outputs
-------
federated_results/centralized_vs_federated_statistical_comparison.csv
federated_results/delong_roc_auc_comparison.csv

Method
------
- Exact one-to-one matching by (COHORT, PATIENT_ID)
- 10,000 stratified paired patient-level bootstrap resamples
- Fixed random seed = 42
- 95% percentile confidence intervals
- Paired bootstrap differences: centralized - federated
- Correlated DeLong test for ROC-AUC
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(".")
RESULTS_DIR = ROOT / "federated_results"

CENTRALIZED_PREDICTIONS = (
    RESULTS_DIR / "matched_centralized_test_predictions_for_stats.csv"
)
FEDERATED_PREDICTIONS = (
    RESULTS_DIR / "final_global_test_predictions_for_stats.csv"
)

STATISTICAL_COMPARISON_CSV = (
    RESULTS_DIR / "centralized_vs_federated_statistical_comparison.csv"
)
DELONG_CSV = RESULTS_DIR / "delong_roc_auc_comparison.csv"

BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 42
THRESHOLD = 0.5
EXPECTED_TEST_PATIENTS = 712


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """Calculate binary classification metrics at the fixed 0.5 threshold."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    return {
        "accuracy": (tp + tn) / len(y_true),
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "pr_auc": average_precision_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_paired_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and strictly pair the two models' predictions by patient key."""
    centralized = pd.read_csv(CENTRALIZED_PREDICTIONS)
    federated = pd.read_csv(FEDERATED_PREDICTIONS)

    if len(centralized) != EXPECTED_TEST_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_PATIENTS} centralized rows, "
            f"found {len(centralized)}."
        )
    if len(federated) != EXPECTED_TEST_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_PATIENTS} federated rows, "
            f"found {len(federated)}."
        )

    if centralized[["COHORT", "PATIENT_ID"]].duplicated().any():
        raise ValueError("Duplicate centralized patient keys detected.")
    if federated[["COHORT", "PATIENT_ID"]].duplicated().any():
        raise ValueError("Duplicate federated patient keys detected.")

    paired = centralized.merge(
        federated,
        on=["COHORT", "PATIENT_ID"],
        how="inner",
        suffixes=("_CENTRALIZED", "_FEDERATED"),
        validate="one_to_one",
    )

    if len(paired) != EXPECTED_TEST_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_PATIENTS} paired rows, "
            f"found {len(paired)}."
        )

    y_c = paired["TARGET_STAGE_BINARY_CENTRALIZED"].to_numpy(dtype=int)
    y_f = paired["TARGET_STAGE_BINARY_FEDERATED"].to_numpy(dtype=int)

    if not np.array_equal(y_c, y_f):
        raise ValueError(
            "True labels differ between centralized and federated files."
        )

    centralized_probabilities = paired[
        "CENTRALIZED_PROBABILITY"
    ].to_numpy(dtype=float)

    federated_probabilities = paired[
        "FEDERATED_PROBABILITY"
    ].to_numpy(dtype=float)

    if not np.isfinite(centralized_probabilities).all():
        raise ValueError("Non-finite centralized probabilities detected.")
    if not np.isfinite(federated_probabilities).all():
        raise ValueError("Non-finite federated probabilities detected.")

    return y_c, centralized_probabilities, federated_probabilities


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    centralized_probabilities: np.ndarray,
    federated_probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Compute model-specific 95% CIs and paired differences using
    10,000 stratified patient-level bootstrap resamples.
    """
    metric_names = [
        "accuracy",
        "f1",
        "roc_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "pr_auc",
    ]

    centralized_point = calculate_metrics(
        y_true,
        centralized_probabilities,
    )
    federated_point = calculate_metrics(
        y_true,
        federated_probabilities,
    )

    class_0_indices = np.where(y_true == 0)[0]
    class_1_indices = np.where(y_true == 1)[0]

    rng = np.random.default_rng(RANDOM_SEED)

    centralized_boot = {
        metric: np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for metric in metric_names
    }
    federated_boot = {
        metric: np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for metric in metric_names
    }
    difference_boot = {
        metric: np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for metric in metric_names
    }

    for bootstrap_index in range(BOOTSTRAP_RESAMPLES):
        sampled_indices = np.concatenate(
            [
                rng.choice(
                    class_0_indices,
                    size=len(class_0_indices),
                    replace=True,
                ),
                rng.choice(
                    class_1_indices,
                    size=len(class_1_indices),
                    replace=True,
                ),
            ]
        )

        sampled_labels = y_true[sampled_indices]

        centralized_metrics = calculate_metrics(
            sampled_labels,
            centralized_probabilities[sampled_indices],
        )
        federated_metrics = calculate_metrics(
            sampled_labels,
            federated_probabilities[sampled_indices],
        )

        for metric in metric_names:
            centralized_boot[metric][bootstrap_index] = (
                centralized_metrics[metric]
            )
            federated_boot[metric][bootstrap_index] = (
                federated_metrics[metric]
            )
            difference_boot[metric][bootstrap_index] = (
                centralized_metrics[metric]
                - federated_metrics[metric]
            )

    rows = []

    for metric in metric_names:
        centralized_ci_low, centralized_ci_high = np.percentile(
            centralized_boot[metric],
            [2.5, 97.5],
        )
        federated_ci_low, federated_ci_high = np.percentile(
            federated_boot[metric],
            [2.5, 97.5],
        )
        difference_ci_low, difference_ci_high = np.percentile(
            difference_boot[metric],
            [2.5, 97.5],
        )

        less_or_equal_zero = np.mean(
            difference_boot[metric] <= 0
        )
        greater_or_equal_zero = np.mean(
            difference_boot[metric] >= 0
        )

        paired_bootstrap_p = min(
            1.0,
            2.0 * min(
                less_or_equal_zero,
                greater_or_equal_zero,
            ),
        )

        rows.append(
            {
                "metric": metric,
                "centralized": centralized_point[metric],
                "centralized_ci_low": centralized_ci_low,
                "centralized_ci_high": centralized_ci_high,
                "federated": federated_point[metric],
                "federated_ci_low": federated_ci_low,
                "federated_ci_high": federated_ci_high,
                "difference_centralized_minus_federated":
                    centralized_point[metric]
                    - federated_point[metric],
                "difference_ci_low": difference_ci_low,
                "difference_ci_high": difference_ci_high,
                "paired_bootstrap_p": paired_bootstrap_p,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "random_seed": RANDOM_SEED,
                "bootstrap_method":
                    "stratified paired patient-level percentile bootstrap",
            }
        )

    return pd.DataFrame(rows)


def correlated_delong_test(
    y_true: np.ndarray,
    centralized_probabilities: np.ndarray,
    federated_probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Perform the correlated DeLong test for two ROC-AUCs evaluated
    on the same patients.
    """
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError(
            "DeLong test requires both positive and negative cases."
        )

    score_vectors = [
        centralized_probabilities,
        federated_probabilities,
    ]

    v10_rows = []
    v01_rows = []
    aucs = []

    for scores in score_vectors:
        positive_scores = scores[positive_indices]
        negative_scores = scores[negative_indices]

        pairwise = (
            (positive_scores[:, None] > negative_scores[None, :])
            .astype(float)
            + 0.5
            * (
                positive_scores[:, None]
                == negative_scores[None, :]
            )
        )

        v10 = pairwise.mean(axis=1)
        v01 = pairwise.mean(axis=0)

        v10_rows.append(v10)
        v01_rows.append(v01)
        aucs.append(float(v10.mean()))

    v10_matrix = np.vstack(v10_rows)
    v01_matrix = np.vstack(v01_rows)

    covariance = (
        np.cov(v10_matrix, bias=False) / len(positive_indices)
        + np.cov(v01_matrix, bias=False) / len(negative_indices)
    )

    auc_difference = aucs[0] - aucs[1]

    variance_difference = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2.0 * covariance[0, 1]
    )

    if variance_difference <= 0:
        raise ValueError(
            "Non-positive DeLong variance for AUC difference."
        )

    standard_error = math.sqrt(variance_difference)
    z_statistic = auc_difference / standard_error
    p_value = 2.0 * norm.sf(abs(z_statistic))

    ci_low = auc_difference - 1.96 * standard_error
    ci_high = auc_difference + 1.96 * standard_error

    return pd.DataFrame(
        [
            {
                "centralized_auc": aucs[0],
                "federated_auc": aucs[1],
                "auc_difference": auc_difference,
                "standard_error_difference": standard_error,
                "z": z_statistic,
                "p_value_two_sided": p_value,
                "difference_ci_low_normal_approx": ci_low,
                "difference_ci_high_normal_approx": ci_high,
            }
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("MATCHED CENTRALIZED VS FEDERATED STATISTICAL COMPARISON")
    print("=" * 78)

    (
        y_true,
        centralized_probabilities,
        federated_probabilities,
    ) = load_paired_predictions()

    centralized_metrics = calculate_metrics(
        y_true,
        centralized_probabilities,
    )
    federated_metrics = calculate_metrics(
        y_true,
        federated_probabilities,
    )

    print(f"Paired held-out patients: {len(y_true)}")
    print(
        "Label counts:",
        {
            "Early": int((y_true == 0).sum()),
            "Advanced": int((y_true == 1).sum()),
        },
    )

    print(
        "Centralized confusion matrix:",
        [
            [
                centralized_metrics["tn"],
                centralized_metrics["fp"],
            ],
            [
                centralized_metrics["fn"],
                centralized_metrics["tp"],
            ],
        ],
    )

    print(
        "Federated confusion matrix:",
        [
            [
                federated_metrics["tn"],
                federated_metrics["fp"],
            ],
            [
                federated_metrics["fn"],
                federated_metrics["tp"],
            ],
        ],
    )

    bootstrap_results = paired_stratified_bootstrap(
        y_true,
        centralized_probabilities,
        federated_probabilities,
    )

    delong_results = correlated_delong_test(
        y_true,
        centralized_probabilities,
        federated_probabilities,
    )

    bootstrap_results.to_csv(
        STATISTICAL_COMPARISON_CSV,
        index=False,
    )

    delong_results.to_csv(
        DELONG_CSV,
        index=False,
    )

    print("\n--- ROC-AUC COMPARISON ---")
    print(
        "Centralized ROC-AUC:",
        f"{delong_results.loc[0, 'centralized_auc']:.6f}",
    )
    print(
        "Federated ROC-AUC:",
        f"{delong_results.loc[0, 'federated_auc']:.6f}",
    )
    print(
        "ROC-AUC difference:",
        f"{delong_results.loc[0, 'auc_difference']:.6f}",
    )
    print(
        "DeLong p-value:",
        f"{delong_results.loc[0, 'p_value_two_sided']:.12g}",
    )

    print("\n--- OUTPUT FILES ---")
    print(STATISTICAL_COMPARISON_CSV)
    print(DELONG_CSV)
    print("\nSTATISTICAL COMPARISON STATUS: PASS")


if __name__ == "__main__":
    main()
