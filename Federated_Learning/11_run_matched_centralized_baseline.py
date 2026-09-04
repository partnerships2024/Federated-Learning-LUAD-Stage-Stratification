"""Run a matched centralized Binary BiLSTM baseline for comparison with FedAvg."""

import csv
import importlib.util
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(".")
RESULTS_DIRECTORY = ROOT / "federated_results"
DATABASE_DIRECTORY = ROOT / "databases"
SOURCE_CSV = ROOT / "luad_301bp_patient_level_dataset.csv"
GLOBAL_TEST_KEYS_CSV = RESULTS_DIRECTORY / "global_test_patient_keys.csv"
CENTRALIZED_METRICS_CSV = RESULTS_DIRECTORY / "matched_centralized_test_metrics.csv"
COMPARISON_CSV = RESULTS_DIRECTORY / "centralized_vs_federated_comparison.csv"
RANDOM_SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
EXPECTED_PARAMETER_COUNT = 31777
EXPECTED_TRAIN_COUNT = 2276
EXPECTED_VALIDATION_COUNT = 570
EXPECTED_TEST_COUNT = 712
FEDERATED_ROC_AUC = 0.675605
FEDERATED_F1 = 0.622642
FEDERATED_ACCURACY = 0.606742
HOSPITAL_DATABASES = {
    "Hospital 1": DATABASE_DIRECTORY / "hospital_1.db",
    "Hospital 2": DATABASE_DIRECTORY / "hospital_2.db",
    "Hospital 3": DATABASE_DIRECTORY / "hospital_3.db",
}


def load_step9_module():
    """Load the existing preprocessing/model reference without executing its main."""
    path = ROOT / "09_validate_local_bilstm_training.py"
    spec = importlib.util.spec_from_file_location("step09_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patient_key(row: dict[str, object]) -> tuple[str, str]:
    """Return the stable cohort/patient key."""
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def load_source_rows() -> dict[tuple[str, str], dict[str, object]]:
    """Load source rows only for final held-out test retrieval."""
    with open(SOURCE_CSV, "r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return {(row["COHORT"], row["PATIENT_ID"]): row for row in rows}


def load_test_keys() -> set[tuple[str, str]]:
    """Load global test keys after centralized training is complete."""
    with open(GLOBAL_TEST_KEYS_CSV, "r", encoding="utf-8", newline="") as csv_file:
        return {(row["COHORT"], row["PATIENT_ID"]) for row in csv.DictReader(csv_file)}


def load_hospital_rows(path: Path) -> list[dict[str, object]]:
    """Read local patient rows without modifying the hospital database."""
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM patients").fetchall()]


def roc_auc_score_local(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate ROC-AUC from positive/negative score rankings."""
    positives = probabilities[labels.astype(int) == 1]
    negatives = probabilities[labels.astype(int) == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positives) * len(negatives)))


def pr_auc_score_local(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate average precision for the binary PR-AUC report."""
    positives = int((labels.astype(int) == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    ordered_labels = labels.astype(int)[order]
    cumulative_positives = np.cumsum(ordered_labels == 1)
    ranks = np.arange(1, len(labels) + 1)
    return float((cumulative_positives[ordered_labels == 1] / ranks[ordered_labels == 1]).sum() / positives)


def evaluate_local_model(model, inputs, labels):
    """Evaluate with model predictions and model.evaluate loss only.

    Threshold 0.5 is used for accuracy, precision, recall, and F1. This helper
    is self-contained and does not depend on Step 9 evaluation functions.
    """
    probabilities = model.predict(inputs, verbose=0).reshape(-1)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions detected")
    integer_labels = labels.astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    tp = int(((integer_labels == 1) & (predictions == 1)).sum())
    tn = int(((integer_labels == 0) & (predictions == 0)).sum())
    fp = int(((integer_labels == 0) & (predictions == 1)).sum())
    fn = int(((integer_labels == 1) & (predictions == 0)).sum())
    accuracy = (tp + tn) / len(integer_labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    loss = float(model.evaluate(inputs, labels, verbose=0, return_dict=True)["loss"])
    result = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_local(integer_labels, probabilities),
        "pr_auc": pr_auc_score_local(integer_labels, probabilities),
        "probabilities": probabilities,
    }
    if not all(math.isfinite(float(result[name])) for name in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
        raise ValueError("Non-finite evaluation metric detected")
    return result


def calculate_metrics(model, inputs, labels):
    """Evaluate predictions and return the metrics needed for reporting."""
    result = evaluate_local_model(model, inputs, labels)
    return {name: result[name] for name in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}, result["probabilities"]


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Save a requested lightweight comparison artifact."""
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the matched centralized baseline and final comparison."""
    print("=" * 78)
    print("MATCHED CENTRALIZED BINARY BILSTM BASELINE")
    print("=" * 78)
    step9 = load_step9_module()
    hospital_rows = {hospital: load_hospital_rows(path) for hospital, path in HOSPITAL_DATABASES.items()}

    # Section 1: Reproduce Step 10's deterministic local memberships exactly.
    train_rows_by_hospital = {}
    validation_rows_by_hospital = {}
    all_training_pool_keys = set()
    for hospital, rows in hospital_rows.items():
        train_rows, validation_rows = step9.stratified_train_validation(rows)
        train_rows_by_hospital[hospital] = train_rows
        validation_rows_by_hospital[hospital] = validation_rows
        all_training_pool_keys.update(patient_key(row) for row in rows)
    centralized_train_rows = sum(train_rows_by_hospital.values(), [])
    centralized_validation_rows = sum(validation_rows_by_hospital.values(), [])
    centralized_train_keys = {patient_key(row) for row in centralized_train_rows}
    centralized_validation_keys = {patient_key(row) for row in centralized_validation_rows}
    print(f"Centralized training patients: {len(centralized_train_rows)}")
    print(f"Centralized validation patients: {len(centralized_validation_rows)}")
    print(f"Train/validation overlap: {len(centralized_train_keys & centralized_validation_keys)}")
    if len(centralized_train_rows) != EXPECTED_TRAIN_COUNT or len(centralized_validation_rows) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Centralized train/validation counts do not match Step 10")

    # Section 2: Fit preprocessing on centralized training patients only.
    imputation, vocabulary, means, scales = step9.fit_preprocessing(centralized_train_rows)
    train_inputs, train_labels = step9.build_arrays(centralized_train_rows, vocabulary, imputation, means, scales)
    validation_inputs, validation_labels = step9.build_arrays(centralized_validation_rows, vocabulary, imputation, means, scales)
    print(f"Training-only gene vocabulary size: {len(vocabulary)}")
    print("Preprocessing fit data: centralized training patients only")
    print("Global held-out test used before final evaluation: NO")

    # Section 3: Import TensorFlow and build the unchanged approved architecture.
    try:
        import tensorflow as tf
    except Exception as error:
        print(f"TensorFlow import status: BLOCKED ({type(error).__name__}: {error})")
        print("No centralized model training or result artifacts were created.")
        print("MATCHED CENTRALIZED BASELINE STATUS: FAIL")
        print("Syntax status: PASS")
        print("Execution status: BLOCKED")
        return

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = step9.build_binary_bilstm(tf, len(vocabulary))
    parameter_count = model.count_params()
    print(f"Model name: {model.name}")
    print(f"Model parameter count: {parameter_count}")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PARAMETER_COUNT} parameters, found {parameter_count}")

    # Section 4: Train exactly three epochs; validation is reporting-only.
    label_counts = Counter(train_labels.astype(int))
    total_train = len(train_labels)
    local_class_weights = {
        label: total_train / (2.0 * count)
        for label, count in sorted(label_counts.items())
    }
    validation_history = []
    print("\n--- CENTRALIZED TRAINING ---")
    for epoch in range(1, EPOCHS + 1):
        print(f"EPOCH {epoch}/{EPOCHS}")
        history = model.fit(
            train_inputs,
            train_labels,
            validation_data=(validation_inputs, validation_labels),
            class_weight=local_class_weights,
            epochs=1,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        epoch_metrics, _ = calculate_metrics(model, validation_inputs, validation_labels)
        train_loss = float(history.history["loss"][-1])
        train_accuracy = float(history.history["accuracy"][-1])
        validation_history.append(epoch_metrics)
        print(
            f"EPOCH {epoch}/{EPOCHS} COMPLETE: validation loss={epoch_metrics['loss']:.6f}, "
            f"accuracy={epoch_metrics['accuracy']:.6f}, F1={epoch_metrics['f1']:.6f}, "
            f"ROC-AUC={epoch_metrics['roc_auc']:.6f}, PR-AUC={epoch_metrics['pr_auc']:.6f}"
        )
        if not all(math.isfinite(value) for value in [train_loss, train_accuracy, *[float(epoch_metrics[name]) for name in epoch_metrics]]):
            raise ValueError("Non-finite centralized training or validation metric")

    # Section 5: Retrieve and evaluate the held-out test set only now.
    print("\n--- FINAL HELD-OUT TEST ---")
    test_keys = load_test_keys()
    if len(test_keys) != EXPECTED_TEST_COUNT:
        raise ValueError(f"Expected {EXPECTED_TEST_COUNT} test patients, found {len(test_keys)}")
    source_by_key = load_source_rows()
    test_rows = [source_by_key[key] for key in sorted(test_keys)]
    if test_keys & centralized_train_keys or test_keys & centralized_validation_keys:
        raise ValueError("Global test overlaps centralized train or validation patients")
    test_inputs, test_labels = step9.build_arrays(test_rows, vocabulary, imputation, means, scales)
    final_metrics, final_probabilities = calculate_metrics(model, test_inputs, test_labels)
    if not np.isfinite(final_probabilities).all():
        raise ValueError("Non-finite final predictions detected")
    final_predictions = (final_probabilities >= 0.5).astype(int)
    tn = int(((test_labels == 0) & (final_predictions == 0)).sum())
    fp = int(((test_labels == 0) & (final_predictions == 1)).sum())
    fn = int(((test_labels == 1) & (final_predictions == 0)).sum())
    tp = int(((test_labels == 1) & (final_predictions == 1)).sum())
    print("FINAL GLOBAL TEST RESULTS")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss"]:
        print(f"{name}: {float(final_metrics[name]):.6f}")
    print(f"Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")

    # Section 6: Save only the two new comparison CSVs.
    save_csv(CENTRALIZED_METRICS_CSV, ["metric", "value"], [{"metric": name, "value": final_metrics[name]} for name in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]])
    comparison_rows = [
        {"metric": "accuracy", "centralized": final_metrics["accuracy"], "federated": FEDERATED_ACCURACY, "absolute_difference": abs(float(final_metrics["accuracy"]) - FEDERATED_ACCURACY)},
        {"metric": "f1", "centralized": final_metrics["f1"], "federated": FEDERATED_F1, "absolute_difference": abs(float(final_metrics["f1"]) - FEDERATED_F1)},
        {"metric": "roc_auc", "centralized": final_metrics["roc_auc"], "federated": FEDERATED_ROC_AUC, "absolute_difference": abs(float(final_metrics["roc_auc"]) - FEDERATED_ROC_AUC)},
    ]
    save_csv(COMPARISON_CSV, ["metric", "centralized", "federated", "absolute_difference"], comparison_rows)

    # Section 7: Final integrity checks and required comparison lines.
    print("\n--- MATCHED COMPARISON ---")
    print(f"Centralized ROC-AUC: {float(final_metrics['roc_auc']):.6f}")
    print(f"Federated ROC-AUC: {FEDERATED_ROC_AUC:.6f}")
    print(f"ROC-AUC difference: {abs(float(final_metrics['roc_auc']) - FEDERATED_ROC_AUC):.6f}")
    print(f"Centralized F1: {float(final_metrics['f1']):.6f}")
    print(f"Federated F1: {FEDERATED_F1:.6f}")
    checks_pass = (
        len(centralized_train_rows) == EXPECTED_TRAIN_COUNT
        and len(centralized_validation_rows) == EXPECTED_VALIDATION_COUNT
        and len(test_rows) == EXPECTED_TEST_COUNT
        and len(centralized_train_keys & centralized_validation_keys) == 0
        and len(centralized_train_keys & test_keys) == 0
        and len(centralized_validation_keys & test_keys) == 0
        and parameter_count == EXPECTED_PARAMETER_COUNT
        and np.isfinite(final_probabilities).all()
    )
    print("MATCHED CENTRALIZED BASELINE STATUS: PASS" if checks_pass else "MATCHED CENTRALIZED BASELINE STATUS: FAIL")
    print("Syntax status: PASS")
    print("Execution status: PASS" if checks_pass else "Execution status: FAIL")


if __name__ == "__main__":
    main()
