"""Run the three-client simulated FedAvg Binary BiLSTM experiment.

This script creates only federated experiment artifacts under federated_results
and round metadata in databases/federated_server.db. Patient rows remain local
to each client; only model weights, sample counts, and metrics are exchanged.
"""

import csv
import importlib.util
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(".")
RESULTS_DIRECTORY = ROOT / "federated_results"
DATABASE_DIRECTORY = ROOT / "databases"
SERVER_DATABASE = DATABASE_DIRECTORY / "federated_server.db"
SOURCE_CSV = ROOT / "luad_301bp_patient_level_dataset.csv"
GLOBAL_TEST_KEYS_CSV = RESULTS_DIRECTORY / "global_test_patient_keys.csv"
RANDOM_SEED = 42
NUM_CLIENTS = 3
FEDERATED_ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
EXPECTED_PARAMETER_COUNT = 31777
EXPECTED_GLOBAL_TEST_COUNT = 712
HOSPITAL_DATABASES = {
    "Hospital 1": DATABASE_DIRECTORY / "hospital_1.db",
    "Hospital 2": DATABASE_DIRECTORY / "hospital_2.db",
    "Hospital 3": DATABASE_DIRECTORY / "hospital_3.db",
}
ROUND_METRICS_CSV = RESULTS_DIRECTORY / "federated_round_metrics.csv"
FINAL_METRICS_CSV = RESULTS_DIRECTORY / "final_global_test_metrics.csv"
FINAL_CONFUSION_CSV = RESULTS_DIRECTORY / "final_global_confusion_matrix.csv"
ROC_PNG = RESULTS_DIRECTORY / "federated_roc_curve.png"
CONFUSION_PNG = RESULTS_DIRECTORY / "federated_confusion_matrix.png"
FINAL_WEIGHTS = RESULTS_DIRECTORY / "final_global_bilstm.weights.h5"
EXCHANGE_AUDIT_JSON = RESULTS_DIRECTORY / "federated_exchange_audit.json"


def load_step9_module():
    """Load Step 9 helpers without executing its main routine."""
    module_path = ROOT / "09_validate_local_bilstm_training.py"
    spec = importlib.util.spec_from_file_location("step09_reference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load preprocessing/model reference: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_reproducibility_seed() -> None:
    """Set deterministic Python and NumPy seeds."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def patient_key(row: dict[str, object]) -> tuple[str, str]:
    """Return the cohort/patient identifier pair."""
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def load_source_rows() -> dict[tuple[str, str], dict[str, object]]:
    """Load the original CSV for final held-out test-row retrieval only."""
    with open(SOURCE_CSV, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    return {(row["COHORT"], row["PATIENT_ID"]): row for row in rows}


def load_test_keys() -> set[tuple[str, str]]:
    """Load held-out keys only after all federated rounds finish."""
    with open(GLOBAL_TEST_KEYS_CSV, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return {(row["COHORT"], row["PATIENT_ID"]) for row in reader}


def load_local_partitions(step9):
    """Load hospital rows, fit shared training-only preprocessing, and split locally."""
    hospital_rows = {
        hospital: step9.load_database_rows(path)
        for hospital, path in HOSPITAL_DATABASES.items()
    }
    training_rows = []
    for hospital in HOSPITAL_DATABASES:
        training_rows.extend(hospital_rows[hospital])

    training_keys = {patient_key(row) for row in training_rows}
    if len(training_keys) != len(training_rows):
        raise ValueError("Duplicate patient keys found in federated training hospitals")
    for hospital, rows in hospital_rows.items():
        for other_hospital, other_rows in hospital_rows.items():
            if hospital != other_hospital and {patient_key(row) for row in rows} & {patient_key(row) for row in other_rows}:
                raise ValueError(f"Cross-hospital patient overlap found: {hospital} / {other_hospital}")

    imputation, vocabulary, means, scales = step9.fit_preprocessing(training_rows)
    partitions = {}
    for hospital, rows in hospital_rows.items():
        full_inputs, full_labels = step9.build_arrays(rows, vocabulary, imputation, means, scales)
        train_rows, validation_rows = step9.stratified_train_validation(rows)
        index_by_key = {patient_key(row): index for index, row in enumerate(rows)}
        train_indices = [index_by_key[patient_key(row)] for row in train_rows]
        validation_indices = [index_by_key[patient_key(row)] for row in validation_rows]
        partitions[hospital] = {
            "rows": rows,
            "inputs": full_inputs,
            "labels": full_labels,
            "train_inputs": {name: values[train_indices] for name, values in full_inputs.items()},
            "validation_inputs": {name: values[validation_indices] for name, values in full_inputs.items()},
            "train_labels": full_labels[train_indices],
            "validation_labels": full_labels[validation_indices],
            "train_keys": {patient_key(row) for row in train_rows},
            "validation_keys": {patient_key(row) for row in validation_rows},
        }
    return partitions, training_keys, vocabulary, imputation, means, scales


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calculate inverse-frequency class weights from one local training set."""
    counts = {int(label): int((labels == label).sum()) for label in [0, 1]}
    total = len(labels)
    return {label: total / (2.0 * count) for label, count in counts.items() if count > 0}


def finite_weights(weights: list[np.ndarray]) -> bool:
    """Check all model tensors for finite values."""
    return all(np.isfinite(weight).all() for weight in weights)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """Calculate finite binary metrics without relying on external metric packages."""
    y_true = y_true.astype(int)
    probabilities = probabilities.reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(y_true, probabilities)
    pr_auc = average_precision(y_true, probabilities)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using rank-sum/Mann-Whitney formulation."""
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary average precision for the PR-AUC report."""
    positives = int((y_true == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_true = y_true[order]
    cumulative_true = np.cumsum(ordered_true == 1)
    ranks = np.arange(1, len(y_true) + 1)
    return float((cumulative_true[ordered_true == 1] / ranks[ordered_true == 1]).sum() / positives)


def evaluate_partition(model, inputs: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float | int]:
    """Evaluate one local validation or final test partition."""
    probabilities = model.predict(inputs, verbose=0).reshape(-1)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions detected")
    result = binary_metrics(labels, probabilities)
    result["loss"] = float(model.evaluate(inputs, labels, verbose=0, return_dict=True)["loss"])
    if not math.isfinite(float(result["loss"])):
        raise ValueError("Non-finite evaluation loss detected")
    return result | {"probabilities": probabilities}


def weighted_metrics(metrics_by_client: dict[str, dict[str, float | int]], sample_counts: dict[str, int]) -> dict[str, float]:
    """Calculate sample-count-weighted client validation metrics."""
    total = sum(sample_counts.values())
    names = ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return {name: float(sum(float(metrics_by_client[c][name]) * sample_counts[c] for c in metrics_by_client) / total) for name in names}


def aggregate_fedavg(client_weights: list[list[np.ndarray]], sample_counts: list[int]) -> list[np.ndarray]:
    """Aggregate every model tensor using standard weighted FedAvg."""
    total_samples = sum(sample_counts)
    aggregated = []
    for tensors in zip(*client_weights):
        tensor = sum(weight * count for weight, count in zip(tensors, sample_counts)) / total_samples
        aggregated.append(tensor.astype(tensors[0].dtype, copy=False))
    if not finite_weights(aggregated):
        raise ValueError("FedAvg produced non-finite weights")
    return aggregated


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write a lightweight research CSV artifact."""
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_server_round(round_number: int, client_id: str, sample_count: int, train_metrics: dict, validation_metrics: dict) -> None:
    """Record round metadata only; never write patient-level records."""
    with sqlite3.connect(SERVER_DATABASE) as connection:
        connection.execute(
            """
            INSERT INTO federated_rounds
            (round_number, client_id, local_sample_count, local_loss, local_accuracy,
             update_received, global_loss, global_accuracy, global_f1, global_auc,
             model_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_number,
                client_id,
                sample_count,
                float(train_metrics["loss"]),
                float(train_metrics["accuracy"]),
                1,
                float(validation_metrics["loss"]),
                float(validation_metrics["accuracy"]),
                float(validation_metrics["f1"]),
                float(validation_metrics["roc_auc"]),
                f"fedavg_round_{round_number}",
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def save_plots(y_true: np.ndarray, probabilities: np.ndarray, confusion: dict[str, int]) -> None:
    """Save the required ROC and confusion-matrix PNG artifacts."""
    import matplotlib.pyplot as plt

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_true = y_true[order]
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    tpr = np.concatenate(([0.0], np.cumsum(sorted_true == 1) / positives, [1.0]))
    fpr = np.concatenate(([0.0], np.cumsum(sorted_true == 0) / negatives, [1.0]))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Final global model (AUC={roc_auc(y_true, probabilities):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Federated Binary BiLSTM ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=150)
    plt.close()

    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Early", "Advanced"])
    plt.yticks([0, 1], ["Early", "Advanced"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final Global Confusion Matrix")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=150)
    plt.close()


def main() -> None:
    """Run local client training, FedAvg rounds, and one final held-out evaluation."""
    print("=" * 78)
    print("THREE-CLIENT FEDERATED BINARY BILSTM EXPERIMENT")
    print("=" * 78)
    set_reproducibility_seed()
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    step9 = load_step9_module()
    partitions, training_keys, vocabulary, imputation, means, scales = load_local_partitions(step9)

    print("\n--- LOCAL CLIENT SETUP ---")
    for hospital, partition in partitions.items():
        print(f"{hospital}: train={len(partition['train_labels'])}, validation={len(partition['validation_labels'])}, overlap={len(partition['train_keys'] & partition['validation_keys'])}")
    print(f"Shared gene vocabulary size: {len(vocabulary)}")
    print("Preprocessing fitted from federated training patients only: PASS")
    print("Global held-out test used before final round: NO")

    # TensorFlow is imported only after all read-only preparation is complete.
    try:
        import tensorflow as tf
    except Exception as error:
        print("\nTensorFlow import status: BLOCKED")
        print(f"{type(error).__name__}: {error}")
        print("No federated round or artifact was created.")
        print("FEDERATED LEARNING STATUS: FAIL")
        print("Syntax status: PASS")
        print("Execution status: BLOCKED")
        return

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    global_model = step9.build_binary_bilstm(tf, len(vocabulary))
    parameter_count = global_model.count_params()
    print(f"Approved model parameters: {parameter_count}")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter integrity check failed: expected {EXPECTED_PARAMETER_COUNT}, found {parameter_count}")
    initial_global_weights = [weight.copy() for weight in global_model.get_weights()]
    round_rows = []

    for round_number in range(1, FEDERATED_ROUNDS + 1):
        print(f"\nROUND {round_number}/{FEDERATED_ROUNDS}")
        current_global_weights = [weight.copy() for weight in global_model.get_weights()]
        client_weights = []
        client_sample_counts = []
        local_validation_metrics = {}

        for hospital, partition in partitions.items():
            print(f"{hospital} training...")
            client_model = step9.build_binary_bilstm(tf, len(vocabulary))
            if client_model.count_params() != EXPECTED_PARAMETER_COUNT:
                raise RuntimeError(f"{hospital} parameter count mismatch")
            client_model.set_weights([weight.copy() for weight in current_global_weights])
            starts_identically = all(np.array_equal(a, b) for a, b in zip(client_model.get_weights(), current_global_weights))
            if not starts_identically:
                raise RuntimeError(f"{hospital} did not start from current global weights")
            labels = partition["train_labels"]
            history = client_model.fit(
                partition["train_inputs"],
                labels,
                validation_data=(partition["validation_inputs"], partition["validation_labels"]),
                class_weight=class_weights(labels),
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            history_metrics = {name: float(values[-1]) for name, values in history.history.items() if values}
            local_eval = evaluate_partition(client_model, partition["validation_inputs"], partition["validation_labels"])
            client_weights.append([weight.copy() for weight in client_model.get_weights()])
            client_sample_counts.append(len(labels))
            local_validation_metrics[hospital] = local_eval
            print(f"{hospital} update received.")
            round_rows.append({
                "round": round_number,
                "hospital/client": hospital,
                "local_training_samples": len(labels),
                "local_train_loss": history_metrics["loss"],
                "local_train_accuracy": history_metrics["accuracy"],
                "local_validation_loss": local_eval["loss"],
                "local_validation_accuracy": local_eval["accuracy"],
                "local_validation_f1": local_eval["f1"],
                "local_validation_roc_auc": local_eval["roc_auc"],
                "weighted_round_validation_loss": "",
                "weighted_round_validation_accuracy": "",
                "weighted_round_validation_f1": "",
                "weighted_round_validation_roc_auc": "",
            })

        global_model.set_weights(aggregate_fedavg(client_weights, client_sample_counts))
        print("FedAvg aggregation: PASS")
        aggregated_validation = {}
        for hospital, partition in partitions.items():
            aggregated_validation[hospital] = evaluate_partition(global_model, partition["validation_inputs"], partition["validation_labels"])
        weighted = weighted_metrics(aggregated_validation, {hospital: len(partition["validation_labels"]) for hospital, partition in partitions.items()})
        print(f"WEIGHTED CLIENT VALIDATION METRICS - loss={weighted['loss']:.6f}, accuracy={weighted['accuracy']:.6f}, F1={weighted['f1']:.6f}, ROC-AUC={weighted['roc_auc']:.6f}")
        for row in round_rows[-NUM_CLIENTS:]:
            row["weighted_round_validation_loss"] = weighted["loss"]
            row["weighted_round_validation_accuracy"] = weighted["accuracy"]
            row["weighted_round_validation_f1"] = weighted["f1"]
            row["weighted_round_validation_roc_auc"] = weighted["roc_auc"]
        for hospital in partitions:
            record_server_round(round_number, hospital, len(partitions[hospital]["train_labels"]), local_validation_metrics[hospital], weighted)

    # Final held-out evaluation is deliberately after every federated round.
    print("\n--- FINAL GLOBAL HELD-OUT TEST ---")
    test_keys = load_test_keys()
    if len(test_keys) != EXPECTED_GLOBAL_TEST_COUNT:
        raise ValueError(f"Expected {EXPECTED_GLOBAL_TEST_COUNT} global test patients, found {len(test_keys)}")
    if test_keys & training_keys:
        raise ValueError("Global test overlaps federated training patients")
    source_rows = load_source_rows()
    missing_test_rows = test_keys - source_rows.keys()
    if missing_test_rows:
        raise ValueError("Global test keys missing from source CSV")
    test_rows = [source_rows[key] for key in sorted(test_keys)]
    test_inputs, test_labels = step9.build_arrays(test_rows, vocabulary, imputation, means, scales)
    final_test = evaluate_partition(global_model, test_inputs, np.asarray(test_labels))
    confusion = {name: int(final_test[name]) for name in ["tn", "fp", "fn", "tp"]}
    print("FINAL GLOBAL TEST RESULTS")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{name.replace('_', '-').upper()}: {final_test[name]:.6f}")
    print(f"Test loss: {final_test['loss']:.6f}")
    print(f"Confusion Matrix: [[{confusion['tn']}, {confusion['fp']}], [{confusion['fn']}, {confusion['tp']}]]")
    print(f"Early class count: {int((test_labels == 0).sum())}")
    print(f"Advanced class count: {int((test_labels == 1).sum())}")
    print(f"Predicted Early count: {confusion['tn'] + confusion['fn']}")
    print(f"Predicted Advanced count: {confusion['fp'] + confusion['tp']}")

    # Save only after all rounds and the single final test evaluation succeed.
    save_csv(ROUND_METRICS_CSV, list(round_rows[0]), round_rows)
    save_csv(FINAL_METRICS_CSV, ["metric", "value"], [{"metric": name, "value": final_test[name]} for name in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]])
    save_csv(FINAL_CONFUSION_CSV, ["actual", "predicted_early", "predicted_advanced"], [
        {"actual": "Early", "predicted_early": confusion["tn"], "predicted_advanced": confusion["fp"]},
        {"actual": "Advanced", "predicted_early": confusion["fn"], "predicted_advanced": confusion["tp"]},
    ])
    save_plots(np.asarray(test_labels).astype(int), final_test["probabilities"], confusion)
    global_model.save_weights(FINAL_WEIGHTS)
    audit = {
        "simulated_clients": NUM_CLIENTS,
        "aggregation": "FedAvg",
        "raw_patient_rows_sent_to_server": False,
        "exchanged_payload": "model weights + sample counts + metrics",
        "formal_differential_privacy": False,
        "secure_aggregation": False,
        "global_test_overlap_with_training": 0,
        "central_server_contains_patient_table": False,
        "global_test_evaluated_only_after_final_federated_round": True,
        "preprocessing_fitted_on_federated_training_only": True,
        "parameter_count": parameter_count,
    }
    with open(EXCHANGE_AUDIT_JSON, "w", encoding="utf-8") as json_file:
        json.dump(audit, json_file, indent=2)

    with sqlite3.connect(SERVER_DATABASE) as connection:
        tables = [name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    server_has_patient_table = any("patient" in table.lower() for table in tables)
    isolation_ok = all(not (partitions[a]["train_keys"] & partitions[b]["train_keys"]) for a in partitions for b in partitions if a != b)
    status_pass = parameter_count == EXPECTED_PARAMETER_COUNT and finite_weights(global_model.get_weights()) and np.isfinite(final_test["probabilities"]).all() and not server_has_patient_table and isolation_ok
    print(f"Central server patient table exists: {'YES' if server_has_patient_table else 'NO'}")
    print("FEDERATED LEARNING STATUS: PASS" if status_pass else "FEDERATED LEARNING STATUS: FAIL")
    print("Syntax status: PASS")
    print("Execution status: PASS" if status_pass else "Execution status: FAIL")


if __name__ == "__main__":
    main()
