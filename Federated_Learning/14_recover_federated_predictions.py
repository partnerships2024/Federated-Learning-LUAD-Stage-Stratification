"""
Recover the exact 712-patient federated BiLSTM test probabilities from the
saved final federated model weights.

This script DOES NOT retrain the federated model.

It reconstructs the preprocessing used by Step 10 from all 2,846 simulated
hospital patients, loads the saved final global weights, evaluates the original
712 held-out test patients, verifies the published metrics, and only then saves
the patient-level probability table required for statistical testing.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(".")
RESULTS_DIR = ROOT / "federated_results"
DATABASE_DIR = ROOT / "databases"

STEP9_PATH = ROOT / "09_validate_local_bilstm_training.py"
STEP10_PATH = ROOT / "10_run_federated_bilstm.py"

SOURCE_CSV = ROOT / "luad_301bp_patient_level_dataset.csv"
TEST_KEYS_CSV = RESULTS_DIR / "global_test_patient_keys.csv"
WEIGHTS_PATH = RESULTS_DIR / "final_global_bilstm.weights.h5"

OUTPUT_CSV = RESULTS_DIR / "final_global_test_predictions_for_stats.csv"

HOSPITAL_DATABASES = [
    DATABASE_DIR / "hospital_1.db",
    DATABASE_DIR / "hospital_2.db",
    DATABASE_DIR / "hospital_3.db",
]

EXPECTED_TRAINING_POOL = 2846
EXPECTED_TEST = 712
EXPECTED_PARAMETERS = 31777

EXPECTED_ACCURACY = 0.606742
EXPECTED_F1 = 0.622642
EXPECTED_ROC_AUC = 0.675605
TOLERANCE = 5e-6


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patient_key(row):
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def load_source_rows():
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["COHORT"], row["PATIENT_ID"]): row
        for row in rows
    }


def load_test_keys():
    with TEST_KEYS_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["COHORT"], row["PATIENT_ID"])
            for row in csv.DictReader(handle)
        }


def close_enough(value: float, expected: float) -> bool:
    return abs(float(value) - expected) <= TOLERANCE


def main():
    print("=" * 78)
    print("RECOVER FEDERATED BILSTM TEST PREDICTIONS")
    print("=" * 78)

    required = [
        STEP9_PATH,
        STEP10_PATH,
        SOURCE_CSV,
        TEST_KEYS_CSV,
        WEIGHTS_PATH,
        *HOSPITAL_DATABASES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(missing)
        )

    step9 = load_module(STEP9_PATH, "step9_reference")
    step10 = load_module(STEP10_PATH, "step10_reference")

    # Reproduce Step 10 exactly:
    # preprocessing was fitted on ALL 2,846 rows across Hospitals 1-3
    # before the local train/validation split.
    training_rows = []
    for database_path in HOSPITAL_DATABASES:
        rows = step9.load_database_rows(database_path)
        training_rows.extend(rows)

    if len(training_rows) != EXPECTED_TRAINING_POOL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_POOL} federated training-pool patients, "
            f"found {len(training_rows)}."
        )

    unique_training_keys = {patient_key(row) for row in training_rows}
    if len(unique_training_keys) != EXPECTED_TRAINING_POOL:
        raise RuntimeError("Duplicate patient keys found in federated training pool.")

    imputation, vocabulary, means, scales = step9.fit_preprocessing(
        training_rows
    )

    print("Federated preprocessing rows:", len(training_rows))
    print("Gene vocabulary size:", len(vocabulary))

    try:
        import tensorflow as tf
    except Exception as error:
        raise RuntimeError(
            "TensorFlow is required. Activate the same .venv used for the "
            "original LUAD federated experiment."
        ) from error

    tf.keras.utils.set_random_seed(step10.RANDOM_SEED)

    model = step9.build_binary_bilstm(tf, len(vocabulary))

    if model.count_params() != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETERS} parameters, "
            f"found {model.count_params()}."
        )

    model.load_weights(WEIGHTS_PATH)

    if not all(np.isfinite(weight).all() for weight in model.get_weights()):
        raise RuntimeError("Loaded federated model contains non-finite weights.")

    test_keys = load_test_keys()
    if len(test_keys) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} held-out test patients, "
            f"found {len(test_keys)}."
        )

    if test_keys & unique_training_keys:
        raise RuntimeError(
            "Held-out test patients overlap the federated training pool."
        )

    source_by_key = load_source_rows()
    ordered_keys = sorted(test_keys)

    missing_source = [
        key for key in ordered_keys
        if key not in source_by_key
    ]
    if missing_source:
        raise RuntimeError(
            f"{len(missing_source)} held-out patient keys are missing "
            "from the source CSV."
        )

    test_rows = [
        source_by_key[key]
        for key in ordered_keys
    ]

    test_inputs, test_labels = step9.build_arrays(
        test_rows,
        vocabulary,
        imputation,
        means,
        scales,
    )

    probabilities = model.predict(
        test_inputs,
        batch_size=16,
        verbose=0,
    ).reshape(-1)

    if len(probabilities) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} probabilities, "
            f"found {len(probabilities)}."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities detected.")

    metrics = step10.binary_metrics(
        test_labels,
        probabilities,
    )

    print()
    print("RECOVERED FEDERATED TEST RESULTS")
    print("-" * 78)
    print(f"Accuracy: {float(metrics['accuracy']):.6f}")
    print(f"F1-score: {float(metrics['f1']):.6f}")
    print(f"ROC-AUC: {float(metrics['roc_auc']):.6f}")
    print(
        "Confusion matrix:",
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ],
    )

    accuracy_match = close_enough(
        metrics["accuracy"],
        EXPECTED_ACCURACY,
    )
    f1_match = close_enough(
        metrics["f1"],
        EXPECTED_F1,
    )
    auc_match = close_enough(
        metrics["roc_auc"],
        EXPECTED_ROC_AUC,
    )

    print()
    print("PUBLISHED RESULT CHECK")
    print("Accuracy match:", "YES" if accuracy_match else "NO")
    print("F1 match:", "YES" if f1_match else "NO")
    print("ROC-AUC match:", "YES" if auc_match else "NO")

    if not (accuracy_match and f1_match and auc_match):
        print()
        print("=" * 78)
        print("STOP")
        print("=" * 78)
        print(
            "The saved federated model did not reproduce all published "
            "held-out test metrics. No prediction CSV was written."
        )
        return

    predictions = (probabilities >= 0.5).astype(int)

    rows_to_save = []
    for index, key in enumerate(ordered_keys):
        rows_to_save.append(
            {
                "COHORT": key[0],
                "PATIENT_ID": key[1],
                "TARGET_STAGE_BINARY": int(test_labels[index]),
                "FEDERATED_PROBABILITY": float(probabilities[index]),
                "FEDERATED_PREDICTION": int(predictions[index]),
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "COHORT",
            "PATIENT_ID",
            "TARGET_STAGE_BINARY",
            "FEDERATED_PROBABILITY",
            "FEDERATED_PREDICTION",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows_to_save)

    print()
    print("=" * 78)
    print("FEDERATED REGENERATION=PASS")
    print("=" * 78)
    print("Rows saved:", len(rows_to_save))
    print("Output:", OUTPUT_CSV)
    print(
        "The federated predictions were regenerated from the saved "
        "model weights and are suitable for paired statistical testing."
    )


if __name__ == "__main__":
    main()
