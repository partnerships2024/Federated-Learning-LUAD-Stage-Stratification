"""Persist the exact training-only preprocessing configuration for inference.

This script reconstructs Step 10 preprocessing from the three local hospital
databases.  It does not train a model, read held-out test patients, or store
patient-level records in the output artifact.
"""

import importlib.util
import json
import math
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIRECTORY = PROJECT_ROOT / "databases"
RESULTS_DIRECTORY = PROJECT_ROOT / "federated_results"
OUTPUT_PATH = RESULTS_DIRECTORY / "federated_preprocessing.json"
STEP_09_PATH = PROJECT_ROOT / "09_validate_local_bilstm_training.py"
RANDOM_SEED = 42
EXPECTED_TRAINING_COUNT = 2276
EXPECTED_LOCAL_TRAINING_COUNTS = {
    "Hospital 1": 759,
    "Hospital 2": 759,
    "Hospital 3": 758,
}
DNA_TOKEN_MAP = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
MAX_WINDOWS = 37
DNA_LENGTH = 301
PAD_GENE = 0
UNK_GENE = 1
CLINICAL_FEATURE_ORDER = [
    "CLINICAL_SEX_BINARY",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_ACTIVE_SMOKING_BINARY",
    "CLINICAL_PACK_YEARS",
]


def load_step09_reference():
    """Load Step 9 helpers without executing its training-validation main."""
    spec = importlib.util.spec_from_file_location("step09_reference", STEP_09_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {STEP_09_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_database_rows(path: Path) -> list[dict[str, object]]:
    """Read a hospital database without modifying it."""
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM patients").fetchall()]


def patient_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def finite_mapping(mapping: dict[str, object]) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in mapping.values())


def main() -> None:
    print("=" * 78)
    print("PERSIST FEDERATED BINARY BILSTM PREPROCESSING")
    print("=" * 78)
    step09 = load_step09_reference()

    hospital_rows = {
        "Hospital 1": load_database_rows(DATABASE_DIRECTORY / "hospital_1.db"),
        "Hospital 2": load_database_rows(DATABASE_DIRECTORY / "hospital_2.db"),
        "Hospital 3": load_database_rows(DATABASE_DIRECTORY / "hospital_3.db"),
    }
    local_train_rows = []
    local_validation_rows = []
    local_train_counts = {}
    for hospital, rows in hospital_rows.items():
        train_rows, validation_rows = step09.stratified_train_validation(rows)
        local_train_rows.extend(train_rows)
        local_validation_rows.extend(validation_rows)
        local_train_counts[hospital] = len(train_rows)

    if local_train_counts != EXPECTED_LOCAL_TRAINING_COUNTS:
        raise ValueError(f"Unexpected local training counts: {local_train_counts}")
    if len(local_train_rows) != EXPECTED_TRAINING_COUNT:
        raise ValueError(f"Expected {EXPECTED_TRAINING_COUNT} training patients, found {len(local_train_rows)}")
    train_keys = {patient_key(row) for row in local_train_rows}
    validation_keys = {patient_key(row) for row in local_validation_rows}
    if train_keys & validation_keys:
        raise ValueError("Training and validation patients overlap")

    # This invokes the exact Step 9/10 fitting logic on training rows only.
    imputation, vocabulary, means, scales = step09.fit_preprocessing(local_train_rows)
    binary_imputation_modes = {
        "CLINICAL_SEX_BINARY": imputation["CLINICAL_SEX_BINARY"],
        "CLINICAL_ACTIVE_SMOKING_BINARY": imputation["CLINICAL_ACTIVE_SMOKING_BINARY"],
    }
    numeric_imputation_medians = {
        "CLINICAL_AGE_YEARS": imputation["CLINICAL_AGE_YEARS"],
        "CLINICAL_PACK_YEARS": imputation["CLINICAL_PACK_YEARS"],
    }
    artifact = {
        "preprocessing_version": "Step10-reconstructed-v1",
        "random_seed": RANDOM_SEED,
        "training_patient_count": len(local_train_rows),
        "gene_vocabulary": vocabulary,
        "gene_vocabulary_size": len(vocabulary),
        "PAD_GENE": PAD_GENE,
        "UNK_GENE": UNK_GENE,
        "DNA_TOKEN_MAP": DNA_TOKEN_MAP,
        "MAX_WINDOWS": MAX_WINDOWS,
        "DNA_LENGTH": DNA_LENGTH,
        "clinical_feature_order": CLINICAL_FEATURE_ORDER,
        "binary_imputation_modes": binary_imputation_modes,
        "numeric_imputation_medians": numeric_imputation_medians,
        "scaler_mean": means,
        "scaler_scale": scales,
        "source_description": "Reconstructed deterministically from the exact federated training patients used in Step 10; no validation or held-out test patients used.",
    }

    if len(vocabulary) != 92:
        raise ValueError(f"Expected gene vocabulary size 92, found {len(vocabulary)}")
    if not finite_mapping(binary_imputation_modes) or not finite_mapping(numeric_imputation_medians):
        raise ValueError("Imputation values are not finite")
    if not finite_mapping(means) or not finite_mapping(scales) or any(float(value) <= 0 for value in scales.values()):
        raise ValueError("Scaler values are invalid")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    required_fields = {
        "preprocessing_version", "random_seed", "training_patient_count",
        "gene_vocabulary", "gene_vocabulary_size", "PAD_GENE", "UNK_GENE",
        "DNA_TOKEN_MAP", "MAX_WINDOWS", "DNA_LENGTH", "clinical_feature_order",
        "binary_imputation_modes", "numeric_imputation_medians", "scaler_mean",
        "scaler_scale", "source_description",
    }
    if not required_fields.issubset(reloaded) or len(reloaded["gene_vocabulary"]) != 92:
        raise ValueError("JSON reload validation failed")
    serialized = json.dumps(reloaded)
    identifier_fields = ("PATIENT_ID", "SAMPLE_ID", "COHORT")
    identifiers_stored = any(field in serialized for field in identifier_fields)
    if identifiers_stored:
        raise ValueError("Patient identifier field found in preprocessing JSON")

    print("\n--- FITTING VALIDATION ---")
    print(f"Hospital local training counts: {local_train_counts}")
    print(f"Combined training patient count: {len(local_train_rows)}")
    print("Validation patients used in fitting: 0")
    print("Global held-out test patients used in fitting: 0 (test keys were not loaded)")
    print(f"Gene vocabulary size: {len(vocabulary)}")
    print(f"Clinical feature order: {CLINICAL_FEATURE_ORDER}")
    print(f"Binary imputation modes: {binary_imputation_modes}")
    print(f"Numeric imputation medians: {numeric_imputation_medians}")
    print(f"Scaler mean: {means}")
    print(f"Scaler scale: {scales}")
    print(f"DNA token map: {DNA_TOKEN_MAP}")
    print(f"JSON reload: PASS; patient identifiers stored: {'YES' if identifiers_stored else 'NO'}")
    print(f"Saved preprocessing artifact: {OUTPUT_PATH}")
    print("FEDERATED PREPROCESSING ARTIFACT STATUS: PASS")


if __name__ == "__main__":
    main()
