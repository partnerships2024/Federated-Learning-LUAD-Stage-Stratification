"""Load the final federated Binary BiLSTM and run one local inference check."""

import contextlib
import importlib.util
import io
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "databases" / "hospital_1.db"
RESULTS_DIRECTORY = PROJECT_ROOT / "federated_results"
PREPROCESSING_PATH = RESULTS_DIRECTORY / "federated_preprocessing.json"
WEIGHTS_PATH = RESULTS_DIRECTORY / "final_global_bilstm.weights.h5"
STEP_09_PATH = PROJECT_ROOT / "09_validate_local_bilstm_training.py"
EXPECTED_PARAMETER_COUNT = 31777


def load_step09_reference():
    """Load the exact Step 9 model builder without executing its main function."""
    spec = importlib.util.spec_from_file_location("step09_reference", STEP_09_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {STEP_09_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_list(value: object, column: str) -> list[object]:
    if value is None or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"{column} is not a JSON list")
    return parsed


def numeric_value(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_one_patient() -> dict[str, object]:
    """Read one local record; its identity is never printed."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM patients LIMIT 1").fetchone()
    if row is None:
        raise ValueError("Hospital 1 database contains no patient record")
    return dict(row)


def clinical_value(row: dict[str, object], column: str, config: dict[str, object]) -> float:
    value = numeric_value(row.get(column))
    if value is None:
        if column in config["binary_imputation_modes"]:
            value = float(config["binary_imputation_modes"][column])
        else:
            value = float(config["numeric_imputation_medians"][column])
    return (value - float(config["scaler_mean"][column])) / float(config["scaler_scale"][column])


def build_single_patient_inputs(row: dict[str, object], config: dict[str, object]) -> dict[str, np.ndarray]:
    """Apply only persisted mappings/statistics to one local record."""
    max_windows = int(config["MAX_WINDOWS"])
    dna_length = int(config["DNA_LENGTH"])
    dna_map = {str(key): int(value) for key, value in config["DNA_TOKEN_MAP"].items()}
    vocabulary = {str(key): int(value) for key, value in config["gene_vocabulary"].items()}
    pad_gene = int(config["PAD_GENE"])
    unk_gene = int(config["UNK_GENE"])
    genes = parse_list(row.get("GENE_SYMBOL_LIST"), "GENE_SYMBOL_LIST")
    windows = parse_list(row.get("MUTATED_WINDOW_301_LIST"), "MUTATED_WINDOW_301_LIST")
    masks = parse_list(row.get("MUTATED_AFFECTED_MASK_301_LIST"), "MUTATED_AFFECTED_MASK_301_LIST")
    if not genes:
        genes = ["NO_MUTATION_GENE"]
        windows = ["N" * dna_length]
        masks = ["0" * dna_length]
    if len(genes) != len(windows) or len(genes) != len(masks) or len(genes) > max_windows:
        raise ValueError("Mutation lists are unaligned or exceed MAX_WINDOWS")

    gene_tokens = np.full((1, max_windows), pad_gene, dtype=np.int32)
    mutated_dna = np.full((1, max_windows, dna_length), dna_map["PAD"], dtype=np.int32)
    position_mask = np.zeros((1, max_windows, dna_length), dtype=np.float32)
    for index, gene in enumerate(genes):
        gene_tokens[0, index] = vocabulary.get(str(gene), unk_gene)
        sequence = str(windows[index]).upper()
        if len(sequence) != dna_length:
            raise ValueError("DNA sequence is not length DNA_LENGTH")
        mutated_dna[0, index] = [dna_map.get(base, dna_map["N"]) for base in sequence]
        raw_mask = list(masks[index]) if isinstance(masks[index], str) else masks[index]
        if not isinstance(raw_mask, list) or len(raw_mask) != dna_length:
            raise ValueError("Mutation mask is not length DNA_LENGTH")
        for position, value in enumerate(raw_mask):
            if str(value) not in {"0", "1"}:
                raise ValueError("Mutation mask contains a non-binary value")
            position_mask[0, index, position] = float(value)

    clinical_names = {
        "CLINICAL_SEX_BINARY": "clinical_sex",
        "CLINICAL_AGE_YEARS": "clinical_age",
        "CLINICAL_ACTIVE_SMOKING_BINARY": "clinical_active_smoking",
        "CLINICAL_PACK_YEARS": "clinical_pack_years",
    }
    clinical = {
        clinical_names[column]: np.array([[clinical_value(row, column, config)]], dtype=np.float32)
        for column in config["clinical_feature_order"]
    }
    return {
        "gene_tokens": gene_tokens,
        "mutated_dna": mutated_dna,
        "mutation_position_mask": position_mask,
        **clinical,
    }


def validate_config(config: dict[str, object]) -> None:
    required = {
        "gene_vocabulary", "gene_vocabulary_size", "PAD_GENE", "UNK_GENE", "DNA_TOKEN_MAP",
        "MAX_WINDOWS", "DNA_LENGTH", "clinical_feature_order", "binary_imputation_modes",
        "numeric_imputation_medians", "scaler_mean", "scaler_scale",
    }
    if not required.issubset(config):
        raise ValueError("Preprocessing artifact is missing required fields")
    if int(config["gene_vocabulary_size"]) != len(config["gene_vocabulary"]):
        raise ValueError("Gene vocabulary size mismatch")
    for mapping_name in ("binary_imputation_modes", "numeric_imputation_medians", "scaler_mean", "scaler_scale"):
        if not all(math.isfinite(float(value)) for value in config[mapping_name].values()):
            raise ValueError(f"Non-finite preprocessing value in {mapping_name}")


def main() -> None:
    print("=" * 78)
    print("FEDERATED BINARY BILSTM LOCAL INFERENCE VALIDATION")
    print("=" * 78)
    with PREPROCESSING_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    print("Preprocessing artifact loaded: YES")

    step09 = load_step09_reference()
    with contextlib.redirect_stderr(io.StringIO()):
        import tensorflow as tf
    model = step09.build_binary_bilstm(tf, int(config["gene_vocabulary_size"]))
    parameter_count = model.count_params()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Expected {EXPECTED_PARAMETER_COUNT} parameters, found {parameter_count}")
    model.load_weights(WEIGHTS_PATH)
    if not all(np.isfinite(weight).all() for weight in model.get_weights()):
        raise ValueError("Loaded model weights contain non-finite values")
    print(f"Model parameter count: {parameter_count}")
    print("Weights loaded: YES")

    row = load_one_patient()
    inputs = build_single_patient_inputs(row, config)
    expected_shapes = {
        "gene_tokens": (1, 37), "mutated_dna": (1, 37, 301),
        "mutation_position_mask": (1, 37, 301), "clinical_sex": (1, 1),
        "clinical_age": (1, 1), "clinical_active_smoking": (1, 1),
        "clinical_pack_years": (1, 1),
    }
    if {name: tuple(value.shape) for name, value in inputs.items()} != expected_shapes:
        raise ValueError("One or more inference tensor shapes are incorrect")
    if not all(np.isfinite(value).all() for value in inputs.values()):
        raise ValueError("Inference tensor contains NaN or Inf")
    probability = float(model.predict(inputs, verbose=0).reshape(-1)[0])
    if not math.isfinite(probability):
        raise ValueError("Prediction probability is non-finite")
    predicted_class = "Advanced" if probability >= 0.5 else "Early"
    stored_class = "Advanced" if int(float(row["TARGET_STAGE_BINARY"])) == 1 else "Early"

    print("\n--- LOCAL INFERENCE ---")
    for name in expected_shapes:
        print(f"{name}: {inputs[name].shape}")
    print(f"Prediction probability: {probability:.8f}")
    print(f"Predicted class: {predicted_class}")
    print("Prediction finite: YES")
    print(f"Stored class: {stored_class}")
    print(f"Technical match: {'YES' if stored_class == predicted_class else 'NO'}")
    print("\n--- PRIVACY CHECK ---")
    print("Patient record read only from hospital_1.db: YES")
    print("Patient record written to federated_server.db: NO")
    print("Raw DNA sent to server: NO")
    print("Inference occurred locally in this script: YES")
    print("Preprocessing refitted: NO")
    print("\nSyntax status: PASS")
    print("Execution status: PASS")
    print("FEDERATED INFERENCE READINESS: PASS")


if __name__ == "__main__":
    main()
