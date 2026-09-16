"""Cached, hospital-local inference for the final federated Binary BiLSTM."""

import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_ROOT / "federated_results"
PREPROCESSING_PATH = RESULTS_DIRECTORY / "federated_preprocessing.json"
WEIGHTS_PATH = RESULTS_DIRECTORY / "final_global_bilstm.weights.h5"
STEP_09_PATH = PROJECT_ROOT / "09_validate_local_bilstm_training.py"
EXPECTED_PARAMETER_COUNT = 31777


def _load_step09_reference():
    spec = importlib.util.spec_from_file_location("step09_inference_reference", STEP_09_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Model reference is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_config(config: dict[str, object]) -> None:
    required = {
        "gene_vocabulary", "gene_vocabulary_size", "PAD_GENE", "UNK_GENE", "DNA_TOKEN_MAP",
        "MAX_WINDOWS", "DNA_LENGTH", "clinical_feature_order", "binary_imputation_modes",
        "numeric_imputation_medians", "scaler_mean", "scaler_scale",
    }
    if not required.issubset(config):
        raise RuntimeError("Preprocessing configuration is incomplete")
    if int(config["gene_vocabulary_size"]) != len(config["gene_vocabulary"]):
        raise RuntimeError("Preprocessing vocabulary is invalid")
    for name in ("binary_imputation_modes", "numeric_imputation_medians", "scaler_mean", "scaler_scale"):
        if not all(math.isfinite(float(value)) for value in config[name].values()):
            raise RuntimeError("Preprocessing statistics are invalid")


@lru_cache(maxsize=1)
def _load_model_and_config():
    if not PREPROCESSING_PATH.is_file() or not WEIGHTS_PATH.is_file():
        raise RuntimeError("Federated model artifacts are unavailable")
    with PREPROCESSING_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    import tensorflow as tf
    reference = _load_step09_reference()
    model = reference.build_binary_bilstm(tf, int(config["gene_vocabulary_size"]))
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Federated model parameter validation failed")
    model.load_weights(WEIGHTS_PATH)
    if not all(np.isfinite(weight).all() for weight in model.get_weights()):
        raise RuntimeError("Federated model weights are invalid")
    return model, config


def predict_local_patient(row: dict[str, object]) -> float:
    """Run prediction locally; row data never leaves this process."""
    model, config = _load_model_and_config()
    reference = _load_step09_reference()
    inputs = reference.build_arrays([row], config["gene_vocabulary"], config["binary_imputation_modes"] | config["numeric_imputation_medians"], config["scaler_mean"], config["scaler_scale"])[0]
    probability = float(model.predict(inputs, verbose=0).reshape(-1)[0])
    if not math.isfinite(probability):
        raise RuntimeError("Model returned an invalid prediction")
    return probability


def _clean_window(value: object, position: int) -> str:
    sequence = "".join(str(value or "").split()).upper()
    dna_length = 301
    if len(sequence) != dna_length:
        raise ValueError(f"Mutation {position}: sequence must contain exactly 301 bases")
    invalid = sorted(set(sequence) - {"A", "C", "G", "T", "N"})
    if invalid:
        raise ValueError(f"Mutation {position}: sequence contains invalid DNA characters")
    return sequence


def _normalize_new_sample_gene(value: object) -> str:
    """Normalize only a manually entered new-sample gene symbol."""
    return str(value).strip().upper()


@lru_cache(maxsize=1)
def get_supported_gene_symbols() -> list[str]:
    """Return biological symbols from the persisted inference vocabulary only."""
    if not PREPROCESSING_PATH.is_file():
        raise RuntimeError("Federated preprocessing configuration is unavailable")
    with PREPROCESSING_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    vocabulary = [str(symbol) for symbol in config["gene_vocabulary"]]
    special_tokens = {"PAD_GENE", "UNK_GENE", "NO_MUTATION"}
    biological_genes = sorted(symbol for symbol in vocabulary if symbol not in special_tokens)
    if int(config["gene_vocabulary_size"]) != 92 or len(vocabulary) != 92:
        raise RuntimeError("Persisted gene vocabulary size is not 92; refusing to build selector")
    if len(biological_genes) != 89:
        raise RuntimeError("Persisted biological gene count is not 89; refusing to build selector")
    print(f"Persisted vocabulary size: {len(vocabulary)}")
    print(f"Special tokens removed: {len(vocabulary) - len(biological_genes)}")
    print(f"Frontend-selectable biological genes: {len(biological_genes)}")
    print(f"First 10 selectable genes: {biological_genes[:10]}")
    print(f"TP53 present: {'TP53' in biological_genes}; FLT4 present: {'FLT4' in biological_genes}; OR8U1 present: {'OR8U1' in biological_genes}")
    return biological_genes


def prepare_new_sample(sample: dict[str, object]) -> tuple[dict[str, np.ndarray], list[str], int, list[str]]:
    """Build new-sample tensors from persisted settings, without fitting anything."""
    _, config = _load_model_and_config()
    mutations = sample.get("mutations", [])
    if not isinstance(mutations, list) or not mutations:
        raise ValueError("Enter at least one mutation input")
    if len(mutations) > int(config["MAX_WINDOWS"]):
        raise ValueError(f"A maximum of {config['MAX_WINDOWS']} mutation inputs is allowed")

    genes = []
    windows = []
    masks = []
    unknown_genes = []
    normalized_genes = []
    vocabulary = {str(key): int(value) for key, value in config["gene_vocabulary"].items()}
    for index, mutation in enumerate(mutations, start=1):
        gene = _normalize_new_sample_gene(mutation.get("gene", ""))
        if not gene:
            raise ValueError(f"Mutation {index}: gene symbol is required")
        sequence = _clean_window(mutation.get("sequence"), index)
        try:
            affected_position = int(str(mutation.get("position", "150")).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"Mutation {index}: affected position must be an integer from 0 to 300") from error
        if not 0 <= affected_position <= 300:
            raise ValueError(f"Mutation {index}: affected position must be between 0 and 300")
        genes.append(gene)
        normalized_genes.append(gene)
        windows.append(sequence)
        mask = ["0"] * 301
        mask[affected_position] = "1"
        masks.append("".join(mask))
        if gene not in vocabulary:
            unknown_genes.append(gene)

    row = {
        "COHORT": "LOCAL_REQUEST",
        "PATIENT_ID": "REQUEST_ONLY",
        "GENE_SYMBOL_LIST": json.dumps(genes),
        "MUTATED_WINDOW_301_LIST": json.dumps(windows),
        "MUTATED_AFFECTED_MASK_301_LIST": json.dumps(masks),
        "CLINICAL_SEX_BINARY": sample.get("sex_binary"),
        "CLINICAL_AGE_YEARS": sample.get("age"),
        "CLINICAL_ACTIVE_SMOKING_BINARY": sample.get("active_smoking_binary"),
        "CLINICAL_PACK_YEARS": sample.get("pack_years"),
        "TARGET_STAGE_BINARY": 0,
    }
    imputation = dict(config["binary_imputation_modes"])
    imputation.update(config["numeric_imputation_medians"])
    inputs, _ = _load_step09_reference().build_arrays(
        [row], vocabulary, imputation, config["scaler_mean"], config["scaler_scale"]
    )
    return inputs, unknown_genes, len(mutations), normalized_genes


def predict_new_sample(sample: dict[str, object]) -> tuple[float, list[str], int, list[str]]:
    """Run one request-local new-sample prediction using the cached model."""
    model, _ = _load_model_and_config()
    inputs, unknown_genes, mutation_count, normalized_genes = prepare_new_sample(sample)
    probability = float(model.predict(inputs, verbose=0).reshape(-1)[0])
    if not math.isfinite(probability):
        raise RuntimeError("Model returned an invalid prediction")
    return probability, unknown_genes, mutation_count, normalized_genes
