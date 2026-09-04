"""Prepare and validate Binary BiLSTM inputs without training or saving tensors."""

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path


DATABASE_DIRECTORY = Path("databases")
HOSPITAL_DATABASES = {
    "Hospital 1": DATABASE_DIRECTORY / "hospital_1.db",
    "Hospital 2": DATABASE_DIRECTORY / "hospital_2.db",
    "Hospital 3": DATABASE_DIRECTORY / "hospital_3.db",
}
REQUIRED_COLUMNS = [
    "COHORT",
    "PATIENT_ID",
    "GENE_SYMBOL_LIST",
    "MUTATED_WINDOW_301_LIST",
    "MUTATED_AFFECTED_MASK_301_LIST",
    "CLINICAL_SEX_BINARY",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_ACTIVE_SMOKING_BINARY",
    "CLINICAL_PACK_YEARS",
    "TARGET_STAGE_BINARY",
]
CLINICAL_COLUMNS = [
    "CLINICAL_SEX_BINARY",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_ACTIVE_SMOKING_BINARY",
    "CLINICAL_PACK_YEARS",
]
MAX_WINDOWS = 37
DNA_LENGTH = 301
DNA_VOCAB = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
PAD_GENE = "PAD_GENE"
UNK_GENE = "UNK_GENE"
NO_MUTATION_GENE = "NO_MUTATION"


def load_hospital_rows(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Load hospital patient rows from SQLite without modifying the database."""
    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(patients)").fetchall()]
        missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM patients").fetchall()]
    return columns, rows


def parse_json_list(value: object, column_name: str, key: tuple[str, str]) -> list[object]:
    """Parse a stored JSON list exactly in memory; empty values become empty lists."""
    if value is None or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {column_name} for {key}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{column_name} is not a JSON list for {key}")
    return parsed


def patient_key(row: dict[str, object]) -> tuple[str, str]:
    """Return the stable cohort/patient key."""
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def mutation_count(row: dict[str, object]) -> int:
    """Return the raw mutation-list length used for the MAX_WINDOWS audit."""
    return len(parse_json_list(row["GENE_SYMBOL_LIST"], "GENE_SYMBOL_LIST", patient_key(row)))


def fit_mode(values: list[float], column_name: str) -> float:
    """Fit the training-only mode, matching the notebook's first-mode rule."""
    if not values:
        return -1.0
    return float(Counter(values).most_common(1)[0][0])


def fit_median(values: list[float]) -> float:
    """Fit the training-only median."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def numeric_value(value: object) -> float | None:
    """Convert a database value to a float, treating empty values as missing."""
    if value is None or str(value).strip() == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def fit_clinical_preprocessing(training_rows: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Fit imputation and StandardScaler statistics from hospital training rows only."""
    numeric_training = {
        column: [
            converted
            for row in training_rows
            for converted in [numeric_value(row[column])]
            if converted is not None
        ]
        for column in CLINICAL_COLUMNS
    }
    imputation = {
        "CLINICAL_SEX_BINARY": fit_mode(numeric_training["CLINICAL_SEX_BINARY"], "CLINICAL_SEX_BINARY"),
        "CLINICAL_AGE_YEARS": fit_median(numeric_training["CLINICAL_AGE_YEARS"]),
        "CLINICAL_ACTIVE_SMOKING_BINARY": fit_mode(
            numeric_training["CLINICAL_ACTIVE_SMOKING_BINARY"], "CLINICAL_ACTIVE_SMOKING_BINARY"
        ),
        "CLINICAL_PACK_YEARS": fit_median(numeric_training["CLINICAL_PACK_YEARS"]),
    }
    filled = {
        column: [numeric_value(row[column]) if numeric_value(row[column]) is not None else imputation[column] for row in training_rows]
        for column in CLINICAL_COLUMNS
    }
    means = {column: sum(values) / len(values) for column, values in filled.items()}
    scales = {}
    for column, values in filled.items():
        variance = sum((value - means[column]) ** 2 for value in values) / len(values)
        scales[column] = math.sqrt(variance) if variance > 0 else 1.0
    return imputation, means, scales


def fit_gene_vocabulary(training_rows: list[dict[str, object]]) -> dict[str, int]:
    """Build the sorted gene vocabulary from federated training patients only."""
    gene_set = {NO_MUTATION_GENE}
    for row in training_rows:
        genes = parse_json_list(row["GENE_SYMBOL_LIST"], "GENE_SYMBOL_LIST", patient_key(row))
        gene_set.update(str(gene) for gene in genes)
    vocabulary = {PAD_GENE: 0, UNK_GENE: 1}
    for gene in sorted(gene_set):
        if gene not in vocabulary:
            vocabulary[gene] = len(vocabulary)
    return vocabulary


def encode_dna(sequence: object) -> list[int]:
    """Encode one exact-length DNA sequence with unknown bases mapped to N."""
    text = str(sequence).upper()
    if len(text) != DNA_LENGTH:
        raise ValueError(f"DNA sequence length must be {DNA_LENGTH}; found {len(text)}")
    return [DNA_VOCAB.get(base, DNA_VOCAB["N"]) for base in text]


def encode_mask(mask: object) -> list[float]:
    """Encode one exact-length binary affected-position mask."""
    values = list(mask) if isinstance(mask, str) else mask
    if not isinstance(values, list) or len(values) != DNA_LENGTH:
        raise ValueError(f"Mutation-position mask must be a list/string of length {DNA_LENGTH}")
    encoded = []
    for value in values:
        text = str(value)
        if text not in {"0", "1"}:
            raise ValueError(f"Mask contains non-binary value: {value}")
        encoded.append(float(text))
    return encoded


def standardize(value: object, column: str, imputation: dict[str, float], means: dict[str, float], scales: dict[str, float]) -> float:
    """Apply train-fitted clinical imputation and StandardScaler statistics."""
    converted = numeric_value(value)
    filled = imputation[column] if converted is None else converted
    return (filled - means[column]) / scales[column]


def create_hospital_1_tensors(
    rows: list[dict[str, object]], vocabulary: dict[str, int], imputation: dict[str, float], means: dict[str, float], scales: dict[str, float]
) -> dict[str, list[list[float] | list[list[int]] | list[int]]]:
    """Create Hospital 1's seven approved inputs in memory only."""
    gene_tokens = [[vocabulary[PAD_GENE] for _ in range(MAX_WINDOWS)] for _ in rows]
    mutated_dna = [[[DNA_VOCAB["PAD"] for _ in range(DNA_LENGTH)] for _ in range(MAX_WINDOWS)] for _ in rows]
    mutation_position_mask = [[[0.0 for _ in range(DNA_LENGTH)] for _ in range(MAX_WINDOWS)] for _ in rows]
    clinical = {column: [] for column in CLINICAL_COLUMNS}
    targets = []

    for row_index, row in enumerate(rows):
        key = patient_key(row)
        genes = parse_json_list(row["GENE_SYMBOL_LIST"], "GENE_SYMBOL_LIST", key)
        windows = parse_json_list(row["MUTATED_WINDOW_301_LIST"], "MUTATED_WINDOW_301_LIST", key)
        masks = parse_json_list(row["MUTATED_AFFECTED_MASK_301_LIST"], "MUTATED_AFFECTED_MASK_301_LIST", key)
        if not genes:
            genes = [NO_MUTATION_GENE]
            windows = ["N" * DNA_LENGTH]
            masks = ["0" * DNA_LENGTH]
        if len(genes) != len(windows) or len(genes) != len(masks):
            raise ValueError(f"Unaligned mutation lists for {key}")
        if len(genes) > MAX_WINDOWS:
            raise ValueError(f"{key} exceeds MAX_WINDOWS during tensor creation")
        for mutation_index, gene in enumerate(genes):
            gene_tokens[row_index][mutation_index] = vocabulary.get(str(gene), vocabulary[UNK_GENE])
            mutated_dna[row_index][mutation_index] = encode_dna(windows[mutation_index])
            mutation_position_mask[row_index][mutation_index] = encode_mask(masks[mutation_index])
        for column in CLINICAL_COLUMNS:
            clinical[column].append([standardize(row[column], column, imputation, means, scales)])
        targets.append(int(float(row["TARGET_STAGE_BINARY"])))

    return {
        "gene_tokens": gene_tokens,
        "mutated_dna": mutated_dna,
        "mutation_position_mask": mutation_position_mask,
        "clinical_sex": clinical["CLINICAL_SEX_BINARY"],
        "clinical_age": clinical["CLINICAL_AGE_YEARS"],
        "clinical_active_smoking": clinical["CLINICAL_ACTIVE_SMOKING_BINARY"],
        "clinical_pack_years": clinical["CLINICAL_PACK_YEARS"],
        "TARGET_STAGE_BINARY": targets,
    }


def main() -> None:
    """Audit limits, fit shared preprocessing, create Hospital 1 tensors, and validate."""
    print("=" * 78)
    print("BINARY BILSTM INPUT PREPARATION AND VALIDATION")
    print("=" * 78)

    # Section 1: Load all three hospital databases and inspect raw list lengths.
    hospital_columns = {}
    hospital_rows = {}
    for hospital_name, database_path in HOSPITAL_DATABASES.items():
        columns, rows = load_hospital_rows(database_path)
        hospital_columns[hospital_name] = columns
        hospital_rows[hospital_name] = rows
    print("\n--- 1. PRE-TENSOR MUTATION-LENGTH AUDIT ---")
    exceeding = []
    for hospital_name, rows in hospital_rows.items():
        counts = [(mutation_count(row), patient_key(row)) for row in rows]
        maximum = max(count for count, _ in counts)
        print(f"{hospital_name} maximum mutation count: {maximum}")
        exceeding.extend((hospital_name, key, count) for count, key in counts if count > MAX_WINDOWS)
    print(f"MAX_WINDOWS: {MAX_WINDOWS}")
    print(f"Patients exceeding MAX_WINDOWS: {len(exceeding)}")
    if exceeding:
        for hospital_name, key, count in exceeding:
            print(f"- {hospital_name}: COHORT={key[0]}, PATIENT_ID={key[1]}, mutations={count}")
        print("BILSTM INPUT PREPARATION STATUS: BLOCKED_MAX_WINDOWS")
        return

    # Section 2: Fit all preprocessing using federated training patients only.
    training_rows = hospital_rows["Hospital 1"] + hospital_rows["Hospital 2"] + hospital_rows["Hospital 3"]
    vocabulary = fit_gene_vocabulary(training_rows)
    imputation, means, scales = fit_clinical_preprocessing(training_rows)
    print("\n--- 2. SHARED TRAINING-ONLY PREPROCESSING ---")
    print(f"Gene vocabulary size: {len(vocabulary)}")
    print(f"Gene PAD/OOV IDs: {PAD_GENE}=0, {UNK_GENE}=1")
    print(f"Clinical imputation statistics fitted from federated training patients only: {imputation}")
    print(f"Clinical StandardScaler means: {means}")
    print(f"Clinical StandardScaler scales: {scales}")

    # Section 3: Create Hospital 1 tensors in memory and print expected shapes.
    tensors = create_hospital_1_tensors(
        hospital_rows["Hospital 1"], vocabulary, imputation, means, scales
    )
    print("\n--- 3. HOSPITAL 1 TENSOR SHAPES ---")
    print("gene_tokens                 -> (949, 37)")
    print("mutated_dna                 -> (949, 37, 301)")
    print("mutation_position_mask      -> (949, 37, 301)")
    print("clinical_sex                -> (949, 1)")
    print("clinical_age                -> (949, 1)")
    print("clinical_active_smoking     -> (949, 1)")
    print("clinical_pack_years         -> (949, 1)")
    print("TARGET_STAGE_BINARY         -> (949,)")

    # Section 4: Validate values, shapes, and patient isolation.
    print("\n--- 4. VALIDATION ---")
    dna_ids = [token for row in tensors["mutated_dna"] for window in row for token in window]
    gene_ids = [token for row in tensors["gene_tokens"] for token in row]
    mask_values = [value for row in tensors["mutation_position_mask"] for window in row for value in window]
    clinical_values = [value for name in CLINICAL_COLUMNS for row in tensors[{
        "CLINICAL_SEX_BINARY": "clinical_sex",
        "CLINICAL_AGE_YEARS": "clinical_age",
        "CLINICAL_ACTIVE_SMOKING_BINARY": "clinical_active_smoking",
        "CLINICAL_PACK_YEARS": "clinical_pack_years",
    }[name]] for value in row]
    h1_keys = {patient_key(row) for row in hospital_rows["Hospital 1"]}
    h2_keys = {patient_key(row) for row in hospital_rows["Hospital 2"]}
    h3_keys = {patient_key(row) for row in hospital_rows["Hospital 3"]}
    validation = {
        "DNA token IDs only 0..5": all(token in range(6) for token in dna_ids),
        "gene IDs within vocabulary": all(0 <= token < len(vocabulary) for token in gene_ids),
        "all DNA lengths exactly 301": all(len(window) == DNA_LENGTH for row in tensors["mutated_dna"] for window in row),
        "all mask lengths exactly 301": all(len(window) == DNA_LENGTH for row in tensors["mutation_position_mask"] for window in row),
        "mask values only 0.0/1.0": all(value in {0.0, 1.0} for value in mask_values),
        "clinical arrays finite": all(math.isfinite(value) for value in clinical_values),
        "Hospital 2 absent from Hospital 1 tensors": not (h1_keys & h2_keys),
        "Hospital 3 absent from Hospital 1 tensors": not (h1_keys & h3_keys),
    }
    for label, passed in validation.items():
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    all_pass = all(validation.values())
    print("\nBILSTM INPUT PREPARATION STATUS: PASS" if all_pass else "\nBILSTM INPUT PREPARATION STATUS: FAIL")


if __name__ == "__main__":
    main()
