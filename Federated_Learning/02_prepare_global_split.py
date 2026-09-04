"""Prepare labelled patients and a global held-out test split.

This script is read-only with respect to the source CSV. It creates only two
lightweight partition-key files under ``federated_results``.
"""

import csv
import math
import random
from collections import Counter
from pathlib import Path


CSV_FILENAME = "luad_301bp_patient_level_dataset.csv"
OUTPUT_DIRECTORY = Path("federated_results")
TRAINING_KEYS_FILENAME = OUTPUT_DIRECTORY / "federated_training_pool_keys.csv"
TEST_KEYS_FILENAME = OUTPUT_DIRECTORY / "global_test_patient_keys.csv"
RANDOM_SEED = 42
TARGET_COLUMN = "TARGET_STAGE_BINARY"
KEY_COLUMNS = ["COHORT", "PATIENT_ID", TARGET_COLUMN]


def load_source_rows() -> tuple[list[str], list[dict[str, str]]]:
    """Load the source CSV without changing it."""
    with open(CSV_FILENAME, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def is_missing(value: str | None) -> bool:
    """Treat empty CSV fields as missing values."""
    return value is None or value.strip() == ""


def stratified_split(
    labelled_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic 80/20 stratified split by binary target."""
    random_generator = random.Random(RANDOM_SEED)
    rows_by_class: dict[str, list[dict[str, str]]] = {}
    for row in labelled_rows:
        rows_by_class.setdefault(row[TARGET_COLUMN], []).append(row)

    total_rows = len(labelled_rows)
    requested_test_count = math.ceil(total_rows * 0.20)
    class_names = sorted(rows_by_class)
    exact_allocations = {
        class_name: len(rows_by_class[class_name]) * requested_test_count / total_rows
        for class_name in class_names
    }
    test_counts = {
        class_name: math.floor(exact_allocations[class_name]) for class_name in class_names
    }
    remaining = requested_test_count - sum(test_counts.values())
    for class_name in sorted(
        class_names,
        key=lambda name: exact_allocations[name] - test_counts[name],
        reverse=True,
    )[:remaining]:
        test_counts[class_name] += 1

    training_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for class_name in class_names:
        class_rows = rows_by_class[class_name].copy()
        random_generator.shuffle(class_rows)
        class_test_count = test_counts[class_name]
        test_rows.extend(class_rows[:class_test_count])
        training_rows.extend(class_rows[class_test_count:])

    random_generator.shuffle(training_rows)
    random_generator.shuffle(test_rows)
    return training_rows, test_rows


def patient_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the preserved cohort/patient identifier pair."""
    return row["COHORT"], row["PATIENT_ID"]


def print_distribution(label: str, rows: list[dict[str, str]]) -> None:
    """Print the binary target distribution for a partition."""
    distribution = Counter(row[TARGET_COLUMN] for row in rows)
    print(f"{label} class distribution: {dict(sorted(distribution.items()))}")


def write_key_file(path: Path, rows: list[dict[str, str]]) -> None:
    """Write only identifiers and target information needed to reproduce splits."""
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row[column] for column in KEY_COLUMNS} for row in rows)


def main() -> None:
    """Load, validate, split, verify, and save the two partition-key files."""
    print("=" * 72)
    print("LABELLED DATASET AND GLOBAL HELD-OUT TEST SPLIT")
    print("=" * 72)

    # Section 1: Load and label-filter the source dataset.
    print("\n--- 1. LABELLED DATASET ---")
    source_columns, source_rows = load_source_rows()
    missing_columns = [column for column in KEY_COLUMNS if column not in source_columns]
    if missing_columns:
        raise ValueError(f"Required source columns are missing: {missing_columns}")

    labelled_rows = [row for row in source_rows if not is_missing(row[TARGET_COLUMN])]
    excluded_rows = len(source_rows) - len(labelled_rows)
    print(f"Total original patients: {len(source_rows)}")
    print(f"Labelled patients: {len(labelled_rows)}")
    print(f"Excluded unlabelled patients: {excluded_rows}")
    print_distribution("Labelled dataset", labelled_rows)

    # Section 2: Create the patient-level, stratified global split.
    print("\n--- 2. GLOBAL PATIENT-LEVEL SPLIT ---")
    training_rows, test_rows = stratified_split(labelled_rows)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Training pool patient count: {len(training_rows)}")
    print(f"Global held-out test patient count: {len(test_rows)}")
    print_distribution("Training pool", training_rows)
    print_distribution("Global held-out test", test_rows)

    # Section 3: Verify identifiers and ensure no patient crosses partitions.
    print("\n--- 3. SPLIT VERIFICATION ---")
    training_keys = {patient_key(row) for row in training_rows}
    test_keys = {patient_key(row) for row in test_rows}
    overlap_count = len(training_keys & test_keys)
    print(f"Training pool unique patient count: {len(training_keys)}")
    print(f"Global test unique patient count: {len(test_keys)}")
    print(f"Patient overlap count: {overlap_count}")

    # Section 4: Save only lightweight partition-key files.
    print("\n--- 4. SAVE PARTITION KEYS ---")
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    write_key_file(TRAINING_KEYS_FILENAME, training_rows)
    write_key_file(TEST_KEYS_FILENAME, test_rows)
    print(f"Created: {TRAINING_KEYS_FILENAME}")
    print(f"Created: {TEST_KEYS_FILENAME}")
    print(f"Key-file columns: {KEY_COLUMNS}")

    # Section 5: Final status.
    print("\n--- 5. FINAL STATUS ---")
    if overlap_count == 0:
        print("GLOBAL SPLIT STATUS: PASS")
    else:
        print("GLOBAL SPLIT STATUS: FAIL")


if __name__ == "__main__":
    main()
