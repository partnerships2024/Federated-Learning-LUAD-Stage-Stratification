"""Inspect the source dataset for federated-learning preparation.

This script performs read-only inspection. It does not create splits, databases,
models, or modified copies of the source CSV.
"""

import csv
from collections import Counter


CSV_FILENAME = "luad_301bp_patient_level_dataset.csv"


def print_value_counts(records: list[dict[str, str]], column_name: str) -> None:
    """Print value counts for one target column, including missing values."""
    print(f"\n{column_name} value counts:")
    counts = Counter(record.get(column_name, "") or "<MISSING>" for record in records)
    for value, count in counts.most_common():
        print(f"{value}: {count}")


def identify_stage_targets(
    records: list[dict[str, str]], columns: list[str]
) -> tuple[list[str], list[str]]:
    """Identify binary and four-class stage targets from existing CSV columns."""
    binary_candidates: list[str] = []
    four_class_candidates: list[str] = []

    for column_name in columns:
        normalized_name = str(column_name).upper()
        if "TARGET_STAGE" not in normalized_name:
            continue

        values = [record.get(column_name, "").strip() for record in records]
        values = [value for value in values if value]
        if not values:
            continue

        normalized_values = {str(value).strip().upper() for value in values}

        # Binary Early/Advanced targets may be represented by names or 0/1.
        if normalized_values.issubset({"EARLY", "ADVANCED"}) or normalized_values.issubset(
            {"0", "1", "0.0", "1.0"}
        ):
            binary_candidates.append(column_name)

        # Four-class targets may be represented by 1/2/3/4 or Stage I/II/III/IV.
        numeric_four_class = normalized_values.issubset(
            {"0", "1", "2", "3", "0.0", "1.0", "2.0", "3.0", "4", "4.0"}
        )
        named_four_class = normalized_values.issubset(
            {"I", "II", "III", "IV", "STAGE I", "STAGE II", "STAGE III", "STAGE IV"}
        )
        if len(normalized_values) >= 3 and (numeric_four_class or named_four_class):
            four_class_candidates.append(column_name)

    return binary_candidates, four_class_candidates


def main() -> None:
    """Run the read-only dataset inspection."""
    print("=" * 72)
    print("FEDERATED-LEARNING SOURCE DATA INSPECTION")
    print("=" * 72)

    # Section 1: Load the source CSV using its relative filename.
    print("\n--- 1. DATASET OVERVIEW ---")
    with open(CSV_FILENAME, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        records = list(reader)

    print(f"Dataset shape: ({len(records)}, {len(columns)})")
    print(f"Total number of rows: {len(records)}")
    print(f"Total number of columns: {len(columns)}")
    print("All column names:")
    for column_name in columns:
        print(f"- {column_name}")

    # Section 2: Cohort inspection.
    print("\n--- 2. COHORT INSPECTION ---")
    if "COHORT" in columns:
        cohort_values = [record.get("COHORT", "") for record in records]
        cohort_names = sorted({value for value in cohort_values if value})
        print(f"Number of unique COHORT values: {len(cohort_names)}")
        print("Cohort names:")
        for cohort_name in cohort_names:
            print(f"- {cohort_name}")
    else:
        print("COHORT column exists: False")

    # Section 3: Patient-identifier inspection; the key remains in memory only.
    print("\n--- 3. PATIENT IDENTIFIER INSPECTION ---")
    has_patient_id = "PATIENT_ID" in columns
    print(f"PATIENT_ID column exists: {has_patient_id}")
    if "COHORT" in columns and has_patient_id:
        patient_keys = [
            f"{record.get('COHORT', '')}::{record.get('PATIENT_ID', '')}"
            for record in records
        ]
        print(f"Number of unique patient keys: {len(set(patient_keys))}")
        key_counts = Counter(patient_keys)
        duplicate_patient_key_count = sum(1 for count in key_counts.values() if count > 1)
        print(f"Duplicate patient-key count: {duplicate_patient_key_count}")
    else:
        print("Patient key could not be created because COHORT or PATIENT_ID is missing.")

    # Section 4: Identify target columns from the columns present in the CSV.
    print("\n--- 4. STAGE TARGET IDENTIFICATION ---")
    binary_candidates, four_class_candidates = identify_stage_targets(records, columns)

    print("Candidate binary Early vs Advanced stage target columns:")
    if binary_candidates:
        for column_name in binary_candidates:
            print(f"- {column_name}")
            print_value_counts(records, column_name)
    else:
        print("- None identified")

    print("\nCandidate four-class Stage I/II/III/IV target columns:")
    if four_class_candidates:
        for column_name in four_class_candidates:
            print(f"- {column_name}")
            print_value_counts(records, column_name)
    else:
        print("- None identified")

    # Section 5: Missing values for the identified target columns.
    print("\n--- 5. MISSING VALUES IN IDENTIFIED TARGETS ---")
    identified_targets = list(dict.fromkeys(binary_candidates + four_class_candidates))
    if identified_targets:
        for column_name in identified_targets:
            missing_count = sum(not record.get(column_name, "").strip() for record in records)
            print(f"{column_name}: {missing_count} missing values")
    else:
        print("No target columns were identified.")

    # Section 6: Final status.
    required_columns_present = "COHORT" in columns and has_patient_id
    targets_present = bool(binary_candidates) and bool(four_class_candidates)
    print("\n--- 6. FINAL STATUS ---")
    if required_columns_present and targets_present:
        print("DATA INSPECTION STATUS: PASS")
    else:
        print("DATA INSPECTION STATUS: FAIL")


if __name__ == "__main__":
    main()
