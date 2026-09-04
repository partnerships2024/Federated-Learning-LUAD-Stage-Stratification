"""Split the federated training pool into three simulated hospitals.

The script preserves lightweight patient-key rows, uses deterministic
cohort/label-stratified allocation, and does not modify any source files.
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


TRAINING_KEYS_FILENAME = Path("federated_results/federated_training_pool_keys.csv")
GLOBAL_TEST_KEYS_FILENAME = Path("federated_results/global_test_patient_keys.csv")
OUTPUT_DIRECTORY = Path("federated_results")
HOSPITAL_FILENAMES = {
    "Hospital 1": OUTPUT_DIRECTORY / "hospital_1_patient_keys.csv",
    "Hospital 2": OUTPUT_DIRECTORY / "hospital_2_patient_keys.csv",
    "Hospital 3": OUTPUT_DIRECTORY / "hospital_3_patient_keys.csv",
}
RANDOM_SEED = 42
TARGET_COLUMN = "TARGET_STAGE_BINARY"
IDENTIFIER_COLUMNS = ["COHORT", "PATIENT_ID"]


def load_key_file(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a lightweight key file."""
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def patient_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the stable cohort/patient identifier pair."""
    return row["COHORT"], row["PATIENT_ID"]


def allocate_rows(
    training_rows: list[dict[str, str]], has_cohort: bool
) -> dict[str, list[dict[str, str]]]:
    """Allocate every row to one hospital by cohort/label strata."""
    random_generator = random.Random(RANDOM_SEED)
    hospitals = list(HOSPITAL_FILENAMES)
    rows_by_stratum: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)

    for row in training_rows:
        stratum = (
            (row.get("COHORT", ""), row[TARGET_COLUMN])
            if has_cohort
            else (row[TARGET_COLUMN],)
        )
        rows_by_stratum[stratum].append(row)

    allocations = {hospital: [] for hospital in hospitals}
    current_counts = {hospital: 0 for hospital in hospitals}

    for stratum in sorted(rows_by_stratum):
        stratum_rows = rows_by_stratum[stratum].copy()
        random_generator.shuffle(stratum_rows)

        # Give each hospital floor(n/3) rows, then assign remainders to the
        # currently smallest hospitals for near-equal total patient counts.
        base_count, remainder = divmod(len(stratum_rows), len(hospitals))
        desired_counts = {hospital: base_count for hospital in hospitals}
        remainder_order = sorted(
            hospitals,
            key=lambda hospital: (current_counts[hospital], hospitals.index(hospital)),
        )
        for hospital in remainder_order[:remainder]:
            desired_counts[hospital] += 1

        offset = 0
        for hospital in hospitals:
            count = desired_counts[hospital]
            allocations[hospital].extend(stratum_rows[offset : offset + count])
            current_counts[hospital] += count
            offset += count

    for hospital in hospitals:
        random_generator.shuffle(allocations[hospital])
    return allocations


def print_hospital_summary(
    hospital_name: str, rows: list[dict[str, str]], has_cohort: bool
) -> None:
    """Print count, target distribution, and optional cohort distribution."""
    print(f"{hospital_name} patient count: {len(rows)}")
    target_distribution = Counter(row[TARGET_COLUMN] for row in rows)
    print(f"{hospital_name} binary target distribution: {dict(sorted(target_distribution.items()))}")
    if has_cohort:
        cohort_distribution = Counter(row["COHORT"] for row in rows)
        print(f"{hospital_name} cohort distribution: {dict(sorted(cohort_distribution.items()))}")


def write_key_file(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write only the columns already present in the lightweight input keys."""
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def main() -> None:
    """Split, verify, and save the three simulated hospital partitions."""
    print("=" * 72)
    print("SIMULATED HOSPITAL CLIENT SPLIT")
    print("=" * 72)

    # Section 1: Load and validate the federated training pool.
    print("\n--- 1. LOAD FEDERATED TRAINING POOL ---")
    training_columns, training_rows = load_key_file(TRAINING_KEYS_FILENAME)
    missing_columns = [
        column for column in IDENTIFIER_COLUMNS + [TARGET_COLUMN] if column not in training_columns
    ]
    if missing_columns:
        raise ValueError(f"Required training-key columns are missing: {missing_columns}")
    if len(training_rows) != 2846:
        raise ValueError(f"Expected 2846 training patients, found {len(training_rows)}")

    global_test_columns, global_test_rows = load_key_file(GLOBAL_TEST_KEYS_FILENAME)
    if any(column not in global_test_columns for column in IDENTIFIER_COLUMNS):
        raise ValueError("Global test key file is missing a required patient identifier column")

    has_cohort = "COHORT" in training_columns
    allocations = allocate_rows(training_rows, has_cohort)

    # Section 2: Print each hospital's distribution.
    print("\n--- 2. HOSPITAL DISTRIBUTIONS ---")
    for hospital_name, rows in allocations.items():
        print_hospital_summary(hospital_name, rows, has_cohort)

    # Section 3: Validate pairwise disjointness and complete coverage.
    print("\n--- 3. CLIENT SPLIT VALIDATION ---")
    hospital_keys = {
        hospital_name: {patient_key(row) for row in rows}
        for hospital_name, rows in allocations.items()
    }
    overlap_12 = len(hospital_keys["Hospital 1"] & hospital_keys["Hospital 2"])
    overlap_13 = len(hospital_keys["Hospital 1"] & hospital_keys["Hospital 3"])
    overlap_23 = len(hospital_keys["Hospital 2"] & hospital_keys["Hospital 3"])
    source_keys = {patient_key(row) for row in training_rows}
    combined_keys = set().union(*hospital_keys.values())
    combined_row_count = sum(len(rows) for rows in allocations.values())
    duplicate_count = combined_row_count - len(combined_keys)
    lost_count = len(source_keys - combined_keys)

    print(f"Hospital 1 vs Hospital 2 overlap: {overlap_12}")
    print(f"Hospital 1 vs Hospital 3 overlap: {overlap_13}")
    print(f"Hospital 2 vs Hospital 3 overlap: {overlap_23}")
    print(f"Combined hospital patient count: {combined_row_count}")
    print(f"No training patient lost: {lost_count == 0} (lost: {lost_count})")
    print(f"No training patient duplicated: {duplicate_count == 0} (duplicates: {duplicate_count})")

    # Section 4: Verify no simulated hospital patient is in the global test set.
    global_test_keys = {patient_key(row) for row in global_test_rows}
    global_test_overlap = len(combined_keys & global_test_keys)
    print(f"Overlap with global test set: {global_test_overlap}")

    # Section 5: Save only the three lightweight hospital key files.
    print("\n--- 5. SAVE HOSPITAL KEY FILES ---")
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for hospital_name, path in HOSPITAL_FILENAMES.items():
        write_key_file(path, training_columns, allocations[hospital_name])
        print(f"Created: {path}")

    # Section 6: Final status.
    all_checks_pass = (
        overlap_12 == 0
        and overlap_13 == 0
        and overlap_23 == 0
        and combined_row_count == len(training_rows)
        and lost_count == 0
        and duplicate_count == 0
        and global_test_overlap == 0
    )
    print("\n--- 6. FINAL STATUS ---")
    print("CLIENT SPLIT STATUS: PASS" if all_checks_pass else "CLIENT SPLIT STATUS: FAIL")


if __name__ == "__main__":
    main()
