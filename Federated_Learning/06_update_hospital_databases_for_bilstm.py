"""Add Binary BiLSTM source columns to the three isolated hospital databases."""

import csv
import sqlite3
from pathlib import Path


SOURCE_CSV = Path("luad_301bp_patient_level_dataset.csv")
DATABASE_DIRECTORY = Path("databases")
SERVER_DATABASE = DATABASE_DIRECTORY / "federated_server.db"
HOSPITALS = {
    "hospital_1": (DATABASE_DIRECTORY / "hospital_1.db", 949),
    "hospital_2": (DATABASE_DIRECTORY / "hospital_2.db", 949),
    "hospital_3": (DATABASE_DIRECTORY / "hospital_3.db", 948),
}
KEY_COLUMNS = ["COHORT", "PATIENT_ID"]
REQUIRED_INPUT_COLUMNS = [
    "GENE_SYMBOL_LIST",
    "MUTATED_WINDOW_301_LIST",
    "MUTATED_AFFECTED_MASK_301_LIST",
    "CLINICAL_SEX_BINARY",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_ACTIVE_SMOKING_BINARY",
    "CLINICAL_PACK_YEARS",
]
ADDED_COLUMN_TYPES = {
    "GENE_SYMBOL_LIST": "TEXT",
    "MUTATED_WINDOW_301_LIST": "TEXT",
    "MUTATED_AFFECTED_MASK_301_LIST": "TEXT",
    "CLINICAL_SEX_BINARY": "INTEGER",
    "CLINICAL_ACTIVE_SMOKING_BINARY": "INTEGER",
    "CLINICAL_PACK_YEARS": "REAL",
}


def quote_identifier(identifier: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def load_source_rows() -> dict[tuple[str, str], dict[str, str]]:
    """Load source rows keyed by cohort and patient, without parsing list fields."""
    with open(SOURCE_CSV, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        source_columns = reader.fieldnames or []
        required_source_columns = set(KEY_COLUMNS + REQUIRED_INPUT_COLUMNS)
        missing = sorted(required_source_columns - set(source_columns))
        if missing:
            raise ValueError(f"Required source columns are missing: {missing}")
        rows = list(reader)

    source_by_key = {(row["COHORT"], row["PATIENT_ID"]): row for row in rows}
    if len(source_by_key) != len(rows):
        raise ValueError("Duplicate COHORT + PATIENT_ID keys found in the source CSV")
    return source_by_key


def table_columns(connection: sqlite3.Connection) -> list[str]:
    """Return patients-table columns in their existing SQLite order."""
    return [row[1] for row in connection.execute("PRAGMA table_info(patients)").fetchall()]


def hospital_keys(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    """Read the existing hospital patient keys before any schema update."""
    return [
        (cohort, patient_id)
        for cohort, patient_id in connection.execute(
            "SELECT COHORT, PATIENT_ID FROM patients"
        ).fetchall()
    ]


def update_hospital_database(
    database_path: Path, source_by_key: dict[tuple[str, str], dict[str, str]]
) -> tuple[int, list[str], int]:
    """Add missing columns and update only the database's existing patients."""
    with sqlite3.connect(database_path) as connection:
        existing_columns = table_columns(connection)
        original_keys = hospital_keys(connection)
        original_key_set = set(original_keys)
        if len(original_keys) != len(original_key_set):
            raise ValueError(f"Duplicate patient keys already exist in {database_path}")

        missing_source_rows = original_key_set - source_by_key.keys()
        if missing_source_rows:
            raise ValueError(f"Hospital patients missing from source CSV: {database_path}")

        added_columns = []
        for column, column_type in ADDED_COLUMN_TYPES.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE patients ADD COLUMN {quote_identifier(column)} {column_type}"
                )
                added_columns.append(column)

        update_columns = list(ADDED_COLUMN_TYPES)
        set_clause = ", ".join(f"{quote_identifier(column)} = ?" for column in update_columns)
        update_sql = (
            f"UPDATE patients SET {set_clause} WHERE {quote_identifier('COHORT')} = ? "
            f"AND {quote_identifier('PATIENT_ID')} = ?"
        )
        for key in original_keys:
            source_row = source_by_key[key]
            values = [
                None if source_row[column] == "" else source_row[column]
                for column in update_columns
            ]
            connection.execute(update_sql, values + [key[0], key[1]])

        row_count = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        duplicate_count = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT COHORT, PATIENT_ID
                FROM patients
                GROUP BY COHORT, PATIENT_ID
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        return row_count, added_columns, duplicate_count


def database_keys(database_path: Path) -> set[tuple[str, str]]:
    """Read patient keys from a hospital database for isolation validation."""
    with sqlite3.connect(database_path) as connection:
        return set(hospital_keys(connection))


def server_tables() -> list[str]:
    """Read server table names without changing the server database."""
    with sqlite3.connect(SERVER_DATABASE) as connection:
        return [
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]


def main() -> None:
    """Update, validate, and report the hospital database readiness state."""
    print("=" * 78)
    print("HOSPITAL DATABASE UPDATE FOR APPROVED BINARY BILSTM INPUTS")
    print("=" * 78)

    # Section 1: Load the source columns as raw values and inspect existing DB keys.
    source_by_key = load_source_rows()
    results = {}
    for hospital_name, (database_path, expected_count) in HOSPITALS.items():
        row_count, added_columns, duplicate_count = update_hospital_database(
            database_path, source_by_key
        )
        results[hospital_name] = {
            "path": database_path,
            "expected_count": expected_count,
            "row_count": row_count,
            "added_columns": added_columns,
            "duplicate_count": duplicate_count,
        }

    # Section 2: Validate required columns, counts, and pairwise isolation.
    print("\n--- DATABASE READINESS VALIDATION ---")
    key_sets = {}
    required_columns_available = True
    for hospital_name, result in results.items():
        with sqlite3.connect(result["path"]) as connection:
            columns = table_columns(connection)
        missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in columns]
        required_columns_available = required_columns_available and not missing_columns
        key_sets[hospital_name] = database_keys(result["path"])
        print(f"{hospital_name} row count: {result['row_count']} (expected {result['expected_count']})")
        print(f"{hospital_name} newly added columns: {result['added_columns'] or 'None; already present'}")
        print(f"{hospital_name} duplicate-key count: {result['duplicate_count']}")
        print(f"{hospital_name} missing required inputs: {missing_columns or 'None'}")

    overlap_12 = len(key_sets["hospital_1"] & key_sets["hospital_2"])
    overlap_13 = len(key_sets["hospital_1"] & key_sets["hospital_3"])
    overlap_23 = len(key_sets["hospital_2"] & key_sets["hospital_3"])
    print(f"Hospital 1 vs Hospital 2 overlap: {overlap_12}")
    print(f"Hospital 1 vs Hospital 3 overlap: {overlap_13}")
    print(f"Hospital 2 vs Hospital 3 overlap: {overlap_23}")

    # Section 3: Confirm the metadata-only server database remains patient-free.
    tables = server_tables()
    has_patient_table = any("patient" in table.lower() for table in tables)
    print(f"federated_server.db tables: {tables}")
    print(f"federated_server.db contains a patient table: {has_patient_table}")

    counts_unchanged = all(
        result["row_count"] == result["expected_count"] for result in results.values()
    )
    duplicate_keys_clear = all(result["duplicate_count"] == 0 for result in results.values())
    isolation_clear = overlap_12 == overlap_13 == overlap_23 == 0
    status_pass = required_columns_available and counts_unchanged and duplicate_keys_clear and isolation_clear and not has_patient_table

    print("\nFEDERATED INPUT DATA READINESS: PASS" if status_pass else "\nFEDERATED INPUT DATA READINESS: FAIL")


if __name__ == "__main__":
    main()
