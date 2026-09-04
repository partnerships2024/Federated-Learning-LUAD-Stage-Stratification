"""Create isolated hospital SQLite databases and server round metadata."""

import csv
import sqlite3
from collections import Counter
from pathlib import Path


SOURCE_CSV = Path("luad_301bp_patient_level_dataset.csv")
RESULTS_DIRECTORY = Path("federated_results")
DATABASE_DIRECTORY = Path("databases")
SERVER_DATABASE = DATABASE_DIRECTORY / "federated_server.db"

HOSPITAL_KEY_FILES = {
    "hospital_1": RESULTS_DIRECTORY / "hospital_1_patient_keys.csv",
    "hospital_2": RESULTS_DIRECTORY / "hospital_2_patient_keys.csv",
    "hospital_3": RESULTS_DIRECTORY / "hospital_3_patient_keys.csv",
}
EXPECTED_COUNTS = {"hospital_1": 949, "hospital_2": 949, "hospital_3": 948}
IDENTIFIER_COLUMNS = ["COHORT", "PATIENT_ID"]
REQUESTED_PATIENT_COLUMNS = [
    "COHORT",
    "PATIENT_ID",
    "SAMPLE_ID",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_SEX",
    "TARGET_STAGE_BINARY",
    "TARGET_STAGE_BINARY_NAME",
    "TARGET_STAGE_4CLASS",
    "TARGET_STAGE_GROUP",
    "VALID_301_MUTATION_COUNT",
    "VALID_301_GENE_COUNT",
    "TOTAL_SELECTED_MUTATION_ROWS",
    "TOTAL_SELECTED_MUTATED_GENES",
    "HAS_HIGH_SEVERITY_MUTATION_FLAG",
    "HAS_FRAMESHIFT_MUTATION_FLAG",
]


def quote_identifier(identifier: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV into memory without changing the file."""
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def patient_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the required cohort/patient identifier."""
    return row["COHORT"], row["PATIENT_ID"]


def sqlite_type(values: list[str]) -> str:
    """Infer a lightweight SQLite type from the non-empty source values."""
    non_empty = [value for value in values if value != ""]
    if not non_empty:
        return "TEXT"
    try:
        for value in non_empty:
            int(value)
        return "INTEGER"
    except ValueError:
        try:
            for value in non_empty:
                float(value)
            return "REAL"
        except ValueError:
            return "TEXT"


def create_hospital_database(
    path: Path, columns: list[str], rows: list[dict[str, str]]
) -> None:
    """Create one local database containing only the assigned patients."""
    column_definitions = []
    for column in columns:
        column_type = sqlite_type([row[column] for row in rows])
        required = " NOT NULL" if column in IDENTIFIER_COLUMNS else ""
        column_definitions.append(f"{quote_identifier(column)} {column_type}{required}")
    primary_key = ", PRIMARY KEY (" + ", ".join(quote_identifier(c) for c in IDENTIFIER_COLUMNS) + ")"

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS patients")
        connection.execute(
            "CREATE TABLE patients (" + ", ".join(column_definitions) + primary_key + ")"
        )
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            f"INSERT INTO patients ({', '.join(quote_identifier(c) for c in columns)}) "
            f"VALUES ({placeholders})"
        )
        values = [
            [None if row[column] == "" else row[column] for column in columns]
            for row in rows
        ]
        connection.executemany(insert_sql, values)


def create_server_database(path: Path) -> None:
    """Create metadata storage without any patient-level table."""
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS federated_rounds")
        connection.execute(
            """
            CREATE TABLE federated_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER,
                client_id TEXT,
                local_sample_count INTEGER,
                local_loss REAL,
                local_accuracy REAL,
                update_received INTEGER,
                global_loss REAL,
                global_accuracy REAL,
                global_f1 REAL,
                global_auc REAL,
                model_version TEXT,
                timestamp TEXT
            )
            """
        )


def database_patient_keys(path: Path) -> set[tuple[str, str]]:
    """Read patient identifiers back from a local database for validation."""
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT COHORT, PATIENT_ID FROM patients"
        ).fetchall()
    return {(cohort, patient_id) for cohort, patient_id in rows}


def database_tables(path: Path) -> list[str]:
    """Return user tables in a SQLite database."""
    with sqlite3.connect(path) as connection:
        return [
            table_name
            for (table_name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]


def main() -> None:
    """Load source data, create databases, and validate isolation."""
    print("=" * 72)
    print("ISOLATED HOSPITAL DATABASE CREATION")
    print("=" * 72)

    # Section 1: Load source data and hospital partition keys.
    source_columns, source_rows = load_csv(SOURCE_CSV)
    missing_identifiers = [column for column in IDENTIFIER_COLUMNS if column not in source_columns]
    if missing_identifiers:
        raise ValueError(f"Source identifiers are missing: {missing_identifiers}")
    selected_columns = [column for column in REQUESTED_PATIENT_COLUMNS if column in source_columns]
    source_by_key = {patient_key(row): row for row in source_rows}
    hospital_keys: dict[str, set[tuple[str, str]]] = {}
    hospital_rows: dict[str, list[dict[str, str]]] = {}

    for hospital_name, key_path in HOSPITAL_KEY_FILES.items():
        key_columns, key_rows = load_csv(key_path)
        if any(column not in key_columns for column in IDENTIFIER_COLUMNS):
            raise ValueError(f"{key_path} is missing a required identifier column")
        keys = {patient_key(row) for row in key_rows}
        if len(key_rows) != len(keys):
            raise ValueError(f"Duplicate patient identifiers found in {key_path}")
        missing_source_rows = keys - source_by_key.keys()
        if missing_source_rows:
            raise ValueError(f"Patients in {key_path} are absent from the source CSV")
        hospital_keys[hospital_name] = keys
        hospital_rows[hospital_name] = [source_by_key[key] for key in sorted(keys)]
        print(f"{hospital_name} expected row count from key CSV: {len(key_rows)}")

    # Section 2: Create the three isolated local hospital databases.
    DATABASE_DIRECTORY.mkdir(exist_ok=True)
    for hospital_name, rows in hospital_rows.items():
        create_hospital_database(
            DATABASE_DIRECTORY / f"{hospital_name}.db", selected_columns, rows
        )

    # Section 3: Create the metadata-only federated server database.
    create_server_database(SERVER_DATABASE)

    # Section 4: Validate row counts, isolation, and server contents.
    print("\n--- DATABASE VALIDATION ---")
    actual_counts: dict[str, int] = {}
    database_keys: dict[str, set[tuple[str, str]]] = {}
    for hospital_name in HOSPITAL_KEY_FILES:
        database_path = DATABASE_DIRECTORY / f"{hospital_name}.db"
        with sqlite3.connect(database_path) as connection:
            actual_counts[hospital_name] = connection.execute(
                "SELECT COUNT(*) FROM patients"
            ).fetchone()[0]
        database_keys[hospital_name] = database_patient_keys(database_path)
        print(f"{hospital_name} rows in database: {actual_counts[hospital_name]}")

    foreign_patient_counts = {}
    for hospital_name, keys in database_keys.items():
        other_keys = set().union(
            *(hospital_keys[other] for other in HOSPITAL_KEY_FILES if other != hospital_name)
        )
        foreign_patient_counts[hospital_name] = len(keys & other_keys)
        print(
            f"{hospital_name} contains a patient belonging to another hospital: "
            f"{foreign_patient_counts[hospital_name] > 0}"
        )

    server_tables = database_tables(SERVER_DATABASE)
    server_has_patient_table = any("patient" in table.lower() for table in server_tables)
    print(f"federated_server.db tables: {server_tables}")
    print(f"federated_server.db contains any patient table: {server_has_patient_table}")

    all_expected = all(actual_counts[name] == EXPECTED_COUNTS[name] for name in HOSPITAL_KEY_FILES)
    all_isolated = all(foreign_patient_counts[name] == 0 for name in HOSPITAL_KEY_FILES)
    status_pass = all_expected and all_isolated and not server_has_patient_table
    print("\nLOCAL DATABASE STATUS: PASS" if status_pass else "\nLOCAL DATABASE STATUS: FAIL")


if __name__ == "__main__":
    main()
