"""The single local-only persistence layer for research report metadata."""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
REPORT_DB_PATHS = {
    1: BASE_DIR / "databases" / "lab_1_reports.db",
    2: BASE_DIR / "databases" / "lab_2_reports.db",
    3: BASE_DIR / "databases" / "lab_3_reports.db",
}

REPORT_FIELDS = (
    "patient_name", "patient_id", "sample_id", "cohort", "created_at", "research_client_id",
    "sex", "age", "active_smoking", "pack_years", "normalized_gene_symbols", "mutation_count",
    "predicted_group", "early_probability", "advanced_probability", "threshold", "model_name",
    "model_version", "aggregation", "federated_rounds", "source_type",
)
REPORT_SCHEMA = {
    "patient_name": "TEXT NOT NULL",
    "patient_id": "TEXT",
    "sample_id": "TEXT",
    "cohort": "TEXT",
    "created_at": "TEXT NOT NULL",
    "research_client_id": "INTEGER NOT NULL",
    "sex": "TEXT",
    "age": "REAL",
    "active_smoking": "TEXT",
    "pack_years": "REAL",
    "normalized_gene_symbols": "TEXT",
    "mutation_count": "INTEGER",
    "predicted_group": "TEXT",
    "early_probability": "REAL",
    "advanced_probability": "REAL",
    "threshold": "REAL",
    "model_name": "TEXT",
    "model_version": "TEXT",
    "aggregation": "TEXT",
    "federated_rounds": "INTEGER",
    "source_type": "TEXT",
}


def get_report_db_path(client_id: int) -> Path:
    """Resolve the one hard-coded report database assigned to a client."""
    try:
        return REPORT_DB_PATHS[int(client_id)].resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unknown research client: {client_id}") from error


def init_report_db(client_id: int) -> Path:
    """Create/migrate only the requested client's report database."""
    path = get_report_db_path(client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                patient_id TEXT,
                sample_id TEXT,
                cohort TEXT,
                created_at TEXT NOT NULL,
                research_client_id INTEGER NOT NULL,
                sex TEXT,
                age REAL,
                active_smoking TEXT,
                pack_years REAL,
                normalized_gene_symbols TEXT,
                mutation_count INTEGER,
                predicted_group TEXT,
                early_probability REAL,
                advanced_probability REAL,
                threshold REAL,
                model_name TEXT,
                model_version TEXT,
                aggregation TEXT,
                federated_rounds INTEGER,
                source_type TEXT
            )
            """
        )
        existing = {row[1] for row in connection.execute("PRAGMA table_info(research_reports)")}
        for column, definition in REPORT_SCHEMA.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE research_reports ADD COLUMN {column} {definition}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_reports_created ON research_reports(created_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_reports_patient ON research_reports(patient_name COLLATE NOCASE)")
        connection.commit()
    except Exception:
        connection.rollback()
        logger.exception("Report database initialization failed for client %s", client_id)
        raise
    finally:
        connection.close()
    return path


def initialize_all_report_databases() -> None:
    """Initialize each isolated report database without using a shared database."""
    for client_id in REPORT_DB_PATHS:
        init_report_db(client_id)


def save_report(client_id: int, report_data: dict[str, object]) -> int:
    """Insert one safe metadata report, commit it, and verify its row ID."""
    path = get_report_db_path(client_id)
    init_report_db(client_id)
    values = [client_id if field == "research_client_id" else report_data.get(field) for field in REPORT_FIELDS]
    connection = sqlite3.connect(path)
    try:
        placeholders = ", ".join("?" for _ in REPORT_FIELDS)
        cursor = connection.execute(
            f"INSERT INTO research_reports ({', '.join(REPORT_FIELDS)}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
        report_id = int(cursor.lastrowid)
        verified = connection.execute(
            "SELECT id FROM research_reports WHERE id = ? AND research_client_id = ?",
            (report_id, client_id),
        ).fetchone()
        if verified is None:
            raise RuntimeError(f"Committed report row {report_id} could not be verified")
        return report_id
    except Exception:
        connection.rollback()
        logger.exception("Report insert failed for client %s at %s", client_id, path)
        raise
    finally:
        connection.close()


def get_report_stats(client_id: int) -> dict[str, int]:
    path = get_report_db_path(client_id)
    init_report_db(client_id)
    month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
    connection = sqlite3.connect(path)
    try:
        total, early, advanced, month = connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN predicted_group = 'Early' THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN predicted_group = 'Advanced' THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN substr(created_at, 1, 7) = ? THEN 1 ELSE 0 END), 0)
            FROM research_reports
            """, (month_prefix,),
        ).fetchone()
        return {"total": int(total), "early": int(early), "advanced": int(advanced), "month": int(month)}
    finally:
        connection.close()


def search_reports(client_id: int, patient_name: str = "", date_from: str | None = None, date_to: str | None = None) -> list[dict[str, object]]:
    path = get_report_db_path(client_id)
    init_report_db(client_id)
    clauses = ["research_client_id = ?"]
    parameters: list[object] = [client_id]
    search = (patient_name or "").strip()
    if search:
        clauses.append("LOWER(patient_name) LIKE LOWER(?)")
        parameters.append(f"%{search}%")
    if date_from and str(date_from).strip():
        clauses.append("substr(created_at, 1, 10) >= ?")
        parameters.append(str(date_from).strip())
    if date_to and str(date_to).strip():
        clauses.append("substr(created_at, 1, 10) <= ?")
        parameters.append(str(date_to).strip())
    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM research_reports WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_report_by_id(client_id: int, report_id: int) -> dict[str, object] | None:
    path = get_report_db_path(client_id)
    init_report_db(client_id)
    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM research_reports WHERE id = ? AND research_client_id = ?",
            (report_id, client_id),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def delete_report_for_test_only(client_id: int, report_id: int) -> None:
    """Delete only a caller-specified synthetic verification row."""
    path = get_report_db_path(client_id)
    init_report_db(client_id)
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM research_reports WHERE id = ?", (report_id,))
        connection.commit()
    except Exception:
        connection.rollback()
        logger.exception("Test report cleanup failed for client %s", client_id)
        raise
    finally:
        connection.close()
