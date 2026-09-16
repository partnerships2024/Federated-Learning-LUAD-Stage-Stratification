"""Flask frontend foundation for the simulated federated-learning prototype."""

import csv
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for

import local_reports


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIRECTORY = PROJECT_ROOT / "databases"
RESULTS_DIRECTORY = PROJECT_ROOT / "federated_results"
METRICS_DIRECTORY = RESULTS_DIRECTORY / "metrics"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-only-local-demo-secret-key")

DEMO_USERS = {
    "hospital1": {"password": "hospital1-demo", "role": "hospital", "hospital_id": 1},
    "hospital2": {"password": "hospital2-demo", "role": "hospital", "hospital_id": 2},
    "hospital3": {"password": "hospital3-demo", "role": "hospital", "hospital_id": 3},
    "admin": {"password": "admin-demo", "role": "admin", "hospital_id": None},
}
HOSPITAL_DATABASES = {
    1: DATABASE_DIRECTORY / "hospital_1.db",
    2: DATABASE_DIRECTORY / "hospital_2.db",
    3: DATABASE_DIRECTORY / "hospital_3.db",
}
APPROVED_RESULT_IMAGES = {"federated_roc_curve.png", "federated_confusion_matrix.png"}

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
local_reports.initialize_all_report_databases()


def login_required(view):
    """Require an authenticated demo user."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def read_metric_csv(filename: str) -> list[dict[str, str]]:
    """Read a result CSV without changing it."""
    path = METRICS_DIRECTORY / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def metric_map(filename: str) -> dict[str, str]:
    """Read metric/value rows into a display mapping."""
    return {row.get("metric", ""): row.get("value", "") for row in read_metric_csv(filename)}


def role_label() -> str:
    """Return the current user's display role."""
    username = session.get("username", "")
    if username == "admin":
        return "Central Federated Server Admin"
    return f"Research Lab Client {session.get('hospital_id', '')}"


def hospital_access_required(hospital_id: int) -> None:
    """Enforce that a hospital user can access only its own dashboard."""
    if session.get("role") != "hospital" or session.get("hospital_id") != hospital_id:
        abort(403)


def server_access_required() -> None:
    """Enforce admin-only access to the central dashboard."""
    if session.get("role") != "admin":
        abort(403)


def _safe_report_payload(
    *, hospital_id: int, patient_name: str, patient_id: str, sample_id: str, cohort: str, sex: str, age: object,
    active_smoking: str, pack_years: object, normalized_genes: list[str], mutation_count: int,
    probability: float, source_type: str,
) -> dict[str, object]:
    """Build report metadata only; no sequence or model tensor enters this payload."""
    return {
        "patient_name": patient_name,
        "patient_id": patient_id or None,
        "sample_id": sample_id or None,
        "cohort": cohort or None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "research_client_id": hospital_id,
        "sex": sex,
        "age": str(age),
        "active_smoking": active_smoking,
        "pack_years": str(pack_years) if pack_years not in (None, "") else "Not provided",
        "normalized_gene_symbols": json.dumps(normalized_genes),
        "mutation_count": mutation_count,
        "predicted_group": "Advanced" if probability >= 0.5 else "Early",
        "early_probability": 1.0 - probability,
        "advanced_probability": probability,
        "threshold": 0.50,
        "model_name": "Federated Binary BiLSTM",
        "model_version": "Final 3-client FedAvg model",
        "aggregation": "FedAvg",
        "federated_rounds": 3,
        "source_type": source_type,
    }


def _report_view(report: dict[str, object]) -> dict[str, object]:
    """Convert stored JSON metadata to the shape shared by report templates."""
    view = dict(report)
    try:
        view["normalized_genes"] = json.loads(str(report.get("normalized_gene_symbols", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        view["normalized_genes"] = []
    view["sample_id"] = report.get("sample_id") or "Not provided"
    view["patient_id"] = report.get("patient_id") or "Not available"
    view["cohort"] = report.get("cohort") or "Not available"
    view["generated_at"] = report.get("created_at", "")
    view["predicted_class"] = report.get("predicted_group", "")
    view["model"] = report.get("model_name", "Federated Binary BiLSTM")
    view["architecture"] = "Bidirectional LSTM"
    view["federated_clients"] = 3
    view["federated_rounds"] = report.get("federated_rounds", 3)
    view["aggregation"] = report.get("aggregation", "FedAvg")
    view["threshold"] = report.get("threshold", 0.50)
    view["is_archived"] = bool(report.get("archive_id") or report.get("id"))
    return view


def build_round_chart_data(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    """Deduplicate repeated per-client CSV rows into one weighted point per round."""
    by_round: dict[int, dict[str, float | int]] = {}
    for row in rows:
        try:
            round_number = int(row["round"])
            if round_number not in by_round:
                by_round[round_number] = {
                    "round": round_number,
                    "roc_auc": float(row["weighted_round_validation_roc_auc"]),
                    "f1": float(row["weighted_round_validation_f1"]),
                }
        except (KeyError, TypeError, ValueError):
            continue
    return [by_round[key] for key in sorted(by_round)]


def build_comparison_chart_data(
    comparison_rows: list[dict[str, str]],
    centralized_metrics: dict[str, str],
    federated_metrics: dict[str, str],
) -> list[dict[str, str | float]]:
    """Use comparison CSV values, with saved final metric files as fallbacks."""
    comparison_by_metric = {row.get("metric", ""): row for row in comparison_rows}
    chart_rows = []
    for metric, label in (("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")):
        row = comparison_by_metric.get(metric, {})
        centralized_value = row.get("centralized") or centralized_metrics.get(metric)
        federated_value = row.get("federated") or federated_metrics.get(metric)
        try:
            chart_rows.append({
                "metric": label,
                "centralized": float(centralized_value),
                "federated": float(federated_value),
            })
        except (TypeError, ValueError):
            continue
    return chart_rows


def query_hospital_summary(hospital_id: int) -> dict[str, object]:
    """Query only the database belonging to the requested hospital."""
    database_path = HOSPITAL_DATABASES[hospital_id]
    with sqlite3.connect(database_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(patients)").fetchall()]
        count = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        early = connection.execute(
            "SELECT COUNT(*) FROM patients WHERE TARGET_STAGE_BINARY = 0"
        ).fetchone()[0]
        advanced = connection.execute(
            "SELECT COUNT(*) FROM patients WHERE TARGET_STAGE_BINARY = 1"
        ).fetchone()[0]
        total_mutations = None
        if "VALID_301_MUTATION_COUNT" in columns:
            total_mutations = connection.execute(
                "SELECT COALESCE(SUM(VALID_301_MUTATION_COUNT), 0) FROM patients"
            ).fetchone()[0]
    return {
        "patient_count": count,
        "early_count": early,
        "advanced_count": advanced,
        "total_mutations": total_mutations,
        "database_name": database_path.name,
    }


@app.route("/", methods=["GET"])
def index():
    """Route users to the correct role dashboard."""
    if "username" not in session:
        return redirect(url_for("login"))
    if session.get("role") == "admin":
        return redirect(url_for("server_dashboard"))
    return redirect(url_for("hospital_dashboard", hospital_id=session["hospital_id"]))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate one of the local prototype accounts."""
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = DEMO_USERS.get(username)
        if user and password == user["password"]:
            session.clear()
            session.update(username=username, role=user["role"], hospital_id=user["hospital_id"])
            return redirect(url_for("index"))
        error = "Invalid demo username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Clear the current session."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/hospital/<int:hospital_id>")
@login_required
def hospital_dashboard(hospital_id: int):
    """Render only the authenticated hospital's local summary."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    summary = query_hospital_summary(hospital_id)
    return render_template(
        "hospital_dashboard.html",
        hospital_id=hospital_id,
        hospital_name=f"Research Lab Client {hospital_id}",
        summary=summary,
        role_label=role_label(),
    )


def local_patient_choices(hospital_id: int) -> list[dict[str, object]]:
    """Return limited metadata from only the authenticated hospital database."""
    with sqlite3.connect(HOSPITAL_DATABASES[hospital_id]) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT COHORT, PATIENT_ID, CLINICAL_AGE_YEARS, CLINICAL_SEX_BINARY
                FROM patients
                ORDER BY COHORT, PATIENT_ID
                """
            ).fetchall()
        ]


def local_patient_record(hospital_id: int, cohort: str, patient_id: str) -> dict[str, object] | None:
    """Fetch one patient only from the requested hospital's database."""
    with sqlite3.connect(HOSPITAL_DATABASES[hospital_id]) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM patients WHERE COHORT = ? AND PATIENT_ID = ? LIMIT 1",
            (cohort, patient_id),
        ).fetchone()
    return dict(row) if row is not None else None


@app.route("/hospital/<int:hospital_id>/predict", methods=["GET", "POST"])
@login_required
def local_prediction(hospital_id: int):
    """Run the final model locally on one record belonging to this hospital."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    from local_inference import get_supported_gene_symbols

    supported_genes = get_supported_gene_symbols()
    if request.method == "GET" or not session.get("prediction_token"):
        session["prediction_token"] = secrets.token_urlsafe(16)
    choices = local_patient_choices(hospital_id)
    selected = None
    result = None
    error = None
    new_form = {"patient_name": "", "patient_id": "", "sample_id": "", "sex": "", "age": "", "active_smoking": "", "pack_years": "", "mutations": [], "error_index": None}
    if request.method == "POST":
        prediction_token = request.form.get("prediction_token", "")
        archived_predictions = session.get("archived_predictions", {})
        if prediction_token and isinstance(archived_predictions, dict):
            prior = archived_predictions.get(prediction_token)
            if isinstance(prior, dict) and prior.get("hospital_id") == hospital_id:
                return redirect(url_for("report_detail", hospital_id=hospital_id, report_id=prior["report_id"]))
        mode = request.form.get("mode", "existing")
        if mode == "new":
            genes = request.form.getlist("mutation_gene")
            sequences = request.form.getlist("mutation_sequence")
            positions = request.form.getlist("mutation_position")
            new_form = {
                "patient_name": request.form.get("patient_name", "").strip(),
                "patient_id": request.form.get("patient_id", "").strip(),
                "sample_id": request.form.get("sample_id", "").strip(),
                "sex": request.form.get("sex", ""),
                "age": request.form.get("age", "").strip(),
                "active_smoking": request.form.get("active_smoking", ""),
                "pack_years": request.form.get("pack_years", "").strip(),
                "mutations": [
                    {"gene": gene, "sequence": sequence, "position": position}
                    for gene, sequence, position in zip(genes, sequences, positions)
                ],
            }
            sex_binary = {"Female": 0, "Male": 1}.get(new_form["sex"])
            smoking_binary = {"Yes": 1, "No": 0}.get(new_form["active_smoking"])
            try:
                if not new_form["patient_name"]:
                    raise ValueError("Enter Patient Name for the research report")
                if sex_binary is None:
                    raise ValueError("Select Female or Male for sex")
                if not new_form["age"]:
                    raise ValueError("Enter age in years")
                float(new_form["age"])
                if new_form["active_smoking"] not in {"Yes", "No", "Unknown"}:
                    raise ValueError("Select Yes, No, or Unknown for active smoking")
                if new_form["pack_years"] and new_form["pack_years"].lower() != "unknown":
                    float(new_form["pack_years"])
                elif new_form["pack_years"].lower() == "unknown":
                    new_form["pack_years"] = ""
                from local_inference import predict_new_sample

                probability, unknown_genes, mutation_count, normalized_genes = predict_new_sample({
                    "sex_binary": sex_binary,
                    "age": new_form["age"],
                    "active_smoking_binary": smoking_binary,
                    "pack_years": new_form["pack_years"] or None,
                    "mutations": new_form["mutations"],
                })
                result = {
                    "predicted_class": "Advanced" if probability >= 0.5 else "Early",
                    "advanced_probability": probability,
                    "early_probability": 1.0 - probability,
                    "mutation_count": mutation_count,
                    "unknown_genes": unknown_genes,
                    "normalized_genes": normalized_genes,
                    "new_sample": True,
                }
                report_payload = _safe_report_payload(
                    hospital_id=hospital_id, patient_name=new_form["patient_name"], patient_id=new_form["patient_id"], sample_id=new_form["sample_id"], cohort="",
                    sex=new_form["sex"], age=new_form["age"], active_smoking=new_form["active_smoking"],
                    pack_years=new_form["pack_years"], normalized_genes=normalized_genes,
                    mutation_count=mutation_count, probability=probability, source_type="new_research_sample",
                )
                report_payload["archive_id"] = local_reports.save_report(hospital_id, report_payload)
                session["local_report"] = report_payload
                archived_predictions = dict(session.get("archived_predictions", {}))
                archived_predictions[prediction_token] = {"hospital_id": hospital_id, "report_id": report_payload["archive_id"]}
                session["archived_predictions"] = archived_predictions
                result["report_available"] = True
                result["archive_id"] = report_payload["archive_id"]
                new_form["mutations"] = []
            except ValueError as validation_error:
                error = str(validation_error)
                if error.startswith("Mutation "):
                    try:
                        new_form["error_index"] = int(error.split(":", 1)[0].split()[1])
                    except (IndexError, ValueError):
                        new_form["error_index"] = None
            except Exception:
                error = "New-sample analysis could not be completed. Check the input values and local model artifacts."
        else:
            cohort = request.form.get("cohort", "").strip()
            patient_id = request.form.get("patient_id", "").strip()
            selected = local_patient_record(hospital_id, cohort, patient_id)
            if selected is None:
                error = "That research record was not found in this research lab's local database."
            else:
                try:
                    from local_inference import predict_local_patient

                    probability = predict_local_patient(selected)
                    result = {
                        "predicted_class": "Advanced" if probability >= 0.5 else "Early",
                        "advanced_probability": probability,
                        "early_probability": 1.0 - probability,
                        "stored_class": "Advanced" if int(float(selected["TARGET_STAGE_BINARY"])) == 1 else "Early",
                        "report_available": True,
                    }
                    result["technical_match"] = result["predicted_class"] == result["stored_class"]
                    genes = json.loads(str(selected.get("GENE_SYMBOL_LIST", "[]")))
                    normalized_genes = [str(gene).strip().upper() for gene in genes]
                    report_payload = _safe_report_payload(
                        hospital_id=hospital_id,
                        patient_name=request.form.get("patient_name", "").strip(),
                        patient_id=str(selected.get("PATIENT_ID", "")).strip(),
                        sample_id=request.form.get("sample_id", "").strip(),
                        cohort=str(selected.get("COHORT", "")).strip(),
                        sex="Female" if int(float(selected.get("CLINICAL_SEX_BINARY", 0))) == 0 else "Male",
                        age=selected.get("CLINICAL_AGE_YEARS", "Not provided"),
                        active_smoking="Yes" if int(float(selected.get("CLINICAL_ACTIVE_SMOKING_BINARY", 0))) == 1 else "No",
                        pack_years=selected.get("CLINICAL_PACK_YEARS"), normalized_genes=normalized_genes,
                        mutation_count=len(normalized_genes), probability=probability, source_type="existing_record",
                    )
                    report_payload["archive_id"] = local_reports.save_report(hospital_id, report_payload)
                    session["local_report"] = report_payload
                    archived_predictions = dict(session.get("archived_predictions", {}))
                    archived_predictions[prediction_token] = {"hospital_id": hospital_id, "report_id": report_payload["archive_id"]}
                    session["archived_predictions"] = archived_predictions
                    result["archive_id"] = report_payload["archive_id"]
                except Exception:
                    error = "Local inference could not be completed. Check the model artifacts and local record format."
    return render_template(
        "local_prediction.html",
        hospital_id=hospital_id,
        hospital_name=f"Research Lab Client {hospital_id}",
        role_label=role_label(),
        choices=choices,
        selected=selected,
        new_form=new_form,
        supported_genes=supported_genes,
        result=result,
        error=error,
    )


@app.route("/hospital/<int:hospital_id>/report")
@login_required
def local_research_report(hospital_id: int):
    """Render the latest safe, request-local report for this research client."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    report = session.get("local_report")
    if not isinstance(report, dict):
        abort(404)
    return render_template(
        "research_report.html",
        hospital_id=hospital_id,
        hospital_name=f"Research Lab Client {hospital_id}",
        role_label=role_label(),
        report=_report_view(report),
    )


@app.route("/hospital/<int:hospital_id>/reports", methods=["GET"])
@login_required
def report_archive(hospital_id: int):
    """Show only metadata from the authenticated client's report archive."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    filters = {
        "search": request.args.get("search", ""),
        "date_from": request.args.get("date_from", ""),
        "date_to": request.args.get("date_to", ""),
    }
    return render_template(
        "report_archive.html",
        hospital_id=hospital_id,
        hospital_name=f"Research Lab Client {hospital_id}",
        role_label=role_label(),
        reports=local_reports.search_reports(hospital_id, filters["search"], filters["date_from"], filters["date_to"]),
        metrics=local_reports.get_report_stats(hospital_id),
        filters=filters,
    )


@app.route("/hospital/<int:hospital_id>/report/save", methods=["POST"])
@login_required
def save_local_report(hospital_id: int):
    """Persist the latest report summary only after explicit researcher action."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    report = session.get("local_report")
    if not isinstance(report, dict):
        abort(404)
    if report.get("archive_id"):
        return redirect(url_for("report_detail", hospital_id=hospital_id, report_id=report["archive_id"]))
    patient_name = str(report.get("patient_name", "")).strip()
    if not patient_name:
        flash("Patient Name is required before saving a research report.", "warning")
        return redirect(url_for("local_research_report", hospital_id=hospital_id))
    try:
        archive_id = local_reports.save_report(hospital_id, report)
    except Exception:
        app.logger.exception("Local report save failed for research client %s", hospital_id)
        flash("Report could not be saved. Please try again.", "warning")
        return redirect(url_for("local_research_report", hospital_id=hospital_id))
    report["archive_id"] = archive_id
    session["local_report"] = report
    flash("Report saved successfully to this research lab's local archive.", "success")
    return redirect(url_for("report_detail", hospital_id=hospital_id, report_id=archive_id))


@app.route("/hospital/<int:hospital_id>/reports/<int:report_id>")
@login_required
def report_detail(hospital_id: int, report_id: int):
    """Render stored report metadata without rerunning inference."""
    if hospital_id not in HOSPITAL_DATABASES:
        abort(404)
    hospital_access_required(hospital_id)
    report = local_reports.get_report_by_id(hospital_id, report_id)
    if report is None:
        abort(404)
    return render_template(
        "report_detail.html",
        hospital_id=hospital_id,
        hospital_name=f"Research Lab Client {hospital_id}",
        role_label=role_label(),
        report=_report_view(report),
    )


@app.route("/server")
@login_required
def server_dashboard():
    """Render central metadata and result files, never patient records."""
    server_access_required()
    with sqlite3.connect(DATABASE_DIRECTORY / "federated_server.db") as connection:
        rounds = [
            dict(zip(["round_number", "client_id", "local_sample_count", "local_loss", "local_accuracy", "update_received", "global_loss", "global_accuracy", "global_f1", "global_auc", "model_version", "timestamp"], row))
            for row in connection.execute(
                """
                SELECT round_number, client_id, local_sample_count, local_loss,
                       local_accuracy, update_received, global_loss, global_accuracy,
                       global_f1, global_auc, model_version, timestamp
                FROM federated_rounds
                WHERE id IN (
                    SELECT MAX(id)
                    FROM federated_rounds
                    GROUP BY round_number, client_id
                )
                ORDER BY round_number ASC, client_id ASC
                """
            ).fetchall()
        ]
    federated_metrics = metric_map("final_global_test_metrics.csv")
    centralized_metrics = metric_map("matched_centralized_test_metrics.csv")
    comparison = read_metric_csv("centralized_vs_federated_comparison.csv")
    round_chart_data = build_round_chart_data(read_metric_csv("federated_round_metrics.csv"))
    comparison_chart_data = build_comparison_chart_data(
        comparison, centralized_metrics, federated_metrics
    )
    confusion_rows = read_metric_csv("final_global_confusion_matrix.csv")
    confusion_matrix = {
        "early_early": "N/A",
        "early_advanced": "N/A",
        "advanced_early": "N/A",
        "advanced_advanced": "N/A",
    }
    for row in confusion_rows:
        actual = row.get("actual", "").strip().lower()
        if actual == "early":
            confusion_matrix["early_early"] = row.get("predicted_early", "N/A")
            confusion_matrix["early_advanced"] = row.get("predicted_advanced", "N/A")
        elif actual == "advanced":
            confusion_matrix["advanced_early"] = row.get("predicted_early", "N/A")
            confusion_matrix["advanced_advanced"] = row.get("predicted_advanced", "N/A")
    return render_template(
        "server_dashboard.html",
        role_label=role_label(),
        rounds=rounds,
        federated_metrics=federated_metrics,
        centralized_metrics=centralized_metrics,
        comparison=comparison,
        round_chart_data=round_chart_data,
        comparison_chart_data=comparison_chart_data,
        confusion_matrix=confusion_matrix,
        result_files=[
            "final_global_test_metrics.csv",
            "federated_round_metrics.csv",
            "centralized_vs_federated_comparison.csv",
            "matched_centralized_test_metrics.csv",
        ],
    )


@app.route("/server/result-image/<path:filename>")
@login_required
def server_result_image(filename: str):
    """Serve only approved research images to authenticated server admins."""
    server_access_required()
    if filename not in APPROVED_RESULT_IMAGES:
        abort(404)
    return send_from_directory(FIGURES_DIRECTORY, filename)


if __name__ == "__main__":
    app.run(debug=True)
