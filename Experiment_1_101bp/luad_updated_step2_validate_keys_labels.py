"""Validate LUAD labels, merge keys, and gene-panel structure.

This is a validation-only step. It does not edit source files, alter existing
datasets, calculate VAF, extract model features, or train models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "luad_processed"
REPORT_PATH = PROCESSED_DIR / "luad_key_label_validation.csv"
SUMMARY_PATH = PROCESSED_DIR / "luad_key_label_validation_summary.txt"

COHORTS = [
    "TCGA_Atlas_2018", "LUNG_MSK_2017", "MSK_NPJPO_2021", "MSKCC_2020",
    "MSKCC_2023", "NCI_2022", "ONCOSG_2020",
]

MUTATION_FILES = {
    "TCGA_Atlas_2018": "tcga_atlas_2018_data_mutations.txt",
    "LUNG_MSK_2017": "lung_msk_2017_data_mutations.txt",
    "MSK_NPJPO_2021": "msk_npjpo_2021_data_mutations.txt",
    "MSKCC_2020": "mskcc_2020_data_mutations.txt",
    "MSKCC_2023": "mskcc_2023_data_mutations.txt",
    "NCI_2022": "nci_2022_data_mutations.txt",
    "ONCOSG_2020": "oncosg_2020_data_mutations.txt",
}

CLINICAL_SAMPLE_FILES = {
    cohort: name.replace("data_mutations", "data_clinical_sample")
    for cohort, name in MUTATION_FILES.items()
}

GENE_PANEL_FILES = {
    cohort: name.replace("data_mutations", "data_gene_panel_matrix")
    for cohort, name in MUTATION_FILES.items()
}

REQUIRED_LABEL_COLUMNS = [
    "COHORT", "SAMPLE_ID", "PATIENT_ID", "STAGE_LABEL_4CLASS", "STAGE_BINARY",
]
LABEL_ALIASES = {
    "COHORT": ["COHORT"], "SAMPLE_ID": ["SAMPLE_ID", "Tumor_Sample_Barcode", "Sample_ID"],
    "PATIENT_ID": ["PATIENT_ID", "Patient_ID"],
    "STAGE_LABEL_4CLASS": ["STAGE_LABEL_4CLASS", "label_stage_4class"],
    "STAGE_BINARY": ["STAGE_BINARY", "label_stage_binary"],
}
SAMPLE_ID_ALIASES = ["Tumor_Sample_Barcode", "SAMPLE_ID", "Sample_ID", "Tumor_Sample_ID"]
MISSING = {"", "na", "n/a", "null", "none", ".", "nan"}


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | clean_series(series).str.lower().isin(MISSING)


def find_column(columns: Iterable[object], aliases: Sequence[str]) -> Optional[str]:
    lookup = {norm(column): str(column) for column in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_tsv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def candidate_paths() -> List[Path]:
    paths = list(BASE_DIR.glob("*.csv")) + list((BASE_DIR / "processed").glob("*.csv"))
    return sorted(set(path.resolve() for path in paths))


def candidate_details(path: Path, frame: pd.DataFrame) -> Dict[str, object]:
    available = [name for name in REQUIRED_LABEL_COLUMNS if find_column(frame.columns, LABEL_ALIASES[name])]
    missing = [name for name in REQUIRED_LABEL_COLUMNS if name not in available]
    cohort_col = find_column(frame.columns, LABEL_ALIASES["COHORT"])
    sample_col = find_column(frame.columns, LABEL_ALIASES["SAMPLE_ID"])
    patient_col = find_column(frame.columns, LABEL_ALIASES["PATIENT_ID"])
    duplicate_rows = 0
    unique_cohorts = unique_samples = unique_patients = 0
    if cohort_col and sample_col:
        keys = pd.DataFrame({"cohort": clean_series(frame[cohort_col]), "sample": clean_series(frame[sample_col])})
        duplicate_rows = int(keys.duplicated(keep=False).sum())
        unique_cohorts = int(keys["cohort"].nunique())
        unique_samples = int(keys["sample"].replace("", pd.NA).nunique())
    if patient_col:
        unique_patients = int(clean_series(frame[patient_col]).replace("", pd.NA).nunique())
    label_col = find_column(frame.columns, LABEL_ALIASES["STAGE_LABEL_4CLASS"])
    missing_labels = int(missing_mask(frame[label_col]).sum()) if label_col else len(frame)
    return {
        "path": path, "shape": frame.shape, "available": available, "missing": missing,
        "unique_cohorts": unique_cohorts, "unique_samples": unique_samples,
        "unique_patients": unique_patients, "duplicate_rows": duplicate_rows,
        "missing_labels": missing_labels, "sample_level": duplicate_rows == 0,
    }


def select_label_dataset(candidates: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], str]:
    valid = [item for item in candidates if not item["missing"]]
    if not valid:
        return None, "No candidate contains all required label and key columns."
    valid.sort(key=lambda item: (bool(item["sample_level"]), -int(item["missing_labels"]), int(item["shape"][0])), reverse=True)
    selected = valid[0]
    return selected, "Selected sample-level candidate with the most complete four-class labels; ties favor the largest row count."


def normalized_key_set(frame: Optional[pd.DataFrame], column: Optional[str]) -> set[str]:
    if frame is None or column is None:
        return set()
    return {norm(value) for value in clean_series(frame[column]) if norm(value)}


def merge_coverage(raw: Optional[pd.DataFrame], raw_column: Optional[str], labels: Optional[pd.DataFrame], label_column: Optional[str]) -> Tuple[int, int, int, float, List[str]]:
    raw_keys = normalized_key_set(raw, raw_column)
    label_keys = normalized_key_set(labels, label_column)
    matched = raw_keys & label_keys
    unmatched = sorted(raw_keys - label_keys)
    percent = round(100.0 * len(matched) / len(raw_keys), 2) if raw_keys else 0.0
    return len(matched), len(unmatched), len(label_keys - raw_keys), percent, unmatched[:10]


def stage_consistency(frame: pd.DataFrame, four_col: Optional[str], binary_col: Optional[str]) -> List[str]:
    if not four_col or not binary_col:
        return ["Required stage columns unavailable."]
    issues = []
    four = clean_series(frame[four_col]).str.lower()
    binary = clean_series(frame[binary_col]).str.lower()
    for index in frame.index:
        stage = four.loc[index]
        label = binary.loc[index]
        stage_match = re.search(r"\b(?:stage\s*)?(IV|III|II|I)\b", stage.upper())
        if not stage_match:
            continue
        stage_number = stage_match.group(1)
        is_early_stage = stage_number in {"I", "II"}
        is_advanced_stage = stage_number in {"III", "IV"}
        if is_early_stage and label in {"advanced", "1"}:
            issues.append(f"row {index}: {four.loc[index]} -> {binary.loc[index]}")
        if is_advanced_stage and label in {"early", "0"}:
            issues.append(f"row {index}: {four.loc[index]} -> {binary.loc[index]}")
    return issues


def panel_structure(frame: Optional[pd.DataFrame]) -> str:
    if frame is None:
        return "MISSING"
    columns = [norm(column) for column in frame.columns]
    panel_terms = ["panel", "profile", "assay", "sequencing"]
    has_panel_id = any(any(term in column for term in panel_terms) for column in columns)
    first_column_gene_like = any(term in columns[0] for term in ["gene", "hugo", "symbol"]) if columns else False
    if first_column_gene_like and frame.shape[1] > frame.shape[0]:
        orientation = "genes-as-rows, samples-as-columns"
    elif frame.shape[0] > frame.shape[1]:
        orientation = "samples-as-rows, columns-as-fields"
    else:
        orientation = "undetermined"
    return f"{orientation}; panel/profile ID column={'Yes' if has_panel_id else 'No'}; tested driver genes cannot be determined from this file alone"


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, object]] = []
    candidate_lines = ["Candidate label datasets:"]
    for path in candidate_paths():
        frame, error = read_csv(path)
        if frame is None:
            candidate_lines.append(f"- {path}: unreadable ({error})")
            continue
        details = candidate_details(path, frame)
        details["frame"] = frame
        candidates.append(details)
        candidate_lines.append(
            f"- {path} shape={details['shape']} available={details['available']} missing={details['missing']} "
            f"cohorts={details['unique_cohorts']} samples={details['unique_samples']} patients={details['unique_patients']} "
            f"duplicate_COHORT_SAMPLE={details['duplicate_rows']} sample_level={details['sample_level']}"
        )

    selected, selection_reason = select_label_dataset(candidates)
    selected_frame = selected["frame"] if selected else None
    selected_path = str(selected["path"]) if selected else "None"
    selected_columns = {name: find_column(selected_frame.columns, LABEL_ALIASES[name]) for name in LABEL_ALIASES} if selected_frame is not None else {}
    all_consistency_issues: List[str] = []
    if selected_frame is not None:
        all_consistency_issues = stage_consistency(selected_frame, selected_columns.get("STAGE_LABEL_4CLASS"), selected_columns.get("STAGE_BINARY"))

    rows: List[Dict[str, object]] = []
    merge_lines = ["Mutation and clinical-sample merge coverage:"]
    panel_lines = ["Gene-panel structure:"]
    for cohort in COHORTS:
        cohort_frame = selected_frame
        if cohort_frame is not None and selected_columns.get("COHORT"):
            cohort_frame = cohort_frame[clean_series(cohort_frame[selected_columns["COHORT"]]).str.lower() == cohort.lower()]
        total = len(cohort_frame) if cohort_frame is not None else 0
        samples = int(clean_series(cohort_frame[selected_columns["SAMPLE_ID"]]).replace("", pd.NA).nunique()) if cohort_frame is not None and selected_columns.get("SAMPLE_ID") else 0
        patients = int(clean_series(cohort_frame[selected_columns["PATIENT_ID"]]).replace("", pd.NA).nunique()) if cohort_frame is not None and selected_columns.get("PATIENT_ID") else 0
        duplicate_sample_rows = int(cohort_frame.duplicated(subset=[selected_columns["COHORT"], selected_columns["SAMPLE_ID"]], keep=False).sum()) if cohort_frame is not None and selected_columns.get("COHORT") and selected_columns.get("SAMPLE_ID") else 0
        four_col = selected_columns.get("STAGE_LABEL_4CLASS")
        binary_col = selected_columns.get("STAGE_BINARY")
        four_missing = int(missing_mask(cohort_frame[four_col]).sum()) if cohort_frame is not None and four_col else total
        binary_missing = int(missing_mask(cohort_frame[binary_col]).sum()) if cohort_frame is not None and binary_col else total

        mutation, mutation_error = read_tsv(BASE_DIR / MUTATION_FILES[cohort]) if (BASE_DIR / MUTATION_FILES[cohort]).is_file() else (None, "missing")
        clinical, clinical_error = read_tsv(BASE_DIR / CLINICAL_SAMPLE_FILES[cohort]) if (BASE_DIR / CLINICAL_SAMPLE_FILES[cohort]).is_file() else (None, "missing")
        mutation_col = find_column(mutation.columns, SAMPLE_ID_ALIASES) if mutation is not None else None
        clinical_col = find_column(clinical.columns, SAMPLE_ID_ALIASES) if clinical is not None else None
        label_col = selected_columns.get("SAMPLE_ID")
        m_count, _, _, m_pct, m_unmatched = merge_coverage(mutation, mutation_col, cohort_frame, label_col)
        c_count, _, _, c_pct, c_unmatched = merge_coverage(clinical, clinical_col, cohort_frame, label_col)
        merge_lines.append(f"- {cohort}: mutation={m_pct:.2f}% matched={m_count}; clinical_sample={c_pct:.2f}% matched={c_count}; mutation_unmatched_examples={m_unmatched}; clinical_unmatched_examples={c_unmatched}")

        panel_path = BASE_DIR / GENE_PANEL_FILES[cohort]
        panel, panel_error = read_tsv(panel_path) if panel_path.is_file() else (None, "missing")
        structure = panel_structure(panel)
        panel_lines.append(f"- {cohort}: shape={panel.shape if panel is not None else 'missing'}; columns={list(panel.columns) if panel is not None else []}; {structure}")
        notes = ""
        if mutation_error:
            notes += f"mutation: {mutation_error}; "
        if clinical_error:
            notes += f"clinical: {clinical_error}; "
        if all_consistency_issues:
            notes += f"stage inconsistencies={len(all_consistency_issues)}; "
        rows.append({
            "COHORT": cohort, "SELECTED_LABEL_FILE": selected_path, "TOTAL_ROWS": total,
            "UNIQUE_SAMPLES": samples, "UNIQUE_PATIENTS": patients,
            "DUPLICATE_SAMPLE_ROWS": duplicate_sample_rows,
            "FOUR_CLASS_LABEL_AVAILABLE": int(four_col is not None and four_missing < total),
            "BINARY_LABEL_AVAILABLE": int(binary_col is not None and binary_missing < total),
            "FOUR_CLASS_MISSING_COUNT": four_missing, "BINARY_LABEL_MISSING_COUNT": binary_missing,
            "MUTATION_SAMPLE_MATCH_COUNT": m_count, "MUTATION_SAMPLE_MATCH_PERCENT": m_pct,
            "CLINICAL_SAMPLE_MATCH_COUNT": c_count, "CLINICAL_SAMPLE_MATCH_PERCENT": c_pct,
            "GENE_PANEL_FILE_AVAILABLE": int(panel is not None), "GENE_PANEL_STRUCTURE": structure,
            "NOTES": notes,
        })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    label_coverage = selected is not None and all(row["FOUR_CLASS_LABEL_AVAILABLE"] and row["BINARY_LABEL_AVAILABLE"] for row in rows)
    merge_complete = bool(rows) and all(
        float(row["MUTATION_SAMPLE_MATCH_PERCENT"]) == 100.0
        and float(row["CLINICAL_SAMPLE_MATCH_PERCENT"]) == 100.0
        for row in rows
    )
    safe = bool(
        selected
        and selected["sample_level"]
        and label_coverage
        and not all_consistency_issues
        and all(row["DUPLICATE_SAMPLE_ROWS"] == 0 for row in rows)
        and merge_complete
    )
    summary = [
        "LUAD key and label validation summary", "", *candidate_lines, "",
        f"Selected label dataset: {selected_path}", f"Selection reason: {selection_reason}",
        f"Selected dataset is sample-level: {'Yes' if selected and selected['sample_level'] else 'No'}", "",
        "Label distribution by cohort:",
    ]
    for cohort in COHORTS:
        subset = selected_frame[clean_series(selected_frame[selected_columns["COHORT"]]).str.lower() == cohort.lower()] if selected_frame is not None and selected_columns.get("COHORT") else pd.DataFrame()
        summary.append(f"- {cohort}: four_class={subset[selected_columns['STAGE_LABEL_4CLASS']].value_counts(dropna=False).to_dict() if selected_frame is not None and selected_columns.get('STAGE_LABEL_4CLASS') else {}}; binary={subset[selected_columns['STAGE_BINARY']].value_counts(dropna=False).to_dict() if selected_frame is not None and selected_columns.get('STAGE_BINARY') else {}}")
    summary.extend(["", "Missing labels and duplicate sample rows:", report[["COHORT", "FOUR_CLASS_MISSING_COUNT", "BINARY_LABEL_MISSING_COUNT", "DUPLICATE_SAMPLE_ROWS"]].to_string(index=False), "", "Inconsistent Early/Advanced mappings:", *(f"- {issue}" for issue in (all_consistency_issues or ["None"])), "", *merge_lines, "", *panel_lines, "", f"Gene-panel profiling status can be determined: No; panel files alone do not establish gene-level tested status.", f"SAFE TO PROCEED WITH FEATURE EXTRACTION: {'Yes' if safe else 'No'}", f"FINAL DECISION: {'SAFE TO PROCEED WITH FEATURE EXTRACTION' if safe else 'NOT SAFE TO PROCEED - FIX REQUIRED'}", "", f"CSV report: {REPORT_PATH.resolve()}", f"Summary report: {SUMMARY_PATH.resolve()}"])
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Selected label dataset: {selected_path}")
    print(f"Selected dataset is sample-level: {'Yes' if selected and selected['sample_level'] else 'No'}")
    print(report.to_string(index=False))
    print(f"Inconsistent stage labels: {len(all_consistency_issues)}")
    print(f"Gene-panel profiling status can be determined: No")
    print(f"CSV report: {REPORT_PATH.resolve()}")
    print(f"Summary report: {SUMMARY_PATH.resolve()}")
    print(f"FINAL DECISION: {'SAFE TO PROCEED WITH FEATURE EXTRACTION' if safe else 'NOT SAFE TO PROCEED - FIX REQUIRED'}")
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
