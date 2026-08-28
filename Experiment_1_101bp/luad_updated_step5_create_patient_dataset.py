"""Create the final lightweight patient-level MinION prototype dataset."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OLD_PROCESSED = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "luad_processed"
WINDOW_INPUT = OUTPUT_DIR / "luad_mutation_adjusted_windows_v3.csv"
LABEL_INPUT = OLD_PROCESSED / "final_luad_multigene_sequence_training_dataset.csv"
FINAL_OUTPUT = OUTPUT_DIR / "final_luad_minion_patient_dataset.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "final_luad_minion_dataset_summary.txt"
FINAL_COLUMNS = ["COHORT", "SAMPLE_ID", "PATIENT_ID", "SEX", "sex_encoded", "mutation_window_count", "unique_mutated_gene_count", "mutated_gene_list", "mutated_dna_window_list", "STAGE_LABEL_4CLASS", "STAGE_BINARY", "label_stage_4class", "label_stage_binary"]
ALLOWED_METHODS = {"EXACT_MUTATED_WINDOW", "VERSION_NORMALISED_MUTATED_WINDOW", "SINGLE_TRANSCRIPT_GENE_FALLBACK"}
DNA_RE = re.compile(r"^[ACGTN]+$")


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def key(frame: pd.DataFrame) -> pd.Series:
    cohort = frame["COHORT"].fillna("").astype(str).str.strip().str.lower()
    sample = frame["SAMPLE_ID"].fillna("").astype(str).str.strip().map(norm)
    return cohort + "\x1f" + sample


def fail(message: str) -> int:
    print("FINAL DATASET NOT CREATED - VALIDATION FAILED")
    print(f"ERROR: {message}")
    return 1


def stage_is_consistent(four_class: object, binary: object) -> bool:
    stage = clean(four_class).lower()
    label = clean(binary).lower()
    if not stage or not label:
        return True
    match = re.search(r"\b(?:stage\s*)?(IV|III|II|I)\b", stage.upper())
    if not match:
        return True
    early = match.group(1) in {"I", "II"}
    return (early and label in {"early", "0"}) or (not early and label in {"advanced", "1"})


def json_list(value: object) -> Optional[List[object]]:
    try:
        parsed = json.loads(clean(value))
        return parsed if isinstance(parsed, list) else None
    except (TypeError, json.JSONDecodeError):
        return None


def distribution(frame: pd.DataFrame, column: str) -> str:
    return json.dumps({str(k): int(v) for k, v in frame[column].fillna("<MISSING>").value_counts(dropna=False).to_dict().items()}, ensure_ascii=True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if FINAL_OUTPUT.exists() or SUMMARY_OUTPUT.exists():
        return fail("One or both final output files already exist; refusing to overwrite them.")
    if not WINDOW_INPUT.is_file() or not LABEL_INPUT.is_file():
        return fail(f"Required input missing: {WINDOW_INPUT if not WINDOW_INPUT.is_file() else LABEL_INPUT}")

    windows = pd.read_csv(WINDOW_INPUT, dtype=str, low_memory=False)
    labels = pd.read_csv(LABEL_INPUT, dtype=str, low_memory=False)
    required_windows = ["COHORT", "SAMPLE_ID", "PATIENT_ID", "SEX", "sex_encoded", "GENE_SYMBOL", "MUTATED_DNA_WINDOW", "MUTATED_WINDOW_VALID", "MUTATION_APPLIED", "SEQUENCE_METHOD"]
    required_labels = ["COHORT", "SAMPLE_ID", "PATIENT_ID", "SEX", "sex_encoded", "STAGE_LABEL_4CLASS", "STAGE_BINARY", "label_stage_4class", "label_stage_binary"]
    missing_windows = [column for column in required_windows if column not in windows.columns]
    missing_labels = [column for column in required_labels if column not in labels.columns]
    if missing_windows or missing_labels:
        return fail(f"Missing required columns. Mutation file: {missing_windows}; label file: {missing_labels}")
    if labels.duplicated(["COHORT", "SAMPLE_ID"]).any():
        return fail("Authoritative label dataset has duplicate COHORT + SAMPLE_ID keys.")

    total_rows = len(windows)
    windows["_KEY"] = key(windows)
    labels["_KEY"] = key(labels)
    label_lookup = labels.set_index("_KEY")
    valid_window = windows["MUTATED_DNA_WINDOW"].map(clean).str.len().eq(101) & windows["MUTATED_DNA_WINDOW"].map(clean).map(lambda value: bool(DNA_RE.fullmatch(value)))
    valid_window &= windows["MUTATED_WINDOW_VALID"].map(clean).str.lower().isin({"1", "true", "yes"})
    valid_window &= windows["MUTATION_APPLIED"].map(clean).str.lower().isin({"1", "true", "yes"})
    valid_window &= windows["SEQUENCE_METHOD"].map(clean).isin(ALLOWED_METHODS)
    valid_window &= windows["GENE_SYMBOL"].map(clean).ne("")
    for column in ["COHORT", "SAMPLE_ID", "PATIENT_ID"]:
        valid_window &= windows[column].map(clean).ne("")
    retained = windows.loc[valid_window].copy()
    excluded_rows = total_rows - len(retained)
    if retained.empty:
        return fail("No valid mutation-adjusted windows remained after filtering.")

    unknown_keys = sorted(set(retained["_KEY"]) - set(label_lookup.index))
    if unknown_keys:
        return fail(f"{len(unknown_keys)} retained mutation rows have no authoritative label key.")
    for label_column in ["PATIENT_ID", "SEX", "sex_encoded", "STAGE_LABEL_4CLASS", "STAGE_BINARY", "label_stage_4class", "label_stage_binary"]:
        authoritative = retained["_KEY"].map(label_lookup[label_column].to_dict()).map(clean)
        original = retained[label_column].map(clean) if label_column in retained.columns else pd.Series("", index=retained.index)
        conflict = (original.ne("") & authoritative.ne("") & original.map(norm).ne(authoritative.map(norm)))
        if conflict.any():
            return fail(f"Conflicting {label_column} values between mutation and authoritative label files: {retained.loc[conflict, ['COHORT', 'SAMPLE_ID', label_column]].head(10).to_dict('records')}")
        retained[label_column] = authoritative

    if not retained.apply(lambda row: stage_is_consistent(row["STAGE_LABEL_4CLASS"], row["STAGE_BINARY"]), axis=1).all():
        return fail("Stage I/II versus Early or Stage III/IV versus Advanced mapping is inconsistent.")
    if not labels.apply(lambda row: stage_is_consistent(row["STAGE_LABEL_4CLASS"], row["STAGE_BINARY"]), axis=1).all():
        return fail("Authoritative label file contains inconsistent stage mappings.")

    # Check patient-level conflicts across all labelled samples before selection.
    patient_groups = labels.groupby([labels["COHORT"].map(norm), labels["PATIENT_ID"].map(norm)], dropna=False)
    for patient_key, group in patient_groups:
        for column in ["STAGE_LABEL_4CLASS", "STAGE_BINARY", "label_stage_4class", "label_stage_binary", "SEX", "sex_encoded"]:
            values = {norm(value) for value in group[column] if clean(value)}
            if len(values) > 1:
                return fail(f"Conflicting {column} values for patient {patient_key}: {sorted(values)}")

    identity = ["COHORT", "SAMPLE_ID", "PATIENT_ID", "GENE_SYMBOL", "TRANSCRIPT_ID_MATCHED", "HGVSc_CLEAN", "MUTATED_DNA_WINDOW"]
    identity = [column for column in identity if column in retained.columns]
    duplicate_count = int(retained.duplicated(identity, keep="first").sum())
    retained = retained.drop_duplicates(identity, keep="first").copy()
    sort_columns = [column for column in ["GENE_SYMBOL", "TRANSCRIPT_ID_MATCHED", "HGVSc_CLEAN", "MUTATED_DNA_WINDOW"] if column in retained.columns]
    retained = retained.sort_values(["COHORT", "SAMPLE_ID"] + sort_columns, kind="stable")

    sample_rows: List[Dict[str, object]] = []
    for (cohort, sample_id), group in retained.groupby(["COHORT", "SAMPLE_ID"], sort=False):
        genes = [clean(value) for value in group["GENE_SYMBOL"]]
        sequences = [clean(value) for value in group["MUTATED_DNA_WINDOW"]]
        sample_rows.append({"COHORT": cohort, "SAMPLE_ID": sample_id, "PATIENT_ID": clean(group["PATIENT_ID"].iloc[0]), "SEX": clean(group["SEX"].iloc[0]), "sex_encoded": clean(group["sex_encoded"].iloc[0]), "mutation_window_count": len(sequences), "unique_mutated_gene_count": len(set(genes)), "mutated_gene_list": json.dumps(genes), "mutated_dna_window_list": json.dumps(sequences), "STAGE_LABEL_4CLASS": clean(group["STAGE_LABEL_4CLASS"].iloc[0]), "STAGE_BINARY": clean(group["STAGE_BINARY"].iloc[0]), "label_stage_4class": clean(group["label_stage_4class"].iloc[0]), "label_stage_binary": clean(group["label_stage_binary"].iloc[0])})
    samples = pd.DataFrame(sample_rows)

    all_label_patients = labels.assign(_PATIENT=labels["PATIENT_ID"].map(norm)).groupby([labels["COHORT"].map(norm), "_PATIENT"]).size()
    valid_patients = samples.assign(_PATIENT=samples["PATIENT_ID"].map(norm)).groupby([samples["COHORT"].map(norm), "_PATIENT"]).size()
    multiple_sample_groups = samples.groupby([samples["COHORT"].map(norm), samples["PATIENT_ID"].map(norm)])
    multiple_sample_count = int(sum(len(group) > 1 for _, group in multiple_sample_groups))
    representative_rows: List[pd.Series] = []
    removed_samples = 0
    for _, group in multiple_sample_groups:
        ranked = group.sort_values(["mutation_window_count", "unique_mutated_gene_count", "SAMPLE_ID"], ascending=[False, False, True], kind="stable")
        representative_rows.append(ranked.iloc[0]); removed_samples += len(group) - 1
    patient = pd.DataFrame(representative_rows).reset_index(drop=True)
    patient = patient[FINAL_COLUMNS]

    missing_final = [column for column in ["COHORT", "SAMPLE_ID", "PATIENT_ID", "sex_encoded", "STAGE_BINARY", "STAGE_LABEL_4CLASS"] if patient[column].map(clean).eq("").any()]
    if missing_final:
        return fail(f"Final dataset has missing required values in: {missing_final}")
    if patient.duplicated(["COHORT", "PATIENT_ID"]).any():
        return fail("Final dataset contains duplicate COHORT + PATIENT_ID keys.")
    for _, row in patient.iterrows():
        genes, sequences = json_list(row["mutated_gene_list"]), json_list(row["mutated_dna_window_list"])
        if genes is None or sequences is None or len(genes) != len(sequences) or len(sequences) != int(row["mutation_window_count"]):
            return fail(f"Gene/window JSON alignment failed for {row['COHORT']} / {row['PATIENT_ID']}.")
        if len(set(genes)) != int(row["unique_mutated_gene_count"]) or any(len(str(sequence)) != 101 or not DNA_RE.fullmatch(str(sequence)) for sequence in sequences):
            return fail(f"Window or gene-count validation failed for {row['COHORT']} / {row['PATIENT_ID']}.")

    patient["mutation_window_count"] = patient["mutation_window_count"].astype(int)
    patient["unique_mutated_gene_count"] = patient["unique_mutated_gene_count"].astype(int)
    patient.to_csv(FINAL_OUTPUT, index=False)
    windows_per_patient = patient["mutation_window_count"]
    summary = [
        "Final LUAD MinION patient dataset summary", f"Input mutation-level file: {WINDOW_INPUT.resolve()}", f"Input labelled dataset: {LABEL_INPUT.resolve()}", f"Total mutation rows read: {total_rows}", f"Valid mutation-adjusted rows retained: {len(retained)}", f"Excluded fallback or failed rows: {excluded_rows}", f"Exact duplicates removed: {duplicate_count}", f"Samples with at least one valid window: {len(samples)}", f"Patients with at least one valid window: {len(patient)}", f"Patients excluded with no valid window: {len(set(all_label_patients.index) - set(valid_patients.index))}", f"Patients with multiple valid-window samples: {multiple_sample_count}", f"Samples removed by representative selection: {removed_samples}", "Representative rule: highest valid-window count, then highest unique-gene count, then alphabetically first SAMPLE_ID.", f"Final dataset shape: {patient.shape}", f"Final columns: {list(patient.columns)}", f"Cohort distribution: {distribution(patient, 'COHORT')}", f"Sex distribution: {distribution(patient, 'SEX')}", f"Binary-stage distribution: {distribution(patient, 'STAGE_BINARY')}", f"Four-class-stage distribution: {distribution(patient, 'STAGE_LABEL_4CLASS')}", f"Mutation-window statistics: min={windows_per_patient.min()}, mean={windows_per_patient.mean():.2f}, median={windows_per_patient.median():.2f}, 75th={windows_per_patient.quantile(.75):.2f}, 90th={windows_per_patient.quantile(.90):.2f}, 95th={windows_per_patient.quantile(.95):.2f}, max={windows_per_patient.max()}", f"Unique-gene statistics: min={patient['unique_mutated_gene_count'].min()}, mean={patient['unique_mutated_gene_count'].mean():.2f}, median={patient['unique_mutated_gene_count'].median():.2f}, max={patient['unique_mutated_gene_count'].max()}", "DNA-window length validation: PASS", "DNA alphabet validation: PASS", "Gene/window list alignment validation: PASS", "One row per COHORT + PATIENT_ID: PASS", "No clinical or CNA features were added.", "Future model inputs: mutated_gene_list, mutated_dna_window_list, sex_encoded.", "Model targets: STAGE_BINARY and STAGE_LABEL_4CLASS.", "Safe for model preparation: YES", f"Final dataset: {FINAL_OUTPUT.resolve()}", f"Summary: {SUMMARY_OUTPUT.resolve()}",
    ]
    SUMMARY_OUTPUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Execution status: SUCCESS\nTotal mutation rows read: {total_rows}\nValid mutation rows retained: {len(retained)}\nUnique valid samples: {len(samples)}\nFinal unique patients: {len(patient)}\nPatients excluded due to no valid sequence: {len(set(all_label_patients.index) - set(valid_patients.index))}\nMultiple-sample patients: {multiple_sample_count}\nFinal dataset shape: {patient.shape}\nCohort distribution: {distribution(patient, 'COHORT')}\nBinary-stage distribution: {distribution(patient, 'STAGE_BINARY')}\nFour-class-stage distribution: {distribution(patient, 'STAGE_LABEL_4CLASS')}\nMedian mutation windows per patient: {windows_per_patient.median():.2f}\n90th percentile mutation windows per patient: {windows_per_patient.quantile(.90):.2f}\nMaximum mutation windows per patient: {windows_per_patient.max()}\nDNA-window validation: PASS\nList-alignment validation: PASS\nFINAL PATIENT DATASET CREATED SUCCESSFULLY\nDataset: {FINAL_OUTPUT.resolve()}\nSummary: {SUMMARY_OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
