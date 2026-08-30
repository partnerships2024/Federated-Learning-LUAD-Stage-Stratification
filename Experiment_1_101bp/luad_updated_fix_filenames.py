"""Safely fix LUAD source filenames and rerun feature inspection.

Only filenames are changed. File contents are never read for rewriting and an
existing target is never overwritten. The old ``processed`` directory is not
used; the log is written to ``luad_processed``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "luad_processed"
LOG_PATH = OUTPUT_DIR / "filename_fix_log.txt"

TCGA_RENAMES = {
    "data_mutations.txt": "tcga_atlas_2018_data_mutations.txt",
    "data_clinical_patient.txt": "tcga_atlas_2018_data_clinical_patient.txt",
    "data_clinical_sample.txt": "tcga_atlas_2018_data_clinical_sample.txt",
    "data_cna.txt": "tcga_atlas_2018_data_cna.txt",
    "data_gene_panel_matrix.txt": "tcga_atlas_2018_data_gene_panel_matrix.txt",
}

OTHER_RENAMES = {
    "oncosg_2020_data_mutation.txt": "oncosg_2020_data_mutations.txt",
    "nci_2022_data_gene_panel_matrix.txt.txt": "nci_2022_data_gene_panel_matrix.txt",
    "mskcc_2023_data_gene_panel_matrix.txt.txt": "mskcc_2023_data_gene_panel_matrix.txt",
}

EXPECTED_FILES: Dict[str, List[str]] = {
    "TCGA_Atlas_2018": [
        "tcga_atlas_2018_data_mutations.txt",
        "tcga_atlas_2018_data_clinical_patient.txt",
        "tcga_atlas_2018_data_clinical_sample.txt",
        "tcga_atlas_2018_data_cna.txt",
        "tcga_atlas_2018_data_gene_panel_matrix.txt",
    ],
    "LUNG_MSK_2017": [
        "lung_msk_2017_data_mutations.txt", "lung_msk_2017_data_clinical_patient.txt",
        "lung_msk_2017_data_clinical_sample.txt", "lung_msk_2017_data_cna.txt",
        "lung_msk_2017_data_gene_panel_matrix.txt",
    ],
    "MSK_NPJPO_2021": [
        "msk_npjpo_2021_data_mutations.txt", "msk_npjpo_2021_data_clinical_patient.txt",
        "msk_npjpo_2021_data_clinical_sample.txt", "msk_npjpo_2021_data_cna.txt",
        "msk_npjpo_2021_data_gene_panel_matrix.txt",
    ],
    "MSKCC_2020": [
        "mskcc_2020_data_mutations.txt", "mskcc_2020_data_clinical_patient.txt",
        "mskcc_2020_data_clinical_sample.txt", "mskcc_2020_data_cna.txt",
        "mskcc_2020_data_gene_panel_matrix.txt",
    ],
    "MSKCC_2023": [
        "mskcc_2023_data_mutations.txt", "mskcc_2023_data_clinical_patient.txt",
        "mskcc_2023_data_clinical_sample.txt", "mskcc_2023_data_cna.txt",
        "mskcc_2023_data_gene_panel_matrix.txt",
    ],
    "NCI_2022": [
        "nci_2022_data_mutations.txt", "nci_2022_data_clinical_patient.txt",
        "nci_2022_data_clinical_sample.txt", "nci_2022_data_gene_panel_matrix.txt",
    ],
    "ONCOSG_2020": [
        "oncosg_2020_data_mutations.txt", "oncosg_2020_data_clinical_patient.txt",
        "oncosg_2020_data_clinical_sample.txt", "oncosg_2020_data_cna.txt",
        "oncosg_2020_data_gene_panel_matrix.txt",
    ],
}


def safe_rename(source_name: str, target_name: str, renamed: List[str], correct: List[str], duplicates: List[str], missing: List[str], skipped: List[str]) -> None:
    source = BASE_DIR / source_name
    target = BASE_DIR / target_name
    if source.is_file() and target.is_file():
        message = f"{source_name} and {target_name} both exist; neither changed."
        duplicates.append(message)
        skipped.append(message)
        return
    if target.is_file():
        correct.append(target_name)
        skipped.append(f"{source_name} skipped because target {target_name} already exists.")
        return
    if not source.is_file():
        missing.append(source_name)
        return
    try:
        source.rename(target)
        renamed.append(f"{source_name} -> {target_name}")
    except OSError as exc:
        skipped.append(f"{source_name} -> {target_name}: rename failed: {exc}")


def verify_expected() -> List[Tuple[str, str, bool, str]]:
    rows = []
    for cohort, names in EXPECTED_FILES.items():
        for name in names:
            rows.append((cohort, name, (BASE_DIR / name).is_file(), ""))
    # NCI has no CNA source by design and is intentionally absent from its list.
    return rows


def format_table(rows: List[Tuple[str, str, bool, str]]) -> str:
    lines = ["COHORT | EXPECTED FILE | AVAILABLE"]
    for cohort, name, available, _ in rows:
        lines.append(f"{cohort} | {name} | {'Yes' if available else 'No'}")
    lines.append("NCI_2022 | nci_2022_data_cna.txt | No (genuinely unavailable)")
    return "\n".join(lines)


def run_inspection() -> Tuple[str, int, str]:
    requested = BASE_DIR / "luad_updated.py"
    fallback = BASE_DIR / "inspect_luad_feature_availability.py"
    script = requested if requested.is_file() else fallback
    if not script.is_file():
        return "No feature-inspection script found", 1, ""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    mode = script.name if script == requested else f"{script.name} (luad_updated.py was absent)"
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    return mode, result.returncode, output


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renamed: List[str] = []
    correct: List[str] = []
    missing: List[str] = []
    duplicates: List[str] = []
    skipped: List[str] = []

    for source, target in {**TCGA_RENAMES, **OTHER_RENAMES}.items():
        safe_rename(source, target, renamed, correct, duplicates, missing, skipped)

    expected_rows = verify_expected()
    inspection_name, inspection_code, inspection_output = run_inspection()
    log_lines = [
        "LUAD filename correction log",
        f"Project: {BASE_DIR}",
        "",
        "Files successfully renamed:",
        *(f"- {item}" for item in (renamed or ["None"])),
        "",
        "Files already correctly named:",
        *(f"- {item}" for item in (correct or ["None"])),
        "",
        "Duplicate files:",
        *(f"- {item}" for item in (duplicates or ["None"])),
        "",
        "Missing files:",
        *(f"- {item}" for item in (missing or ["None"])),
        "",
        "Files skipped to avoid overwrite:",
        *(f"- {item}" for item in (skipped or ["None"])),
        "",
        "Source contents modified: No. Only filesystem filenames were changed.",
        "",
        "Final expected-file availability:",
        format_table(expected_rows),
        "",
        f"Feature inspection executed: {inspection_name}",
        f"Feature inspection exit code: {inspection_code}",
        f"Feature inspection succeeded: {'Yes' if inspection_code == 0 else 'No'}",
    ]
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    if inspection_output:
        print("\nFEATURE INSPECTION OUTPUT\n" + inspection_output)
    print(f"\nFilename-fix log: {LOG_PATH.resolve()}")
    return 0 if inspection_code == 0 else inspection_code


if __name__ == "__main__":
    raise SystemExit(main())
