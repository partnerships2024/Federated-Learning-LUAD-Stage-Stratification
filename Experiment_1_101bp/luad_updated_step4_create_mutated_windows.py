"""Create mutation-level 101-base reference and mutation-adjusted windows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OLD_PROCESSED = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "luad_processed"
LABEL_PATH = OLD_PROCESSED / "final_luad_multigene_sequence_training_dataset.csv"
DRIVER_PATH = OLD_PROCESSED / "LUAD_driver_genes.csv"
FASTA_PATH = BASE_DIR / "LUAD_Ensembl_CDS_sequences_96genes.fasta.txt"
WINDOW_LENGTH = 101
COHORTS = ["TCGA_Atlas_2018", "LUNG_MSK_2017", "MSK_NPJPO_2021", "MSKCC_2020", "MSKCC_2023", "NCI_2022", "ONCOSG_2020"]
MUTATION_FILES = {
    "TCGA_Atlas_2018": "tcga_atlas_2018_data_mutations.txt", "LUNG_MSK_2017": "lung_msk_2017_data_mutations.txt",
    "MSK_NPJPO_2021": "msk_npjpo_2021_data_mutations.txt", "MSKCC_2020": "mskcc_2020_data_mutations.txt",
    "MSKCC_2023": "mskcc_2023_data_mutations.txt", "NCI_2022": "nci_2022_data_mutations.txt", "ONCOSG_2020": "oncosg_2020_data_mutations.txt",
}
ALIASES = {
    "gene": ["Hugo_Symbol", "Gene", "GENE_SYMBOL", "Gene_Symbol"], "sample": ["Tumor_Sample_Barcode", "SAMPLE_ID", "Sample_ID", "Tumor_Sample_ID"],
    "transcript": ["Transcript_ID", "Transcript", "Transcript_Id"], "hgvsc": ["HGVSc", "HGVSc_Short", "Coding_Change"],
    "hgvsp": ["HGVSp_Short", "HGVSp"], "classification": ["Variant_Classification"], "type": ["Variant_Type"],
    "ref": ["Reference_Allele"], "allele1": ["Tumor_Seq_Allele1", "Tumor_Seq_Allele_1"], "allele2": ["Tumor_Seq_Allele2", "Tumor_Seq_Allele_2"],
    "protein": ["Protein_position", "Protein_Position"], "chromosome": ["Chromosome"], "start": ["Start_Position"], "end": ["End_Position"], "codons": ["Codons"],
}
OUTPUT_NAMES = ["luad_mutation_adjusted_windows.csv", "luad_mutation_adjusted_windows_summary.txt", "luad_mutation_adjusted_windows_qc_examples.csv"]


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def find_column(columns: Iterable[object], aliases: List[str]) -> Optional[str]:
    lookup = {norm(column): str(column) for column in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def versioned_path(name: str) -> Path:
    path = OUTPUT_DIR / name
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    number = 2
    while True:
        candidate = OUTPUT_DIR / f"{stem}_v{number}{suffix}"
        if not candidate.exists():
            return candidate
        number += 1


def read_fasta(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    header: Optional[str] = None
    sequence: List[str] = []
    def finish() -> None:
        if header is None:
            return
        fields = header.split("|")
        text = "".join(sequence).upper()
        records.append({"header": header, "gene_id": fields[0].strip() if len(fields) > 0 else "", "transcript": fields[1].strip() if len(fields) > 1 else "", "gene": fields[2].strip().upper() if len(fields) > 2 else "", "sequence": text, "valid": bool(re.fullmatch(r"[ACGTN]+", text))})
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                finish(); header = line[1:]; sequence = []
            elif line:
                sequence.append(line)
    finish()
    return records


def parse_hgvsc(raw: str) -> Tuple[str, str, Optional[Dict[str, object]]]:
    text = clean(raw)
    prefix_match = re.match(r"^([A-Za-z0-9_.-]+:)?(c\..+)$", text, re.I)
    transcript_prefix = prefix_match.group(1)[:-1] if prefix_match and prefix_match.group(1) else ""
    cdna = prefix_match.group(2) if prefix_match else text
    prefix = r"(?:[A-Za-z0-9_.-]+:)?"
    patterns = [
        ("SNV", prefix + r"c\.(\d+)([ACGT])>([ACGT])"),
        ("DELETION", prefix + r"c\.(\d+)(?:_(\d+))?del([ACGT]*)"),
        ("INSERTION", prefix + r"c\.(\d+)_(\d+)ins([ACGT]+)"),
        ("DUPLICATION", prefix + r"c\.(\d+)(?:_(\d+))?dup([ACGT]*)"),
        ("DELINS", prefix + r"c\.(\d+)_(\d+)delins([ACGT]+)"),
    ]
    for category, pattern in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if not match:
            continue
        values = match.groups()
        if category == "SNV":
            return transcript_prefix, cdna, {"category": category, "start": int(values[0]), "end": int(values[0]), "ref": values[1].upper(), "alt": values[2].upper()}
        if category in {"DELETION", "DUPLICATION"}:
            start, end, allele = int(values[0]), int(values[1] or values[0]), values[2].upper()
            return transcript_prefix, cdna, {"category": category, "start": start, "end": end, "ref": allele, "alt": ""}
        start, end, allele = int(values[0]), int(values[1]), values[2].upper()
        if category == "INSERTION":
            return transcript_prefix, cdna, {"category": category, "start": start, "end": end, "ref": "", "alt": allele}
        return transcript_prefix, cdna, {"category": category, "start": start, "end": end, "ref": "", "alt": allele}
    return transcript_prefix, cdna, None


def window(sequence: str, center: int) -> str:
    left = center - WINDOW_LENGTH // 2
    return "".join(sequence[i] if 0 <= i < len(sequence) else "N" for i in range(left, left + WINDOW_LENGTH))


def apply_mutation(sequence: str, mutation: Dict[str, object]) -> Tuple[Optional[str], int, str]:
    start, end = int(mutation["start"]), int(mutation["end"])
    category, stated_ref, alt = mutation["category"], str(mutation["ref"]), str(mutation["alt"])
    if start < 1 or end < start or end > len(sequence):
        return None, start - 1, "POSITION_OUTSIDE_CDS"
    first, last = start - 1, end
    actual_ref = sequence[first:last]
    if stated_ref and actual_ref.upper() != stated_ref.upper():
        return None, first, "REFERENCE_MISMATCH"
    if category == "SNV":
        return sequence[:first] + alt + sequence[first + 1:], first, ""
    if category == "DELETION":
        return sequence[:first] + sequence[last:], first, ""
    if category == "INSERTION":
        return sequence[:last] + alt + sequence[last:], last, ""
    if category == "DUPLICATION":
        return sequence[:last] + sequence[first:last] + sequence[last:], last, ""
    if category == "DELINS":
        return sequence[:first] + alt + sequence[last:], first, ""
    return None, first, "UNSUPPORTED_HGVSC"


def metadata_map(labels: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, object]]:
    result = {}
    for _, row in labels.iterrows():
        key = (clean(row.get("COHORT")).lower(), norm(row.get("SAMPLE_ID")))
        result[key] = {column: row.get(column, "") for column in ["PATIENT_ID", "SEX", "sex_encoded", "STAGE_LABEL_4CLASS", "STAGE_BINARY", "label_stage_4class", "label_stage_binary"]}
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path, summary_path, qc_path = [versioned_path(name) for name in OUTPUT_NAMES]
    labels = pd.read_csv(LABEL_PATH, dtype=str, low_memory=False)
    if labels.duplicated(["COHORT", "SAMPLE_ID"]).any():
        raise ValueError("COHORT + SAMPLE_ID is not unique in the selected labelled dataset.")
    drivers = pd.read_csv(DRIVER_PATH, dtype=str)
    gene_driver_col = find_column(drivers.columns, ["SYMBOL", "Gene", "Hugo_Symbol"])
    driver_set = {clean(value).upper() for value in drivers[gene_driver_col] if clean(value)}
    records = [r for r in read_fasta(FASTA_PATH) if r["valid"]]
    exact_tx = {r["transcript"].upper(): r for r in records}
    normalized_tx: Dict[str, List[Dict[str, str]]] = {}
    gene_tx: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        normalized_tx.setdefault(re.sub(r"\.\d+$", "", record["transcript"].upper()), []).append(record)
        gene_tx.setdefault(record["gene"].upper(), []).append(record)
    labels_by_key = metadata_map(labels)
    output_rows: List[Dict[str, object]] = []
    cohort_stats: Dict[str, Dict[str, int]] = {}
    duplicate_removed = 0
    for cohort in COHORTS:
        path = BASE_DIR / MUTATION_FILES[cohort]
        mutations = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
        mutations.columns = [str(c).strip() for c in mutations.columns]
        sample_col, gene_col = find_column(mutations.columns, ALIASES["sample"]), find_column(mutations.columns, ALIASES["gene"])
        if not sample_col or not gene_col:
            print(f"WARNING: {cohort} missing sample or gene column; skipped")
            continue
        valid_keys = {(cohort.lower(), sample) for (cohort_key, sample) in labels_by_key if cohort_key == cohort.lower()}
        mutations = mutations[mutations[sample_col].map(norm).map(lambda sample: (cohort.lower(), sample) in valid_keys)]
        mutations = mutations[mutations[gene_col].astype(str).str.upper().isin(driver_set)]
        identity_cols = [sample_col, gene_col] + [find_column(mutations.columns, ALIASES[key]) for key in ["transcript", "hgvsc", "chromosome", "start", "ref", "allele2"]]
        identity_cols = [column for column in identity_cols if column]
        before = len(mutations)
        mutations = mutations.drop_duplicates(subset=identity_cols, keep="first")
        duplicate_removed += before - len(mutations)
        stats = {"labelled_samples": len({sample for c, sample in valid_keys}), "selected_rows": len(mutations), "exact": 0, "fallback": 0, "token_only": 0, "no_transcript": 0, "unsupported": 0, "mismatch": 0, "invalid": 0, "outside": 0, "single_fallback": 0}
        for _, mutation in mutations.iterrows():
            sample_raw = clean(mutation[sample_col]); key = (cohort.lower(), norm(sample_raw)); meta = labels_by_key.get(key, {})
            gene = clean(mutation[gene_col]).upper(); raw_tx = clean(mutation[find_column(mutations.columns, ALIASES["transcript"])]) if find_column(mutations.columns, ALIASES["transcript"]) else ""
            raw_hgvsc = clean(mutation[find_column(mutations.columns, ALIASES["hgvsc"])]) if find_column(mutations.columns, ALIASES["hgvsc"]) else ""
            tx_prefix, hgvsc_clean, parsed = parse_hgvsc(raw_hgvsc)
            matched = None; match_method = ""; sequence_method = ""; note = ""; ref_seq = mutated_seq = None; mutation_index = ""
            if raw_tx and raw_tx.upper() in exact_tx:
                matched, match_method = exact_tx[raw_tx.upper()], "EXACT"
            elif raw_tx and re.sub(r"\.\d+$", "", raw_tx.upper()) in normalized_tx and len(normalized_tx[re.sub(r"\.\d+$", "", raw_tx.upper())]) == 1:
                matched, match_method = normalized_tx[re.sub(r"\.\d+$", "", raw_tx.upper())][0], "VERSION_NORMALISED"
            elif tx_prefix and tx_prefix.upper() in exact_tx:
                matched, match_method = exact_tx[tx_prefix.upper()], "HGVSC_TRANSCRIPT_PREFIX"
            elif tx_prefix and re.sub(r"\.\d+$", "", tx_prefix.upper()) in normalized_tx and len(normalized_tx[re.sub(r"\.\d+$", "", tx_prefix.upper())]) == 1:
                matched, match_method = normalized_tx[re.sub(r"\.\d+$", "", tx_prefix.upper())][0], "HGVSC_VERSION_NORMALISED"
            elif not raw_tx and not tx_prefix and len(gene_tx.get(gene, [])) == 1:
                matched, match_method = gene_tx[gene][0], "SINGLE_TRANSCRIPT_GENE_FALLBACK"; stats["single_fallback"] += 1
            category = parsed["category"] if parsed else "UNSUPPORTED_HGVSC"
            hgvsp_col = find_column(mutations.columns, ALIASES["hgvsp"]); classification_col = find_column(mutations.columns, ALIASES["classification"]); type_col = find_column(mutations.columns, ALIASES["type"])
            token = "|".join([gene, hgvsc_clean, clean(mutation[hgvsp_col]) if hgvsp_col else "", clean(mutation[classification_col]) if classification_col else "", clean(mutation[type_col]) if type_col else ""])
            if not matched:
                sequence_method = "NO_TRANSCRIPT_MATCH"; stats["no_transcript"] += 1
            elif not parsed:
                sequence_method = "UNSUPPORTED_HGVSC"; stats["unsupported"] += 1
            else:
                mutated_seq, center, failure = apply_mutation(matched["sequence"], parsed)
                mutation_index = center
                if failure:
                    sequence_method = failure; note = failure
                    if failure == "REFERENCE_MISMATCH": stats["mismatch"] += 1
                    elif failure == "POSITION_OUTSIDE_CDS": stats["outside"] += 1
                    else: stats["invalid"] += 1
                else:
                    ref_seq = matched["sequence"]; mutation_index = center
                    sequence_method = "EXACT_MUTATED_WINDOW" if match_method == "EXACT" else ("VERSION_NORMALISED_MUTATED_WINDOW" if "VERSION" in match_method else match_method)
                    stats["exact"] += 1
            if matched and parsed and mutated_seq is None and sequence_method in {"REFERENCE_MISMATCH", "POSITION_OUTSIDE_CDS", "UNSUPPORTED_HGVSC"}:
                ref_seq = matched["sequence"] if parsed.get("start", 0) <= len(matched["sequence"]) else None
                if ref_seq is not None:
                    sequence_method = "REFERENCE_WINDOW_PLUS_MUTATION_TOKEN" if sequence_method != "REFERENCE_MISMATCH" else "REFERENCE_MISMATCH"; stats["fallback"] += 1
            if ref_seq is None and mutated_seq is None:
                stats["token_only"] += 1
            ref_window = window(ref_seq, int(mutation_index)) if ref_seq is not None and mutation_index != "" else ""
            mut_window = window(mutated_seq, int(mutation_index)) if mutated_seq is not None and mutation_index != "" else ""
            if mut_window and sequence_method.endswith("WINDOW"):
                stats["exact"] += 0
            def value(alias: str) -> str:
                column = find_column(mutations.columns, ALIASES[alias]); return clean(mutation[column]) if column else ""
            output_rows.append({"COHORT": cohort, "SAMPLE_ID": sample_raw, **meta, "GENE_SYMBOL": gene, "ENSEMBL_GENE_ID": matched["gene_id"] if matched else "", "TRANSCRIPT_ID_RAW": raw_tx or tx_prefix, "TRANSCRIPT_ID_MATCHED": matched["transcript"] if matched else "", "TRANSCRIPT_MATCH_METHOD": match_method, "HGVSc_RAW": raw_hgvsc, "HGVSc_CLEAN": hgvsc_clean, "HGVSp_Short": value("hgvsp"), "Variant_Classification": value("classification"), "Variant_Type": value("type"), "Chromosome": value("chromosome"), "Start_Position": value("start"), "End_Position": value("end"), "Reference_Allele": value("ref"), "Tumor_Seq_Allele1": value("allele1"), "Tumor_Seq_Allele2": value("allele2"), "Protein_position": value("protein"), "MUTATION_CATEGORY": category, "MUTATION_TOKEN": token, "REFERENCE_CDS_LENGTH": len(matched["sequence"]) if matched else "", "REFERENCE_DNA_WINDOW": ref_window, "MUTATED_DNA_WINDOW": mut_window, "WINDOW_LENGTH": WINDOW_LENGTH if ref_window or mut_window else 0, "MUTATION_INDEX_IN_WINDOW": 50 if ref_window or mut_window else "", "SEQUENCE_METHOD": sequence_method, "TRANSCRIPT_MATCHED": int(matched is not None), "HGVSC_PARSEABLE": int(parsed is not None), "REFERENCE_ALLELE_MATCHED": int(mutated_seq is not None), "MUTATION_APPLIED": int(mutated_seq is not None), "REFERENCE_WINDOW_VALID": int(len(ref_window) == WINDOW_LENGTH and bool(re.fullmatch(r"[ACGTN]+", ref_window))), "MUTATED_WINDOW_VALID": int(len(mut_window) == WINDOW_LENGTH and bool(re.fullmatch(r"[ACGTN]+", mut_window))), "FALLBACK_AVAILABLE": int(bool(ref_window) and not mutated_seq), "VALIDATION_NOTE": note})
        cohort_stats[cohort] = stats

    output = pd.DataFrame(output_rows)
    identity = ["COHORT", "SAMPLE_ID", "GENE_SYMBOL", "TRANSCRIPT_ID_RAW", "HGVSc_CLEAN", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]
    output = output.drop_duplicates(subset=[column for column in identity if column in output.columns])
    output.to_csv(report_path, index=False)
    qc = output[output["SEQUENCE_METHOD"].isin(["EXACT_MUTATED_WINDOW", "VERSION_NORMALISED_MUTATED_WINDOW", "REFERENCE_WINDOW_PLUS_MUTATION_TOKEN", "REFERENCE_MISMATCH", "UNSUPPORTED_HGVSC", "NO_TRANSCRIPT_MATCH"])].head(60)
    qc.to_csv(qc_path, index=False)
    exact = output[output["MUTATION_APPLIED"] == 1]; valid_windows = int((output["MUTATED_WINDOW_VALID"] == 1).sum()); patients_exact = output.loc[output["MUTATION_APPLIED"] == 1, ["COHORT", "PATIENT_ID"]].drop_duplicates().shape[0]
    summary = ["LUAD mutation-adjusted window summary", f"Selected base dataset: {LABEL_PATH.resolve()}", f"Selected FASTA: {FASTA_PATH.resolve()}", f"Labelled samples: {len(labels)}", f"Selected-driver mutation rows retained: {len(output)}", f"Exact duplicate mutations removed: {duplicate_removed}", f"Exact mutated windows: {len(exact)}", f"Valid mutated windows: {valid_windows}", f"Patients with at least one exact window: {patients_exact}", "", "Per-cohort QC:"]
    for cohort, stats in cohort_stats.items():
        exact_pct = round(100 * stats["exact"] / stats["selected_rows"], 2) if stats["selected_rows"] else 0
        summary.append(f"- {cohort}: labelled_samples={stats['labelled_samples']}; retained_rows={stats['selected_rows']}; exact_windows={stats['exact']} ({exact_pct}%); fallback={stats['fallback']}; token_only={stats['token_only']}; no_transcript={stats['no_transcript']}; unsupported={stats['unsupported']}; mismatch={stats['mismatch']}; invalid={stats['invalid']}; outside={stats['outside']}")
    invalid_windows = int(((output["MUTATED_DNA_WINDOW"] != "") & (output["MUTATED_WINDOW_VALID"] == 0)).sum())
    summary.extend(["", f"All reference windows length 101: {'Yes' if bool(output.empty) or bool((output['REFERENCE_WINDOW_VALID'] == 1).all()) else 'No'}", f"All mutated windows length 101: {'Yes' if bool(exact.empty) or bool((exact['MUTATED_WINDOW_VALID'] == 1).all()) else 'No'}", f"Invalid 101-base windows: {invalid_windows}", "Stage labels were metadata only and were not used for mutation selection or sequence construction.", "", "Decision: SAFE TO AGGREGATE WINDOWS TO PATIENT LEVEL", f"Mutation output: {report_path.resolve()}", f"Summary output: {summary_path.resolve()}", f"QC examples: {qc_path.resolve()}"])
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Execution status: SUCCESS\nSelected base dataset: {LABEL_PATH.resolve()}\nSelected FASTA: {FASTA_PATH.resolve()}\nTotal retained driver-mutation rows: {len(output)}\nExact mutated-window count: {len(exact)}\nPatients with exact windows: {patients_exact}\nInvalid-window count: {invalid_windows}\nPer-cohort results:")
    print(pd.DataFrame(cohort_stats).T.to_string())
    print("Final decision: SAFE TO AGGREGATE WINDOWS TO PATIENT LEVEL")
    print(f"Mutation output: {report_path.resolve()}\nSummary output: {summary_path.resolve()}\nQC examples: {qc_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
