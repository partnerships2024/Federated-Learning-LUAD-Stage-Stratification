"""Check whether mutation-adjusted LUAD sequence reconstruction is feasible."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
READ_ONLY_PROCESSED = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "luad_processed"
REPORT_PATH = OUTPUT_DIR / "luad_mutated_sequence_feasibility.csv"
SUMMARY_PATH = OUTPUT_DIR / "luad_mutated_sequence_feasibility_summary.txt"
LABEL_PATH = READ_ONLY_PROCESSED / "final_luad_multigene_sequence_training_dataset.csv"
DRIVER_PATH = READ_ONLY_PROCESSED / "LUAD_driver_genes.csv"
FASTA_CANDIDATES = [BASE_DIR / "LUAD_Ensembl_CDS_sequences_96genes.fasta.txt", BASE_DIR / "LUAD_Ensembl_CDS_sequences.fasta.txt"]

COHORTS = ["TCGA_Atlas_2018", "LUNG_MSK_2017", "MSK_NPJPO_2021", "MSKCC_2020", "MSKCC_2023", "NCI_2022", "ONCOSG_2020"]
MUTATION_FILES = {c: f.lower() + "_data_mutations.txt" for c in []}
MUTATION_FILES = {
    "TCGA_Atlas_2018": "tcga_atlas_2018_data_mutations.txt", "LUNG_MSK_2017": "lung_msk_2017_data_mutations.txt",
    "MSK_NPJPO_2021": "msk_npjpo_2021_data_mutations.txt", "MSKCC_2020": "mskcc_2020_data_mutations.txt",
    "MSKCC_2023": "mskcc_2023_data_mutations.txt", "NCI_2022": "nci_2022_data_mutations.txt",
    "ONCOSG_2020": "oncosg_2020_data_mutations.txt",
}
ALIASES = {
    "gene": ["Hugo_Symbol", "Gene", "GENE_SYMBOL", "Gene_Symbol"],
    "sample": ["Tumor_Sample_Barcode", "SAMPLE_ID", "Sample_ID", "Tumor_Sample_ID"],
    "hgvsc": ["HGVSc", "HGVSc_Short", "Coding_Change"],
    "transcript": ["Transcript_ID", "Transcript", "Transcript_Id"],
    "ref": ["Reference_Allele", "Reference_Allele_Seq"],
    "allele1": ["Tumor_Seq_Allele1", "Tumor_Seq_Allele_1"],
    "allele2": ["Tumor_Seq_Allele2", "Tumor_Seq_Allele_2"],
    "classification": ["Variant_Classification"], "type": ["Variant_Type"],
    "chromosome": ["Chromosome", "Chromosome_Name"], "start": ["Start_Position"],
    "end": ["End_Position"], "protein_position": ["Protein_position", "Protein_Position"],
    "hgvsp": ["HGVSp_Short", "HGVSp"], "codons": ["Codons"],
}


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def find_column(columns: Iterable[object], aliases: List[str]) -> Optional[str]:
    lookup = {norm(column): str(column) for column in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return None


def parse_fasta(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    header = None
    sequence: List[str] = []
    def finish() -> None:
        if header is None:
            return
        text = "".join(sequence).upper()
        gene_match = re.search(r"(?:gene_symbol|gene|symbol)[:=]([A-Za-z0-9_.-]+)", header, re.I)
        gene = gene_match.group(1) if gene_match else ""
        ensembl_genes = re.findall(r"(?:ENSG\d+(?:\.\d+)?)", header, re.I)
        ensembl_tx = re.findall(r"(?:ENST\d+(?:\.\d+)?)", header, re.I)
        pipe_fields = header.split("|")
        if not gene and len(pipe_fields) >= 3:
            gene = pipe_fields[2].strip()
        records.append({"header": header, "gene": gene, "gene_id": ensembl_genes[0] if ensembl_genes else "", "transcript": ensembl_tx[0] if ensembl_tx else "", "sequence": text})
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                finish(); header = line[1:]; sequence = []
            elif line:
                sequence.append(line)
    finish()
    return records


def driver_symbols() -> set[str]:
    if not DRIVER_PATH.is_file():
        return set()
    try:
        frame = pd.read_csv(DRIVER_PATH, dtype=str)
        column = find_column(frame.columns, ["SYMBOL", "Hugo_Symbol", "Gene"])
        return {norm(value) for value in clean(frame[column]) if norm(value)} if column else set()
    except Exception as exc:
        print(f"WARNING: Could not read driver-gene file: {exc}")
        return set()


def transcript_key(value: object) -> str:
    return re.sub(r"\.\d+$", "", str(value).strip()).upper()


def hgvsc_category(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "."}:
        return "missing"
    prefix = r"(?:[A-Za-z0-9_.-]+:)?"
    if re.fullmatch(prefix + r"c\.\d+[ACGT]>[ACGT]", text, re.I):
        return "simple SNV"
    if re.fullmatch(prefix + r"c\.\d+(?:_\d+)?del", text, re.I):
        return "simple deletion"
    if re.fullmatch(prefix + r"c\.\d+_\d+ins[ACGT]+", text, re.I):
        return "simple insertion"
    if re.fullmatch(prefix + r"c\.\d+(?:_\d+)?dup", text, re.I):
        return "simple duplication"
    if re.fullmatch(prefix + r"c\.\d+_\d+delins[ACGT]+", text, re.I):
        return "simple delins"
    return "complex or unsupported"


def snv_reference(value: str) -> Optional[Tuple[int, str]]:
    match = re.fullmatch(r"(?:[A-Za-z0-9_.-]+:)?c\.(\d+)([ACGT])>[ACGT]", value.strip(), re.I)
    return (int(match.group(1)), match.group(2).upper()) if match else None


def fasta_summary(records: List[Dict[str, str]], drivers: set[str]) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
    by_tx: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        by_tx.setdefault(transcript_key(record["transcript"]), []).append(record)
    genes = {norm(r["gene"]) for r in records if r["gene"]}
    represented = len(drivers & genes)
    valid = sum(bool(re.fullmatch(r"[ACGTN]+", r["sequence"])) for r in records)
    multi = sorted(g for g in genes if sum(norm(r["gene"]) == g for r in records) > 1)
    versions = sum(bool(re.search(r"ENST\d+\.\d+", r["header"], re.I)) for r in records)
    lines = [f"Total FASTA records: {len(records)}", f"Unique genes: {len(genes)}", f"Unique transcripts: {len(by_tx)}", f"Selected driver genes represented: {represented}/{len(drivers)}", f"Valid DNA-only sequences: {valid}/{len(records)}", f"Genes with multiple transcript sequences: {len(multi)}", f"Transcript version suffixes present: {'Yes' if versions else 'No'}", "Example headers:"]
    lines.extend(f"- {r['header']}" for r in records[:5])
    return "\n".join(lines), by_tx


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABEL_PATH, dtype=str) if LABEL_PATH.is_file() else None
    drivers = driver_symbols()
    fasta = next((path for path in FASTA_CANDIDATES if path.is_file()), None)
    records = parse_fasta(fasta) if fasta else []
    fasta_text, fasta_by_tx = fasta_summary(records, drivers)
    print(f"Selected label dataset: {LABEL_PATH.resolve() if labels is not None else 'MISSING'}")
    print(f"Selected FASTA file: {fasta.resolve() if fasta else 'MISSING'}")
    print(fasta_text)

    rows: List[Dict[str, object]] = []
    summary_lines = ["LUAD mutated-sequence feasibility summary", "", f"Selected label dataset: {LABEL_PATH.resolve() if labels is not None else 'MISSING'}", f"Selected FASTA file: {fasta.resolve() if fasta else 'MISSING'}", fasta_text, ""]
    for cohort in COHORTS:
        cohort_labels = labels[clean(labels["COHORT"]).str.lower() == cohort.lower()] if labels is not None and "COHORT" in labels else pd.DataFrame()
        label_samples = {norm(v) for v in clean(cohort_labels["SAMPLE_ID"]) if norm(v)} if "SAMPLE_ID" in cohort_labels else set()
        mutations = read_tsv(BASE_DIR / MUTATION_FILES[cohort])
        sample_col = find_column(mutations.columns, ALIASES["sample"]) if mutations is not None else None
        mutation_samples = {norm(v) for v in clean(mutations[sample_col]) if norm(v)} if mutations is not None and sample_col else set()
        matched_samples = label_samples & mutation_samples
        unmatched_samples = sorted(label_samples - mutation_samples)
        label_pct = round(100 * len(matched_samples) / len(label_samples), 2) if label_samples else 0.0
        selected = mutations[clean(mutations[sample_col]).map(norm).isin(matched_samples)] if mutations is not None and sample_col else pd.DataFrame()
        gene_col = find_column(selected.columns, ALIASES["gene"]) if not selected.empty else None
        selected = selected[selected[gene_col].map(norm).isin(drivers)] if gene_col else selected
        hgvsc_col = find_column(selected.columns, ALIASES["hgvsc"]) if not selected.empty else None
        transcript_col = find_column(selected.columns, ALIASES["transcript"]) if not selected.empty else None
        hgvsc_values = clean(selected[hgvsc_col]) if hgvsc_col else pd.Series(dtype=str)
        transcript_values = clean(selected[transcript_col]) if transcript_col else pd.Series(dtype=str)
        hgvsc_available = int((hgvsc_values.str.lower() != "").sum())
        transcript_available = int((transcript_values.str.lower() != "").sum())
        categories = {name: 0 for name in ["simple SNV", "simple deletion", "simple insertion", "simple duplication", "simple delins", "complex or unsupported"]}
        examples: Dict[str, List[str]] = {name: [] for name in categories}
        for value in hgvsc_values:
            category = hgvsc_category(value)
            if category in categories:
                categories[category] += 1
                if len(examples[category]) < 3:
                    examples[category].append(value)
        parseable = sum(categories.values()) - categories["complex or unsupported"]
        matched_tx = version_matched = gene_cds = 0
        ref_match = ref_mismatch = outside = unable = 0
        if not selected.empty:
            for _, mutation in selected.iterrows():
                gene = norm(mutation[gene_col]) if gene_col else ""
                tx = transcript_key(mutation[transcript_col]) if transcript_col else ""
                candidates = fasta_by_tx.get(tx, []) if tx else []
                if candidates:
                    matched_tx += 1
                    if str(mutation[transcript_col]).strip().upper() != candidates[0]["transcript"].upper():
                        version_matched += 1
                if any(norm(record["gene"]) == gene for record in records):
                    gene_cds += 1
                hgvsc = str(mutation[hgvsc_col]) if hgvsc_col else ""
                parsed = snv_reference(hgvsc)
                if not candidates or not parsed:
                    unable += 1
                else:
                    position, reference = parsed
                    sequence = candidates[0]["sequence"]
                    if position < 1 or position > len(sequence):
                        outside += 1
                    elif sequence[position - 1] == reference:
                        ref_match += 1
                    else:
                        ref_mismatch += 1
        match_pct = round(100 * matched_tx / transcript_available, 2) if transcript_available else 0.0
        parse_pct = round(100 * parseable / hgvsc_available, 2) if hgvsc_available else 0.0
        ref_total = ref_match + ref_mismatch
        ref_pct = round(100 * ref_match / ref_total, 2) if ref_total else 0.0
        method = "Option A: 101-base mutation-centred DNA windows" if match_pct >= 80 and parse_pct >= 80 and ref_pct >= 80 else "Option D: reference CDS plus separate mutation token"
        notes = f"Unmatched labelled samples: {unmatched_samples[:10]}; HGVSc examples: {examples}"
        rows.append({"COHORT": cohort, "LABELLED_SAMPLE_COUNT": len(label_samples), "LABELLED_SAMPLES_MATCHED": len(matched_samples), "LABEL_SIDE_MATCH_PERCENT": label_pct, "SELECTED_DRIVER_MUTATION_ROWS": len(selected), "HGVSC_AVAILABLE_COUNT": hgvsc_available, "HGVSC_AVAILABLE_PERCENT": round(100 * hgvsc_available / len(selected), 2) if len(selected) else 0.0, "TRANSCRIPT_ID_AVAILABLE_COUNT": transcript_available, "TRANSCRIPT_MATCH_COUNT": matched_tx, "TRANSCRIPT_MATCH_PERCENT": match_pct, "GENE_CDS_AVAILABLE_COUNT": gene_cds, "HGVSC_PARSEABLE_COUNT": parseable, "HGVSC_PARSEABLE_PERCENT": parse_pct, "REFERENCE_ALLELE_MATCH_COUNT": ref_match, "REFERENCE_ALLELE_MISMATCH_COUNT": ref_mismatch, "REFERENCE_ALLELE_MATCH_PERCENT": ref_pct, "RECOMMENDED_SEQUENCE_METHOD": method, "NOTES": notes})
        summary_lines.append(f"{cohort}: label-side mutation match={label_pct}%; selected-driver rows={len(selected)}; transcript match={match_pct}%; HGVSc parseable={parse_pct}%; reference match={ref_pct}%; method={method}; outside_CDS={outside}; unable_to_check={unable}")

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    mean_tx = report["TRANSCRIPT_MATCH_PERCENT"].mean() if not report.empty else 0
    mean_parse = report["HGVSC_PARSEABLE_PERCENT"].mean() if not report.empty else 0
    mean_ref = report["REFERENCE_ALLELE_MATCH_PERCENT"].mean() if not report.empty else 0
    if mean_tx >= 80 and mean_parse >= 80 and mean_ref >= 80:
        decision = "SAFE TO CREATE MUTATION-ADJUSTED DNA WINDOWS"
    elif mean_tx > 0 or mean_parse > 0:
        decision = "PARTIALLY SAFE - USE SUPPORTED MUTATIONS AND FLAG FALLBACK ROWS"
    else:
        decision = "NOT SAFE - REFERENCE/TRANSCRIPT MAPPING MUST BE FIXED FIRST"
    summary_lines.extend(["", "Unsupported mutation examples are included in the NOTES column of the CSV.", f"Mean transcript-match percentage: {mean_tx:.2f}%", f"Mean HGVSc parseability: {mean_parse:.2f}%", f"Mean reference-allele agreement: {mean_ref:.2f}%", "CNA and gene-panel availability are not required for this prototype.", f"Final decision: {decision}", f"CSV report: {REPORT_PATH.resolve()}", f"Summary report: {SUMMARY_PATH.resolve()}"])
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\nFINAL FEASIBILITY TABLE")
    print(report[["COHORT", "LABEL_SIDE_MATCH_PERCENT", "TRANSCRIPT_MATCH_PERCENT", "HGVSC_PARSEABLE_PERCENT", "REFERENCE_ALLELE_MATCH_PERCENT", "RECOMMENDED_SEQUENCE_METHOD"]].to_string(index=False))
    print(f"Recommended sequence format: {report['RECOMMENDED_SEQUENCE_METHOD'].mode().iloc[0] if not report.empty else 'None'}")
    print(f"Final decision: {decision}")
    print(f"CSV report: {REPORT_PATH.resolve()}")
    print(f"Summary report: {SUMMARY_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
