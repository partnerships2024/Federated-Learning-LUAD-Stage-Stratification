# CHI Lab Project II - LUAD Genomic Deep Learning

This repository contains the code, datasets, processed data, and experimental outputs for a Lung Adenocarcinoma (LUAD) stage-prediction research project using mutation-centred genomic DNA sequences.

## Experiments

### Experiment 1 - 101-bp Mutation-Centred Windows
The baseline experiment uses 101-bp sequence windows with 50 bases upstream, the centred mutation position, and 50 bases downstream. The mutation index is 50.

Code: Experiment_1_101bp/

### Experiment 2 - 301-bp Mutation-Centred Windows
The main experiment uses 301-bp sequence windows with 150 bases upstream, the centred mutation position, and 150 bases downstream. The mutation index is 150.

Code and results: Experiment_2_301bp/

## Data Sources

The study uses LUAD genomic and clinical data from:
- cBioPortal LUAD cohorts
- IntOGen cancer driver-gene resources
- Ensembl CDS reference sequences

Seven LUAD cohorts are included:
1. TCGA Atlas 2018
2. LUNG MSK 2017
3. MSK NPJ Precision Oncology 2021
4. MSKCC 2020
5. MSKCC 2023
6. NCI 2022
7. ONCOSG 2020

## Repository Structure

- data/raw/cbioportal/
- data/raw/intogen/
- data/raw/ensembl/
- data/processed/101bp/
- data/processed/301bp/
- Experiment_1_101bp/
- Experiment_2_301bp/

## Git LFS

Large genomic and processed dataset files are stored using Git LFS.
After cloning the repository, run:
git lfs install
git lfs pull

## Research Scope

The project evaluates two LUAD stage-prediction tasks:
- Binary: Early Stage (I-II) vs Advanced Stage (III-IV)
- Four-class: Stage I, Stage II, Stage III, and Stage IV

This repository represents an experimental research pipeline and is not a clinically validated diagnostic system.
