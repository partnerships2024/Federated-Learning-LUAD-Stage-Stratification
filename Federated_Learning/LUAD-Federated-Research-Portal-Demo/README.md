## CHI-Lab LUAD-FedStage

### Federated Genomic Stage Stratification Research Portal

This Flask portal is the final CHI Lab research prototype for binary LUAD stage-group classification: Early (Stages I-II) versus Advanced (Stages III-IV). It uses mutation-centred 301-bp genomic windows with a Federated Binary BiLSTM and sample-weighted Federated Averaging (FedAvg) across three simulated research clients/hospitals.

The portal provides:

- Data-local genomic inference using the validated global model and persisted preprocessing configuration.
- A local report archive that stores report metadata in the client's generated local database.
- A central federated server dashboard with federated rounds, validation/test metrics, ROC-AUC, PR-AUC, confusion-matrix, and comparison results.
- CHI Lab branding with the ICRI-STE research affiliation.

The demonstration is privacy-aware and data-local; it does not claim differential privacy, secure aggregation, or a formal privacy guarantee.

> **Research prototype - not for clinical diagnosis or treatment decisions.**

### Local setup

Use Python 3.10 with the pinned dependencies in `requirements.txt`, preferably inside a project-local `.venv`. Install dependencies and run `python app.py` from this directory. Demo-only usernames are `hospital1`, `hospital2`, `hospital3`, and `admin`; these accounts and their credentials are for local demonstration only, not production deployment.

### Included runtime artifacts

The portal includes the validated global BiLSTM weights, persisted preprocessing configuration, dashboard metrics/figures, required sanitized simulated client/server databases, templates, static assets, and official CHI Lab and ICRI-STE logos. Generated `lab_*_reports.db` files are recreated locally and ignored by Git.

### Federated Genomic Stage Stratification Research Portal Demo Video

## CHI Lab Research Project Demo

https://github.com/partnerships2024/CHI-Lab-Project-II/blob/main/Federated_Learning/LUAD-Federated-Research-Portal-Demo/2026-09-15%2017-52-38.mp4 
