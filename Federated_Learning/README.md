# Federated Learning Experiment

## Final CHI-Lab LUAD-FedStage Portal

The final tested application is in [`LUAD-Federated-Research-Portal-Demo/`](LUAD-Federated-Research-Portal-Demo/). It is the **Federated Genomic Stage Stratification Research Portal**, a privacy-aware, data-local research prototype for Early (Stages I-II) versus Advanced (Stages III-IV) LUAD stage-group classification.

The portal uses mutation-centred 301-bp genomic windows and a Federated Binary BiLSTM trained through sample-weighted FedAvg across three simulated research clients/hospitals. It supports local genomic inference, a local report archive, and the central federated server dashboard, with CHI Lab branding and ICRI-STE affiliation.

> **Research prototype - not for clinical diagnosis or treatment decisions.**

Privacy-preserving federated learning experiment for binary LUAD stage prediction using 301-bp mutation-centred genomic DNA windows.

This experiment simulates three independent hospitals and trains a global BiLSTM model using sample-weighted Federated Averaging (FedAvg), without transferring raw patient records to the central server.

## Task

Binary LUAD stage classification:

- Early: Stage I–II
- Advanced: Stage III–IV

## Federated Setup

- Simulated hospitals: 3
- Federated rounds: 3
- Local epochs per round: 1
- Batch size: 16
- Aggregation: Sample-weighted FedAvg
- Global held-out test patients: 712

## Model

301-bp mutation-centred Binary BiLSTM integrating:

- DNA mutation windows
- Mutation-position masks
- Gene information
- Clinical features

## Repository Contents

The folder contains the federated-learning pipeline, trained global model, evaluation metrics, figures, and experiment summaries.

Raw patient-level genomic data and patient identifier files are intentionally excluded from the repository.
