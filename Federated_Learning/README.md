# Federated Learning Experiment

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
