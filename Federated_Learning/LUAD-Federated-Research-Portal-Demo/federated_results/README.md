# Portal Federated Result Artifacts

The portal keeps its runtime federated-learning artifacts grouped by purpose:

- `model/` — final trained federated model weights used for local inference.
- `preprocessing/` — persisted inference configuration, gene vocabulary, and scaling statistics.
- `metrics/` — federated and centralized evaluation metrics, round metrics, comparisons, and confusion-matrix data.
- `figures/` — evaluation visualizations used by the central server dashboard, including ROC and confusion-matrix figures.

These are the validated runtime artifacts for the CHI-Lab LUAD-FedStage research portal. They are not training checkpoints.
