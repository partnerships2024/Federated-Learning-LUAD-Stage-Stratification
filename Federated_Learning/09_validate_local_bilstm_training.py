"""Validate one local Hospital 1 epoch for the approved Binary BiLSTM.

The model graph mirrors the recovered notebook's Binary BiLSTM architecture.
This script never reads the global test-key file and never saves model output.
"""

import json
import contextlib
import io
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


NOTEBOOK_REFERENCE = Path("05_301bp_luad_model_training_RECOVERED.ipynb")
DATABASE_DIRECTORY = Path("databases")
HOSPITAL_DATABASE = DATABASE_DIRECTORY / "hospital_1.db"
OTHER_HOSPITAL_DATABASES = [
    DATABASE_DIRECTORY / "hospital_2.db",
    DATABASE_DIRECTORY / "hospital_3.db",
]
RANDOM_SEED = 42
MAX_MUTATIONS = 37
WINDOW_LENGTH = 301
DNA_VOCAB = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
GENE_PAD_TOKEN = "PAD_GENE"
GENE_UNK_TOKEN = "UNK_GENE"
NO_MUTATION_GENE_TOKEN = "NO_MUTATION"
CLINICAL_COLUMNS = [
    "CLINICAL_SEX_BINARY",
    "CLINICAL_AGE_YEARS",
    "CLINICAL_ACTIVE_SMOKING_BINARY",
    "CLINICAL_PACK_YEARS",
]


def set_reproducibility_seed() -> None:
    """Set Python/NumPy/TensorFlow seeds without enabling model training yet."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def load_database_rows(path: Path) -> list[dict[str, object]]:
    """Load all rows from one hospital database without modifying it."""
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM patients").fetchall()]


def patient_key(row: dict[str, object]) -> tuple[str, str]:
    """Return the stable patient key."""
    return str(row["COHORT"]), str(row["PATIENT_ID"])


def parse_json_list(value: object, column: str, key: tuple[str, str]) -> list[object]:
    """Parse one stored list column; no source value is written or transformed on disk."""
    if value is None or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {column} for {key}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{column} is not a list for {key}")
    return parsed


def numeric_value(value: object) -> float | None:
    """Convert a clinical value to finite float or missing."""
    if value is None or str(value).strip() == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def mode_or_unknown(values: list[float]) -> float:
    """Use the first most-common training value, with notebook fallback -1.0."""
    return float(Counter(values).most_common(1)[0][0]) if values else -1.0


def median(values: list[float]) -> float:
    """Compute the training-only median."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else float((ordered[middle - 1] + ordered[middle]) / 2.0)


def fit_preprocessing(training_rows: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, float]]:
    """Fit clinical statistics and gene vocabulary from Hospitals 1-3 only."""
    numeric_training = {
        column: [converted for row in training_rows for converted in [numeric_value(row[column])] if converted is not None]
        for column in CLINICAL_COLUMNS
    }
    imputation = {
        "CLINICAL_SEX_BINARY": mode_or_unknown(numeric_training["CLINICAL_SEX_BINARY"]),
        "CLINICAL_AGE_YEARS": median(numeric_training["CLINICAL_AGE_YEARS"]),
        "CLINICAL_ACTIVE_SMOKING_BINARY": mode_or_unknown(numeric_training["CLINICAL_ACTIVE_SMOKING_BINARY"]),
        "CLINICAL_PACK_YEARS": median(numeric_training["CLINICAL_PACK_YEARS"]),
    }
    filled = {
        column: [numeric_value(row[column]) if numeric_value(row[column]) is not None else imputation[column] for row in training_rows]
        for column in CLINICAL_COLUMNS
    }
    means = {column: sum(values) / len(values) for column, values in filled.items()}
    scales = {}
    for column, values in filled.items():
        variance = sum((value - means[column]) ** 2 for value in values) / len(values)
        scales[column] = math.sqrt(variance) if variance > 0 else 1.0

    gene_set = {NO_MUTATION_GENE_TOKEN}
    for row in training_rows:
        genes = parse_json_list(row["GENE_SYMBOL_LIST"], "GENE_SYMBOL_LIST", patient_key(row))
        gene_set.update(str(gene) for gene in genes)
    vocabulary = {GENE_PAD_TOKEN: 0, GENE_UNK_TOKEN: 1}
    for gene in sorted(gene_set):
        if gene not in vocabulary:
            vocabulary[gene] = len(vocabulary)
    return imputation, vocabulary, means, scales


def standardize(value: object, column: str, imputation: dict[str, float], means: dict[str, float], scales: dict[str, float]) -> float:
    """Apply the training-fitted clinical imputation and StandardScaler transform."""
    converted = numeric_value(value)
    filled = imputation[column] if converted is None else converted
    return (filled - means[column]) / scales[column]


def build_arrays(rows: list[dict[str, object]], vocabulary: dict[str, int], imputation: dict[str, float], means: dict[str, float], scales: dict[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build the seven approved model inputs in memory."""
    n_patients = len(rows)
    gene_tokens = np.full((n_patients, MAX_MUTATIONS), vocabulary[GENE_PAD_TOKEN], dtype=np.int32)
    mutated_dna = np.full((n_patients, MAX_MUTATIONS, WINDOW_LENGTH), DNA_VOCAB["PAD"], dtype=np.int32)
    mutation_position_mask = np.zeros((n_patients, MAX_MUTATIONS, WINDOW_LENGTH), dtype=np.float32)
    clinical = {column: np.zeros((n_patients, 1), dtype=np.float32) for column in CLINICAL_COLUMNS}
    labels = np.zeros((n_patients,), dtype=np.float32)

    for row_index, row in enumerate(rows):
        key = patient_key(row)
        genes = parse_json_list(row["GENE_SYMBOL_LIST"], "GENE_SYMBOL_LIST", key)
        windows = parse_json_list(row["MUTATED_WINDOW_301_LIST"], "MUTATED_WINDOW_301_LIST", key)
        masks = parse_json_list(row["MUTATED_AFFECTED_MASK_301_LIST"], "MUTATED_AFFECTED_MASK_301_LIST", key)
        if not genes:
            genes = [NO_MUTATION_GENE_TOKEN]
            windows = ["N" * WINDOW_LENGTH]
            masks = ["0" * WINDOW_LENGTH]
        if len(genes) != len(windows) or len(genes) != len(masks):
            raise ValueError(f"Unaligned mutation lists for {key}")
        if len(genes) > MAX_MUTATIONS:
            raise ValueError(f"{key} exceeds MAX_MUTATIONS={MAX_MUTATIONS}; no truncation is allowed")
        for mutation_index, gene in enumerate(genes):
            gene_tokens[row_index, mutation_index] = vocabulary.get(str(gene), vocabulary[GENE_UNK_TOKEN])
            sequence = str(windows[mutation_index]).upper()
            if len(sequence) != WINDOW_LENGTH:
                raise ValueError(f"DNA length mismatch for {key}")
            mutated_dna[row_index, mutation_index, :] = [DNA_VOCAB.get(base, DNA_VOCAB["N"]) for base in sequence]
            raw_mask = list(masks[mutation_index]) if isinstance(masks[mutation_index], str) else masks[mutation_index]
            if not isinstance(raw_mask, list) or len(raw_mask) != WINDOW_LENGTH:
                raise ValueError(f"Mask length mismatch for {key}")
            for position, value in enumerate(raw_mask):
                if str(value) not in {"0", "1"}:
                    raise ValueError(f"Non-binary mask value for {key}")
                mutation_position_mask[row_index, mutation_index, position] = float(value)
        for column in CLINICAL_COLUMNS:
            clinical[column][row_index, 0] = standardize(row[column], column, imputation, means, scales)
        labels[row_index] = float(row["TARGET_STAGE_BINARY"])

    return {
        "gene_tokens": gene_tokens,
        "mutated_dna": mutated_dna,
        "mutation_position_mask": mutation_position_mask,
        "clinical_sex": clinical["CLINICAL_SEX_BINARY"],
        "clinical_age": clinical["CLINICAL_AGE_YEARS"],
        "clinical_active_smoking": clinical["CLINICAL_ACTIVE_SMOKING_BINARY"],
        "clinical_pack_years": clinical["CLINICAL_PACK_YEARS"],
    }, labels


def stratified_train_validation(rows: list[dict[str, object]], validation_fraction: float = 0.20) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create a deterministic local stratified split without external split dependencies."""
    generator = random.Random(RANDOM_SEED)
    by_label = {}
    for row in rows:
        by_label.setdefault(str(row["TARGET_STAGE_BINARY"]), []).append(row)
    train_rows, validation_rows = [], []
    total_validation = math.ceil(len(rows) * validation_fraction)
    allocations = {label: math.floor(len(group) * total_validation / len(rows)) for label, group in by_label.items()}
    remainder = total_validation - sum(allocations.values())
    order = sorted(by_label, key=lambda label: len(by_label[label]) * total_validation / len(rows) - allocations[label], reverse=True)
    for label in order[:remainder]:
        allocations[label] += 1
    for label in sorted(by_label):
        group = by_label[label].copy()
        generator.shuffle(group)
        validation_rows.extend(group[:allocations[label]])
        train_rows.extend(group[allocations[label]:])
    generator.shuffle(train_rows)
    generator.shuffle(validation_rows)
    return train_rows, validation_rows


def build_binary_bilstm(tf, vocabulary_size: int):
    """Reproduce the complete Binary BiLSTM graph from the recovered notebook."""
    from tensorflow.keras import layers, models, regularizers

    l2_weight = 1e-4
    dna_embedding_dim = 8
    gene_embedding_dim = 8
    lstm_units = 32

    def position_masked_mean_pooling(inputs):
        sequence_features, position_mask = inputs
        position_mask = tf.cast(tf.expand_dims(position_mask, axis=-1), sequence_features.dtype)
        feature_sum = tf.reduce_sum(sequence_features * position_mask, axis=2)
        valid_position_count = tf.maximum(tf.reduce_sum(position_mask, axis=2), 1.0)
        pooled_features = feature_sum / valid_position_count
        has_affected_position = tf.cast(tf.reduce_sum(position_mask, axis=2) > 0, sequence_features.dtype)
        return pooled_features * has_affected_position

    def masked_mean_pooling(inputs):
        mutation_features, mutation_padding_mask = inputs
        mutation_padding_mask = tf.cast(tf.expand_dims(mutation_padding_mask, axis=-1), mutation_features.dtype)
        feature_sum = tf.reduce_sum(mutation_features * mutation_padding_mask, axis=1)
        valid_mutation_count = tf.maximum(tf.reduce_sum(mutation_padding_mask, axis=1), 1.0)
        return feature_sum / valid_mutation_count

    def masked_max_pooling(inputs):
        mutation_features, mutation_padding_mask = inputs
        mutation_padding_mask = tf.cast(tf.expand_dims(mutation_padding_mask, axis=-1), tf.bool)
        very_negative_value = tf.fill(tf.shape(mutation_features), tf.cast(-1e9, mutation_features.dtype))
        return tf.reduce_max(tf.where(mutation_padding_mask, mutation_features, very_negative_value), axis=1)

    gene_tokens_input = layers.Input(shape=(MAX_MUTATIONS,), dtype="int32", name="gene_tokens")
    mutated_dna_input = layers.Input(shape=(MAX_MUTATIONS, WINDOW_LENGTH), dtype="int32", name="mutated_dna")
    mutation_position_mask_input = layers.Input(shape=(MAX_MUTATIONS, WINDOW_LENGTH), dtype="float32", name="mutation_position_mask")
    clinical_sex_input = layers.Input(shape=(1,), dtype="float32", name="clinical_sex")
    clinical_age_input = layers.Input(shape=(1,), dtype="float32", name="clinical_age")
    clinical_active_smoking_input = layers.Input(shape=(1,), dtype="float32", name="clinical_active_smoking")
    clinical_pack_years_input = layers.Input(shape=(1,), dtype="float32", name="clinical_pack_years")

    mutation_padding_mask = layers.Lambda(lambda x: tf.cast(tf.not_equal(x, 0), tf.float32), name="mutation_padding_mask")(gene_tokens_input)
    dna_embedding = layers.Embedding(input_dim=len(DNA_VOCAB), output_dim=dna_embedding_dim, mask_zero=False, name="mutated_dna_embedding")(mutated_dna_input)
    position_mask_channel = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="position_mask_channel")(mutation_position_mask_input)
    dna_features_with_position_mask = layers.Concatenate(axis=-1, name="dna_features_with_position_mask")([dna_embedding, position_mask_channel])
    sequence_lstm_features = layers.TimeDistributed(
        layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True, dropout=0.10, recurrent_dropout=0.0)),
        name="per_mutation_bilstm",
    )(dna_features_with_position_mask)
    global_sequence_features = layers.TimeDistributed(layers.GlobalMaxPooling1D(), name="global_sequence_pooling")(sequence_lstm_features)
    affected_sequence_features = layers.Lambda(position_masked_mean_pooling, name="affected_position_pooling")([sequence_lstm_features, mutation_position_mask_input])
    sequence_features = layers.Concatenate(name="sequence_features")([global_sequence_features, affected_sequence_features])
    gene_features = layers.Embedding(input_dim=vocabulary_size, output_dim=gene_embedding_dim, mask_zero=False, name="gene_embedding")(gene_tokens_input)
    mutation_features = layers.Concatenate(name="mutation_sequence_gene_features")([sequence_features, gene_features])
    mutation_features = layers.Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(l2_weight), name="mutation_dense")(mutation_features)
    mutation_features = layers.Dropout(rate=0.20, name="mutation_dropout")(mutation_features)
    masked_mean_features = layers.Lambda(masked_mean_pooling, name="masked_mean_pooling")([mutation_features, mutation_padding_mask])
    masked_max_features = layers.Lambda(masked_max_pooling, name="masked_max_pooling")([mutation_features, mutation_padding_mask])
    patient_mutation_features = layers.Concatenate(name="patient_mutation_features")([masked_mean_features, masked_max_features])
    clinical_features = layers.Concatenate(name="clinical_features")([
        clinical_sex_input, clinical_age_input, clinical_active_smoking_input, clinical_pack_years_input,
    ])
    clinical_features = layers.Dense(units=16, activation="relu", name="clinical_dense")(clinical_features)
    combined_patient_features = layers.Concatenate(name="combined_patient_features")([patient_mutation_features, clinical_features])
    classifier = layers.Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(l2_weight), name="classifier_dense_1")(combined_patient_features)
    classifier = layers.Dropout(rate=0.30, name="classifier_dropout_1")(classifier)
    classifier = layers.Dense(units=32, activation="relu", kernel_regularizer=regularizers.l2(l2_weight), name="classifier_dense_2")(classifier)
    classifier = layers.Dropout(rate=0.20, name="classifier_dropout_2")(classifier)
    binary_output = layers.Dense(units=1, activation="sigmoid", name="binary_stage_output")(classifier)
    model = models.Model(
        inputs=[gene_tokens_input, mutated_dna_input, mutation_position_mask_input, clinical_sex_input, clinical_age_input, clinical_active_smoking_input, clinical_pack_years_input],
        outputs=binary_output,
        name="LUAD_301bp_Binary_LSTM",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="advanced_recall"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def main() -> None:
    """Prepare Hospital 1, build the model, forward-test, and train one epoch."""
    print("=" * 78)
    print("LOCAL HOSPITAL 1 BINARY BILSTM TRAINING VALIDATION")
    print("=" * 78)
    set_reproducibility_seed()

    # Section 1: Load all federated training hospitals; never load global test data.
    hospital_1_rows = load_database_rows(HOSPITAL_DATABASE)
    training_rows = hospital_1_rows[:]
    for database_path in OTHER_HOSPITAL_DATABASES:
        training_rows.extend(load_database_rows(database_path))
    if len(hospital_1_rows) != 949:
        raise ValueError(f"Expected 949 Hospital 1 patients, found {len(hospital_1_rows)}")

    # Section 2: Fit shared preprocessing on federated training patients only.
    imputation, vocabulary, means, scales = fit_preprocessing(training_rows)
    h1_inputs, h1_labels = build_arrays(hospital_1_rows, vocabulary, imputation, means, scales)
    train_rows, validation_rows = stratified_train_validation(hospital_1_rows)
    train_indices = [hospital_1_rows.index(row) for row in train_rows]
    validation_indices = [hospital_1_rows.index(row) for row in validation_rows]
    train_inputs = {name: values[train_indices] for name, values in h1_inputs.items()}
    validation_inputs = {name: values[validation_indices] for name, values in h1_inputs.items()}
    train_labels = h1_labels[train_indices]
    validation_labels = h1_labels[validation_indices]
    train_keys = {patient_key(row) for row in train_rows}
    validation_keys = {patient_key(row) for row in validation_rows}
    overlap_count = len(train_keys & validation_keys)

    print("\n--- LOCAL SPLIT ---")
    print(f"Local train patient count: {len(train_rows)}")
    print(f"Local validation patient count: {len(validation_rows)}")
    print(f"Local train class distribution: {dict(sorted(Counter(train_labels.astype(int)).items()))}")
    print(f"Local validation class distribution: {dict(sorted(Counter(validation_labels.astype(int)).items()))}")
    print(f"Train/validation patient overlap: {overlap_count}")
    print("\n--- INPUT SHAPES ---")
    for name, values in train_inputs.items():
        print(f"{name}: {values.shape}")

    # Section 3: Import TensorFlow only for the requested model/functional test.
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import tensorflow as tf
    except Exception as error:
        print("\n--- TENSORFLOW ENVIRONMENT CHECK ---")
        print(f"TensorFlow import status: BLOCKED ({type(error).__name__}: {error})")
        print("The installed TensorFlow binary is incompatible with the installed NumPy runtime.")
        print("LOCAL BILSTM TRAINING VALIDATION: FAIL")
        return

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = build_binary_bilstm(tf, len(vocabulary))
    print("\n--- MODEL ---")
    print(f"Model name: {model.name}")
    print(f"Total parameters: {model.count_params()}")
    print(f"Trainable parameters: {sum(int(np.prod(variable.shape)) for variable in model.trainable_variables)}")
    print(f"Output shape: {model.output_shape}")

    # Section 4: Verify a small forward pass before the one-epoch test.
    model_inputs = {name: values[:2] for name, values in train_inputs.items()}
    predictions = model(model_inputs, training=False).numpy()
    forward_pass_ok = predictions.shape == (2, 1) and np.isfinite(predictions).all()
    print("\n--- FORWARD PASS ---")
    print(f"Forward-pass status: {'PASS' if forward_pass_ok else 'FAIL'}")
    print(f"Prediction shape: {predictions.shape}")

    # Section 5: One functional local epoch with train-only class weights.
    label_counts = Counter(train_labels.astype(int))
    total_train = len(train_labels)
    class_weights = {label: total_train / (2.0 * count) for label, count in sorted(label_counts.items())}
    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(validation_inputs, validation_labels),
        class_weight=class_weights,
        epochs=1,
        batch_size=16,
        verbose=1,
    )
    metrics = history.history
    training_loss = float(metrics["loss"][-1])
    training_accuracy = float(metrics["accuracy"][-1])
    validation_loss = float(metrics["val_loss"][-1])
    validation_accuracy = float(metrics["val_accuracy"][-1])
    validation_roc_auc = float(metrics["val_roc_auc"][-1])
    finite_metrics = all(math.isfinite(value) for value in [training_loss, training_accuracy, validation_loss, validation_accuracy, validation_roc_auc])
    print("\n--- ONE-EPOCH FUNCTIONAL TEST ---")
    print(f"Training loss: {training_loss}")
    print(f"Training accuracy: {training_accuracy}")
    print(f"Validation loss: {validation_loss}")
    print(f"Validation accuracy: {validation_accuracy}")
    print(f"Validation ROC-AUC: {validation_roc_auc}")
    status_pass = model.name == "LUAD_301bp_Binary_LSTM" and forward_pass_ok and finite_metrics and overlap_count == 0
    print("\nLOCAL BILSTM TRAINING VALIDATION: PASS" if status_pass else "\nLOCAL BILSTM TRAINING VALIDATION: FAIL")


if __name__ == "__main__":
    main()
