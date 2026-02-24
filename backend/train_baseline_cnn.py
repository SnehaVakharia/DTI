from __future__ import annotations

import argparse
import json
import math
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


SMILES_COLUMN = "smiles"
PROTEIN_COLUMN = "protein_sequence"
LABEL_COLUMN = "label"
AFFINITY_COLUMN = "affinity_nm"

PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
SMILES_ALPHABET = list("CNOPSFIBrcnol[]=#()+-123456789/@\\")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reproducible baseline CNN model for DTI binary classification."
    )
    parser.add_argument("--data-path", type=Path, default=Path("backend/data/synthetic_dti.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Deliverables/baseline_cnn_run"),
        help="Root directory where per-run artifacts are saved under <output-dir>/<run_id>/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier. Defaults to UTC timestamp (YYYYmmddTHHMMSSZ).",
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("Deliverables/models/baseline_cnn_weights.pt")
    )
    parser.add_argument(
        "--results-csv", type=Path, default=Path("Deliverables/baseline_results.csv")
    )
    parser.add_argument("--results-md", type=Path, default=Path("Deliverables/baseline_results.md"))
    parser.add_argument(
        "--generate-synthetic-data",
        action="store_true",
        help="Generate a synthetic DTI-like dataset to the data path before training.",
    )
    parser.add_argument(
        "--prepare-davis-data",
        action="store_true",
        help=(
            "Prepare a flattened DAVIS CSV at --data-path before training. "
            "Uses data/DAVIS/{SMILES.txt,target_seq.txt,affinity.txt} by default."
        ),
    )
    parser.add_argument("--davis-smiles-path", type=Path, default=Path("data/DAVIS/SMILES.txt"))
    parser.add_argument("--davis-protein-path", type=Path, default=Path("data/DAVIS/target_seq.txt"))
    parser.add_argument("--davis-affinity-path", type=Path, default=Path("data/DAVIS/affinity.txt"))
    parser.add_argument(
        "--davis-threshold-nm",
        type=float,
        default=300.0,
        help="Pairs with affinity <= threshold are labeled positive.",
    )
    parser.add_argument(
        "--davis-censor-value-nm",
        type=float,
        default=10000.0,
        help="Affinity value used for censored/missing measurements in DAVIS.",
    )
    parser.add_argument(
        "--davis-keep-censored",
        action="store_true",
        help="Keep pairs where affinity >= --davis-censor-value-nm (usually hard negatives).",
    )
    parser.add_argument("--synthetic-samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-smiles-len", type=int, default=120)
    parser.add_argument("--max-protein-len", type=int, default=800)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binarize-threshold",
        type=float,
        default=None,
        help="If provided and labels are not binary, labels >= threshold are mapped to 1, else 0.",
    )
    args = parser.parse_args()

    if args.generate_synthetic_data and args.prepare_davis_data:
        parser.error("Use only one data-generation mode: --generate-synthetic-data or --prepare-davis-data.")
    if args.val_size <= 0 or args.test_size <= 0:
        parser.error("--val-size and --test-size must be > 0.")
    if args.val_size + args.test_size >= 1:
        parser.error("--val-size + --test-size must be < 1.")
    if args.davis_threshold_nm <= 0:
        parser.error("--davis-threshold-nm must be > 0.")

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def synthetic_smiles(rng: np.random.Generator, length: int) -> str:
    chars = rng.choice(SMILES_ALPHABET, size=length)
    return "".join(chars)


def synthetic_protein(rng: np.random.Generator, length: int) -> str:
    chars = rng.choice(PROTEIN_ALPHABET, size=length)
    return "".join(chars)


def generate_synthetic_dataset(path: Path, num_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for _ in range(num_samples):
        smiles = synthetic_smiles(rng, int(rng.integers(24, 90)))
        protein = synthetic_protein(rng, int(rng.integers(120, 650)))

        rule_1 = int("N" in smiles and "K" in protein)
        rule_2 = int("S" in smiles and "R" in protein)
        rule_3 = int("P" in smiles and "G" in protein)
        rule_4 = int(smiles.count("C") > 5 and protein.count("W") > 2)

        logit = -1.1 + 1.1 * rule_1 + 0.9 * rule_2 + 0.7 * rule_3 + 1.0 * rule_4
        logit += float(rng.normal(0.0, 0.7))
        label = int(rng.random() < sigmoid(logit))

        rows.append(
            {
                SMILES_COLUMN: smiles,
                PROTEIN_COLUMN: protein,
                LABEL_COLUMN: label,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_json_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Expected a non-empty JSON object in: {path}")
    return {str(key): str(value) for key, value in mapping.items()}


def prepare_davis_dataset(
    output_path: Path,
    smiles_path: Path,
    protein_path: Path,
    affinity_path: Path,
    threshold_nm: float,
    censor_value_nm: float,
    keep_censored: bool,
    seed: int,
) -> pd.DataFrame:
    smiles_map = load_json_mapping(smiles_path)
    protein_map = load_json_mapping(protein_path)
    smiles_items = list(smiles_map.items())
    protein_items = list(protein_map.items())

    if not affinity_path.exists():
        raise FileNotFoundError(f"Affinity matrix not found: {affinity_path}")
    affinity = np.loadtxt(affinity_path)

    expected_shape = (len(smiles_items), len(protein_items))
    if affinity.shape == (expected_shape[1], expected_shape[0]):
        affinity = affinity.T
    if affinity.shape != expected_shape:
        raise ValueError(
            f"DAVIS matrix shape mismatch: got {affinity.shape}, expected {expected_shape} "
            "(or its transpose)."
        )

    rows: list[dict[str, object]] = []
    for drug_idx, (_, smiles) in enumerate(smiles_items):
        for protein_idx, (_, protein_sequence) in enumerate(protein_items):
            affinity_nm = float(affinity[drug_idx, protein_idx])
            if not np.isfinite(affinity_nm):
                continue
            if not keep_censored and affinity_nm >= censor_value_nm:
                continue
            label = int(affinity_nm <= threshold_nm)
            rows.append(
                {
                    SMILES_COLUMN: smiles,
                    PROTEIN_COLUMN: protein_sequence,
                    LABEL_COLUMN: label,
                    AFFINITY_COLUMN: affinity_nm,
                }
            )

    if not rows:
        raise ValueError("No DAVIS rows were produced. Check censor/threshold settings.")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_dataset(df_path: Path, threshold: float | None) -> pd.DataFrame:
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset not found: {df_path}")

    df = pd.read_csv(df_path)
    required = {SMILES_COLUMN, PROTEIN_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Dataset missing required columns: {missing_txt}")

    df = df[[SMILES_COLUMN, PROTEIN_COLUMN, LABEL_COLUMN]].copy()
    df[SMILES_COLUMN] = df[SMILES_COLUMN].astype(str).str.replace(r"\s+", "", regex=True)
    df[PROTEIN_COLUMN] = (
        df[PROTEIN_COLUMN].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    )
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())
    is_binary = set(unique_labels).issubset({0, 1})
    if not is_binary:
        if threshold is None:
            raise ValueError(
                "Labels are not binary. Pass --binarize-threshold to map values into {0,1}."
            )
        df[LABEL_COLUMN] = (df[LABEL_COLUMN] >= threshold).astype(int)

    if len(df) < 50:
        raise ValueError("Need at least 50 rows for a stable train/validation/test split.")

    class_count = df[LABEL_COLUMN].value_counts()
    if len(class_count) < 2:
        raise ValueError("Dataset must contain at least one positive and one negative label.")

    return df


class CharVocab:
    def __init__(self, texts: list[str]):
        chars = sorted(set("".join(texts)))
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        for idx, ch in enumerate(chars, start=2):
            self.stoi[ch] = idx

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text[:max_len]]
        if len(tokens) < max_len:
            tokens.extend([self.stoi[self.pad_token]] * (max_len - len(tokens)))
        return tokens


class DTIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_vocab: CharVocab,
        protein_vocab: CharVocab,
        max_smiles_len: int,
        max_protein_len: int,
    ):
        self.smiles = torch.tensor(
            [smiles_vocab.encode(v, max_smiles_len) for v in df[SMILES_COLUMN].tolist()],
            dtype=torch.long,
        )
        self.proteins = torch.tensor(
            [protein_vocab.encode(v, max_protein_len) for v in df[PROTEIN_COLUMN].tolist()],
            dtype=torch.long,
        )
        self.labels = torch.tensor(df[LABEL_COLUMN].tolist(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.smiles[idx], self.proteins[idx], self.labels[idx]


class SequenceCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, conv_channels, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = conv_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled_features = []
        for conv in self.convs:
            feat = torch.relu(conv(emb))
            pooled_features.append(torch.max(feat, dim=2).values)
        return self.dropout(torch.cat(pooled_features, dim=1))


class DTICNNBaseline(nn.Module):
    def __init__(
        self,
        smiles_vocab_size: int,
        protein_vocab_size: int,
        embed_dim: int,
        conv_channels: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.drug_encoder = SequenceCNNEncoder(smiles_vocab_size, embed_dim, conv_channels, dropout=dropout)
        self.target_encoder = SequenceCNNEncoder(
            protein_vocab_size, embed_dim, conv_channels, dropout=dropout
        )
        joint_dim = self.drug_encoder.output_dim + self.target_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, smiles: torch.Tensor, proteins: torch.Tensor) -> torch.Tensor:
        drug_repr = self.drug_encoder(smiles)
        target_repr = self.target_encoder(proteins)
        logits = self.classifier(torch.cat([drug_repr, target_repr], dim=1))
        return logits.squeeze(1)


@dataclass
class EvalMetrics:
    loss: float
    auroc: float
    auprc: float
    accuracy: float
    f1: float


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float) -> EvalMetrics:
    y_pred = (y_prob >= 0.5).astype(int)
    unique = np.unique(y_true)

    if len(unique) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = float("nan")
        auprc = float("nan")

    return EvalMetrics(
        loss=float(loss),
        auroc=auroc,
        auprc=auprc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_count = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for smiles, proteins, y in data_loader:
        smiles = smiles.to(device)
        proteins = proteins.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(smiles, proteins)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        batch_prob = torch.sigmoid(logits.detach()).cpu().numpy()
        probs.append(batch_prob)
        labels.append(y.detach().cpu().numpy())

    mean_loss = total_loss / max(total_count, 1)
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    return mean_loss, y_true, y_prob


def append_results_row(
    path: Path,
    row: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        row_df.to_csv(path, index=False)


def write_markdown_table(csv_path: Path, md_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.copy()
    float_cols = ["val_auroc", "val_auprc", "test_auroc", "test_auprc", "test_accuracy", "test_f1"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = detect_device()
    run_started_at = datetime.now(timezone.utc)
    run_id = args.run_id or run_started_at.strftime("%Y%m%dT%H%M%SZ")
    run_output_dir = args.output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prepare_davis_data:
        print(
            f"[INFO] Preparing DAVIS dataset at {args.data_path} "
            f"(threshold<={args.davis_threshold_nm} nM, keep_censored={args.davis_keep_censored})"
        )
        davis_df = prepare_davis_dataset(
            output_path=args.data_path,
            smiles_path=args.davis_smiles_path,
            protein_path=args.davis_protein_path,
            affinity_path=args.davis_affinity_path,
            threshold_nm=args.davis_threshold_nm,
            censor_value_nm=args.davis_censor_value_nm,
            keep_censored=args.davis_keep_censored,
            seed=args.seed,
        )
        print(
            f"[INFO] DAVIS dataset rows={len(davis_df)} "
            f"pos_rate={davis_df[LABEL_COLUMN].mean():.4f}"
        )
    elif args.generate_synthetic_data or not args.data_path.exists():
        print(f"[INFO] Generating synthetic dataset at {args.data_path} ({args.synthetic_samples} rows)")
        generate_synthetic_dataset(args.data_path, args.synthetic_samples, args.seed)

    df = load_dataset(args.data_path, args.binarize_threshold)

    train_df, holdout_df = train_test_split(
        df,
        test_size=args.val_size + args.test_size,
        random_state=args.seed,
        stratify=df[LABEL_COLUMN],
    )
    relative_test = args.test_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        random_state=args.seed,
        stratify=holdout_df[LABEL_COLUMN],
    )

    smiles_vocab = CharVocab(train_df[SMILES_COLUMN].tolist())
    protein_vocab = CharVocab(train_df[PROTEIN_COLUMN].tolist())

    train_dataset = DTIDataset(
        train_df, smiles_vocab, protein_vocab, args.max_smiles_len, args.max_protein_len
    )
    val_dataset = DTIDataset(val_df, smiles_vocab, protein_vocab, args.max_smiles_len, args.max_protein_len)
    test_dataset = DTIDataset(
        test_df, smiles_vocab, protein_vocab, args.max_smiles_len, args.max_protein_len
    )

    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=data_loader_rng
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DTICNNBaseline(
        smiles_vocab_size=smiles_vocab.size,
        protein_vocab_size=protein_vocab.size,
        embed_dim=args.embed_dim,
        conv_channels=args.conv_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    positives = float(train_df[LABEL_COLUMN].sum())
    negatives = float(len(train_df) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_auroc = -1.0
    best_state_dict = None
    best_epoch_row: dict[str, object] | None = None
    epoch_rows: list[dict[str, object]] = []

    print(
        f"[INFO] Training on {device.type} | train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"| class_balance(train_pos_rate)={train_df[LABEL_COLUMN].mean():.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_y, train_prob = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_y, val_prob = run_epoch(model, val_loader, criterion, None, device)

        train_metrics = compute_classification_metrics(train_y, train_prob, train_loss)
        val_metrics = compute_classification_metrics(val_y, val_prob, val_loss)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_auroc": train_metrics.auroc,
            "train_auprc": train_metrics.auprc,
            "train_accuracy": train_metrics.accuracy,
            "train_f1": train_metrics.f1,
            "val_loss": val_metrics.loss,
            "val_auroc": val_metrics.auroc,
            "val_auprc": val_metrics.auprc,
            "val_accuracy": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
        }
        epoch_rows.append(epoch_row)

        if not np.isnan(val_metrics.auroc) and val_metrics.auroc > best_val_auroc:
            best_val_auroc = val_metrics.auroc
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch_row = dict(epoch_row)

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_auroc={format_metric(val_metrics.auroc)} val_auprc={format_metric(val_metrics.auprc)} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_y, test_prob = run_epoch(model, test_loader, criterion, None, device)
    test_metrics = compute_classification_metrics(test_y, test_prob, test_loss)

    model_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint = {
        "model_state_dict": model_state_cpu,
        "smiles_vocab": smiles_vocab.stoi,
        "protein_vocab": protein_vocab.stoi,
        "config": vars(args),
        "run_id": run_id,
    }
    run_model_path = run_output_dir / "baseline_cnn_weights.pt"
    torch.save(checkpoint, run_model_path)
    torch.save(checkpoint, args.model_path)

    log_path = run_output_dir / "training_log.csv"
    pd.DataFrame(epoch_rows).to_csv(log_path, index=False)

    predictions_path = run_output_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "y_true": test_y.astype(int),
            "y_prob": test_prob,
            "y_pred": (test_prob >= 0.5).astype(int),
        }
    ).to_csv(predictions_path, index=False)

    run_finished_at = datetime.now(timezone.utc)
    metrics_payload = {
        "timestamp_utc": run_finished_at.isoformat(),
        "run_id": run_id,
        "data_path": str(args.data_path),
        "device": device.type,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "class_balance": {
            "overall_pos_rate": float(df[LABEL_COLUMN].mean()),
            "train_pos_rate": float(train_df[LABEL_COLUMN].mean()),
            "val_pos_rate": float(val_df[LABEL_COLUMN].mean()),
            "test_pos_rate": float(test_df[LABEL_COLUMN].mean()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "best_epoch": None if best_epoch_row is None else int(best_epoch_row["epoch"]),
        "best_val_auroc": None if best_val_auroc < 0 else float(best_val_auroc),
        "test_metrics": {
            "loss": float(test_metrics.loss),
            "auroc": float(test_metrics.auroc),
            "auprc": float(test_metrics.auprc),
            "accuracy": float(test_metrics.accuracy),
            "f1": float(test_metrics.f1),
        },
        "artifacts": {
            "model_weights": str(run_model_path),
            "latest_model_weights": str(args.model_path),
            "training_log": str(log_path),
            "test_predictions": str(predictions_path),
        },
    }

    metrics_path = run_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary_row = {
        "timestamp_utc": metrics_payload["timestamp_utc"],
        "run_id": run_id,
        "dataset": str(args.data_path),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "val_auroc": metrics_payload["best_val_auroc"],
        "val_auprc": None if best_epoch_row is None else float(best_epoch_row["val_auprc"]),
        "test_auroc": float(test_metrics.auroc),
        "test_auprc": float(test_metrics.auprc),
        "test_accuracy": float(test_metrics.accuracy),
        "test_f1": float(test_metrics.f1),
        "model_weights": str(run_model_path),
        "latest_model_weights": str(args.model_path),
        "run_metrics": str(metrics_path),
    }
    append_results_row(args.results_csv, summary_row)
    write_markdown_table(args.results_csv, args.results_md)

    print("[INFO] Training complete.")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Model weights (run): {run_model_path}")
    print(f"[INFO] Model weights (latest): {args.model_path}")
    print(f"[INFO] Run metrics: {metrics_path}")
    print(f"[INFO] Baseline table CSV: {args.results_csv}")
    print(f"[INFO] Baseline table Markdown: {args.results_md}")


if __name__ == "__main__":
    main()
