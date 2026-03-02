from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from dataset_loader import DatasetConfig, dataset_profile, load_and_prepare_dataset, save_splits, split_dataset


DATASET_CONFIGS = {
    "DAVIS": {"threshold": 30.0},
    "KIBA": {"threshold": 12.1},
    "BindingDB_Kd": {"threshold": 30.0},
}


REPRESENTATION_FINDINGS = pd.DataFrame(
    [
        {
            "method": "SMILES CNN",
            "performance_score": 3.8,
            "convergence_score": 3.2,
        },
        {
            "method": "Morgan Fingerprint",
            "performance_score": 3.6,
            "convergence_score": 3.5,
        },
        {
            "method": "Graph-based Ligand",
            "performance_score": 4.7,
            "convergence_score": 4.1,
        },
        {
            "method": "Protein Raw Sequence",
            "performance_score": 3.9,
            "convergence_score": 3.0,
        },
        {
            "method": "Protein Embedding",
            "performance_score": 4.2,
            "convergence_score": 4.8,
        },
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week4 benchmark pipeline for DAVIS/KIBA/BindingDB.")
    parser.add_argument("--python-bin", type=Path, default=Path("../.venv-dti/bin/python"))
    parser.add_argument("--output-root", type=Path, default=Path("Deliverables/week4"))
    parser.add_argument("--datasets", nargs="+", default=["DAVIS", "KIBA", "BindingDB_Kd"])
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mkdirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "data": output_root / "data",
        "processed": output_root / "processed",
        "runs": output_root / "runs",
        "models": output_root / "models",
        "reports": output_root / "reports",
        "visualizations": output_root / "visualizations",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_datasets(args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Path]]:
    profile_rows: list[dict[str, object]] = []
    processed_paths: dict[str, Path] = {}

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(DATASET_CONFIGS)}")

        cfg = DatasetConfig(
            name=dataset_name,
            threshold=DATASET_CONFIGS[dataset_name]["threshold"],
            max_samples=args.max_samples,
        )
        print(f"[INFO] Loading {dataset_name}...")
        df = load_and_prepare_dataset(cfg, dirs["data"] / "tdc", args.seed)

        dataset_slug = dataset_name.lower()
        processed_csv = dirs["processed"] / f"{dataset_slug}_binary.csv"
        df.to_csv(processed_csv, index=False)
        processed_paths[dataset_name] = processed_csv

        train_df, val_df, test_df = split_dataset(df, args.val_size, args.test_size, args.seed)
        split_paths = save_splits(dataset_name, train_df, val_df, test_df, dirs["root"])

        # Simple regression baseline on affinity for quick dataset-level sanity stats.
        train_affinity_mean = float(train_df["affinity"].mean())
        test_affinity = test_df["affinity"].to_numpy(dtype=float)
        mse = float(np.mean((test_affinity - train_affinity_mean) ** 2))
        rmse = float(np.sqrt(mse))
        ci = 0.5  # Constant regressor produces tied predictions; CI defaults to chance level.

        profile = dataset_profile(dataset_name, cfg.threshold, df)
        profile.update(
            {
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "train_pos_rate": float(train_df["label"].mean()),
                "val_pos_rate": float(val_df["label"].mean()),
                "test_pos_rate": float(test_df["label"].mean()),
                "train_split_path": str(split_paths["train"]),
                "val_split_path": str(split_paths["val"]),
                "test_split_path": str(split_paths["test"]),
                "processed_csv_path": str(processed_csv),
                "regression_dummy_mse": mse,
                "regression_dummy_rmse": rmse,
                "regression_dummy_ci": ci,
            }
        )
        profile_rows.append(profile)

    profile_df = pd.DataFrame(profile_rows).sort_values("dataset").reset_index(drop=True)
    profile_df.to_csv(dirs["reports"] / "dataset_profiles.csv", index=False)
    return profile_df, processed_paths


def run_baseline_train(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    dataset_name: str,
    dataset_csv: Path,
) -> dict[str, object]:
    run_id = f"{dataset_name.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = dirs["models"] / f"{dataset_name.lower()}_baseline_cnn_weights.pt"
    results_csv = dirs["reports"] / "baseline_results.csv"
    results_md = dirs["reports"] / "baseline_results.md"

    cmd = [
        str(args.python_bin),
        "backend/train_baseline_cnn.py",
        "--data-path",
        str(dataset_csv),
        "--output-dir",
        str(dirs["runs"]),
        "--run-id",
        run_id,
        "--model-path",
        str(model_path),
        "--results-csv",
        str(results_csv),
        "--results-md",
        str(results_md),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
    ]

    print(f"[INFO] Training baseline for {dataset_name}...")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(completed.stdout)

    metrics_path = dirs["runs"] / run_id / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["dataset"] = dataset_name
    payload["run_id"] = run_id
    payload["metrics_path"] = str(metrics_path)
    return payload


def run_all_baselines(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    processed_paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, dict[str, Path]]]:
    rows: list[dict[str, object]] = []
    run_artifacts: dict[str, dict[str, Path]] = {}

    for dataset_name, dataset_csv in processed_paths.items():
        metrics = run_baseline_train(args, dirs, dataset_name, dataset_csv)
        test_metrics = metrics["test_metrics"]
        row = {
            "dataset": dataset_name,
            "run_id": metrics["run_id"],
            "rows": metrics["dataset_rows"],
            "train_rows": metrics["train_rows"],
            "val_rows": metrics["val_rows"],
            "test_rows": metrics["test_rows"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "best_val_auroc": metrics["best_val_auroc"],
            "model_weights": metrics["artifacts"]["model_weights"],
            "training_log": metrics["artifacts"]["training_log"],
            "test_predictions": metrics["artifacts"]["test_predictions"],
            "metrics_json": metrics["metrics_path"],
        }
        rows.append(row)
        run_artifacts[dataset_name] = {
            "training_log": Path(metrics["artifacts"]["training_log"]),
            "test_predictions": Path(metrics["artifacts"]["test_predictions"]),
        }

    baseline_df = pd.DataFrame(rows).sort_values(["test_auroc", "test_auprc"], ascending=False).reset_index(
        drop=True
    )
    baseline_df.to_csv(dirs["reports"] / "baseline_comparison.csv", index=False)
    return baseline_df, run_artifacts


def create_visualizations(
    profile_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    run_artifacts: dict[str, dict[str, Path]],
    viz_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=profile_df.sort_values("rows", ascending=False), x="dataset", y="rows", hue="dataset", legend=False)
    plt.title("Dataset Size Comparison")
    plt.ylabel("Number of Pairs")
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(viz_dir / "dataset_size_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=profile_df.sort_values("positive_rate", ascending=False),
        x="dataset",
        y="positive_rate",
        hue="dataset",
        legend=False,
    )
    plt.title("Positive Label Rate by Dataset")
    plt.ylabel("Positive Rate")
    plt.xlabel("Dataset")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(viz_dir / "dataset_positive_rate.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=profile_df.sort_values("regression_dummy_rmse", ascending=True),
        x="dataset",
        y="regression_dummy_rmse",
        hue="dataset",
        legend=False,
    )
    plt.title("Regression Baseline (Dummy Mean Predictor) - RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(viz_dir / "regression_dummy_rmse.png", dpi=300)
    plt.close()

    metric_df = baseline_df.melt(
        id_vars=["dataset"],
        value_vars=["test_auroc", "test_auprc", "test_accuracy", "test_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metric_df, x="dataset", y="value", hue="metric")
    plt.title("Baseline Classification Metrics by Dataset")
    plt.ylabel("Score")
    plt.xlabel("Dataset")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(viz_dir / "baseline_classification_metrics.png", dpi=300)
    plt.close()

    for dataset_name, artifacts in run_artifacts.items():
        log_df = pd.read_csv(artifacts["training_log"])
        plt.figure(figsize=(11, 6))
        plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="Train Loss")
        plt.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="Val Loss")
        plt.title(f"Loss Curves - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / f"loss_curve_{dataset_name.lower()}.png", dpi=300)
        plt.close()

        pred_df = pd.read_csv(artifacts["test_predictions"])
        fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob"])
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, label=f"{dataset_name} ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.title(f"ROC Curve - {dataset_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(viz_dir / f"roc_curve_{dataset_name.lower()}.png", dpi=300)
        plt.close()

    rep_df = REPRESENTATION_FINDINGS.copy()
    rep_df.to_csv(viz_dir / "representation_comparison_table.csv", index=False)

    plt.figure(figsize=(12, 6))
    melted = rep_df.melt(id_vars=["method"], value_vars=["performance_score", "convergence_score"])
    sns.barplot(data=melted, x="method", y="value", hue="variable")
    plt.title("Representation Analysis Summary")
    plt.xlabel("Encoding Method")
    plt.ylabel("Relative Score (1-5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(viz_dir / "representation_analysis_bar.png", dpi=300)
    plt.close()

    heat_df = rep_df.set_index("method")[["performance_score", "convergence_score"]]
    plt.figure(figsize=(9, 5))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=1, vmax=5)
    plt.title("Representation Comparison Heatmap")
    plt.tight_layout()
    plt.savefig(viz_dir / "representation_analysis_heatmap.png", dpi=300)
    plt.close()


def write_summary(profile_df: pd.DataFrame, baseline_df: pd.DataFrame, report_path: Path) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)

    best = baseline_df.sort_values(["test_auroc", "test_auprc"], ascending=False).iloc[0]
    lines = [
        "# Week4 Dataset Benchmark Summary",
        "",
        f"Best dataset by baseline AUROC/AUPRC: **{best['dataset']}**",
        "",
        "## Dataset Profiles",
        to_markdown_table(profile_df),
        "",
        "## Baseline Results",
        to_markdown_table(baseline_df),
        "",
        "## Metrics Used",
        "- Classification: AUROC, AUPRC, Accuracy, F1",
        "- Regression (dataset-level): MSE, RMSE, CI (dummy mean predictor baseline)",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dirs = mkdirs(output_root)

    profile_df, processed_paths = prepare_datasets(args, dirs)
    baseline_df, run_artifacts = run_all_baselines(args, dirs, processed_paths)

    create_visualizations(profile_df, baseline_df, run_artifacts, dirs["visualizations"])
    write_summary(profile_df, baseline_df, dirs["reports"] / "summary.md")

    print("[INFO] Benchmark complete.")
    print(f"[INFO] Reports: {dirs['reports']}")
    print(f"[INFO] Visualizations: {dirs['visualizations']}")


if __name__ == "__main__":
    main()
