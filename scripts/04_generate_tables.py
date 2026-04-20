import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_METRICS = ["NDCG@10", "MRR@10", "Recall@100"]
DEFAULT_MODELS = [
    ("bm25", r"BM25 ($\alpha=0$)"),
    ("dense", r"Dense ($\alpha=1$)"),
    ("alpharouter_advanced", "AlphaRouter (Adv)"),
    ("alpharouter_mlp", "AlphaRouter (MLP)"),
    ("static_oracle", "Static Oracle"),
]
FAMILY_BY_PREFIX = {
    "NDCG": "NDCG",
    "MAP": "MAP",
    "Recall": "Recall",
    "P": "Precision",
    "MRR": "MRR",
}


class TableGenerationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from evaluation_summary.json.")
    parser.add_argument("--input", default="results/plots/evaluation_summary.json", help="Path to evaluation summary JSON.")
    parser.add_argument("--output", default="results/latex_tables.tex", help="Path to output LaTeX file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric keys to render, e.g. NDCG@10 MRR@10 Recall@100 MAP@10 P@10.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset subset/order. Defaults to the order recorded in the evaluation summary.",
    )
    parser.add_argument(
        "--caption-template",
        default=(
            "Comparison of {metric_key} across {dataset_count} BEIR datasets. "
            "Best practical results are in \\textbf{{bold}}; static oracle upper bounds are in \\textit{{italics}}."
        ),
        help="Caption template. Available fields: {metric_key}, {metric_family}, {dataset_count}.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise TableGenerationError(f"Evaluation summary not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise TableGenerationError(f"Invalid JSON in {path}: {exc}") from exc


def infer_metric_family(metric_key: str) -> str:
    prefix = metric_key.split("@", 1)[0]
    if prefix not in FAMILY_BY_PREFIX:
        raise TableGenerationError(
            f"Unsupported metric key '{metric_key}'. Expected one of: "
            f"{', '.join(sorted(FAMILY_BY_PREFIX.keys()))} with @k suffix."
        )
    return FAMILY_BY_PREFIX[prefix]


def require_mapping(mapping: Dict, key: str, context: str):
    if key not in mapping:
        raise TableGenerationError(f"Missing key '{key}' in {context}.")
    return mapping[key]


def validate_summary(data: Dict) -> None:
    static_oracle = require_mapping(data, "static_oracle", "evaluation summary")
    dynamic_routers = require_mapping(data, "dynamic_routers", "evaluation summary")

    require_mapping(static_oracle, "best_by_dataset", "static_oracle")
    require_mapping(static_oracle, "alpha_grid", "static_oracle")
    require_mapping(dynamic_routers, "alpharouter_advanced", "dynamic_routers")
    require_mapping(dynamic_routers, "alpharouter_mlp", "dynamic_routers")
    require_mapping(dynamic_routers["alpharouter_advanced"], "datasets", "dynamic_routers.alpharouter_advanced")
    require_mapping(dynamic_routers["alpharouter_mlp"], "datasets", "dynamic_routers.alpharouter_mlp")


def resolve_dataset_order(data: Dict, requested_datasets: List[str] | None) -> List[str]:
    mlp_datasets = list(data["dynamic_routers"]["alpharouter_mlp"]["datasets"].keys())
    if requested_datasets is None:
        return mlp_datasets

    missing = [dataset for dataset in requested_datasets if dataset not in data["static_oracle"]["best_by_dataset"]]
    if missing:
        raise TableGenerationError(f"Requested datasets not found in evaluation summary: {', '.join(missing)}")
    return requested_datasets


def find_alpha_entry(alpha_metrics: Iterable[Dict], target_alpha: float, dataset: str) -> Dict:
    for entry in alpha_metrics:
        if abs(entry["alpha"] - target_alpha) < 1e-9:
            return entry
    raise TableGenerationError(f"Alpha {target_alpha} not found in static oracle metrics for dataset '{dataset}'.")


def get_metric_value(metrics: Dict, family: str, key: str, context: str) -> float:
    family_metrics = require_mapping(metrics, family, context)
    if key not in family_metrics:
        raise TableGenerationError(f"Missing metric '{key}' in {context}.{family}.")
    return float(family_metrics[key])


def collect_dataset_values(data: Dict, dataset: str, metric_family: str, metric_key: str) -> Dict[str, float]:
    static_dataset = require_mapping(data["static_oracle"]["best_by_dataset"], dataset, "static_oracle.best_by_dataset")
    alpha_metrics = require_mapping(static_dataset, "alpha_metrics", f"static_oracle.best_by_dataset.{dataset}")

    bm25_entry = find_alpha_entry(alpha_metrics, 0.0, dataset)
    dense_entry = find_alpha_entry(alpha_metrics, 1.0, dataset)

    bm25_value = get_metric_value(bm25_entry["metrics"], metric_family, metric_key, f"{dataset}.bm25")
    dense_value = get_metric_value(dense_entry["metrics"], metric_family, metric_key, f"{dataset}.dense")
    oracle_value = max(
        get_metric_value(entry["metrics"], metric_family, metric_key, f"{dataset}.static_oracle")
        for entry in alpha_metrics
    )

    advanced_metrics = require_mapping(
        require_mapping(data["dynamic_routers"]["alpharouter_advanced"]["datasets"], dataset, "dynamic_routers.alpharouter_advanced.datasets"),
        "metrics",
        f"dynamic_routers.alpharouter_advanced.datasets.{dataset}",
    )
    mlp_metrics = require_mapping(
        require_mapping(data["dynamic_routers"]["alpharouter_mlp"]["datasets"], dataset, "dynamic_routers.alpharouter_mlp.datasets"),
        "metrics",
        f"dynamic_routers.alpharouter_mlp.datasets.{dataset}",
    )

    return {
        "bm25": bm25_value,
        "dense": dense_value,
        "alpharouter_advanced": get_metric_value(advanced_metrics, metric_family, metric_key, f"{dataset}.alpharouter_advanced"),
        "alpharouter_mlp": get_metric_value(mlp_metrics, metric_family, metric_key, f"{dataset}.alpharouter_mlp"),
        "static_oracle": oracle_value,
    }


def format_value(value: float, best_practical: float, is_oracle: bool) -> str:
    rounded = f"{value:.4f}"
    if is_oracle:
        return rf"\textit{{{rounded}}}"
    if abs(value - best_practical) < 1e-12:
        return rf"\textbf{{{rounded}}}"
    return rounded


def build_table(data: Dict, datasets: List[str], metric_key: str, caption_template: str) -> str:
    metric_family = infer_metric_family(metric_key)
    dataset_headers = [dataset.upper() for dataset in datasets]
    rows = {row_id: [] for row_id, _ in DEFAULT_MODELS}

    for dataset in datasets:
        values = collect_dataset_values(data, dataset, metric_family, metric_key)
        best_practical = max(
            values[row_id]
            for row_id in ("bm25", "dense", "alpharouter_advanced", "alpharouter_mlp")
        )
        for row_id, _ in DEFAULT_MODELS:
            rows[row_id].append(
                format_value(
                    values[row_id],
                    best_practical=best_practical,
                    is_oracle=(row_id == "static_oracle"),
                )
            )

    caption = caption_template.format(
        metric_key=metric_key,
        metric_family=metric_family,
        dataset_count=len(datasets),
    )
    label = f"tab:{metric_key.replace('@', '_').replace('.', '_').lower()}"
    col_format = "l" + "c" * len(datasets)

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        "Model & " + " & ".join(dataset_headers) + r" \\",
        r"\midrule",
        ]

    for row_index, (row_id, row_label) in enumerate(DEFAULT_MODELS):
        latex_lines.append(f"{row_label} & " + " & ".join(rows[row_id]) + r" \\")
        if row_index in {1, 3}:
            latex_lines.append(r"\midrule")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(latex_lines)


def build_alpha_stats_table(data: Dict, datasets: List[str]) -> str:
    """Generates the Appendix table for Mean and Std of Alpha."""
    caption = r"Mean and standard deviation of predicted fusion weights ($\alpha$) across datasets, demonstrating the reduced variance and domain-aware behavior of the MLP router compared to the Advanced model."
    label = "tab:alpha_stats"

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Mean $\alpha$ (MLP)} & \textbf{Std $\alpha$ (MLP)} & \textbf{Mean $\alpha$ (Adv)} & \textbf{Std $\alpha$ (Adv)} \\",
        r"\midrule",
    ]

    for dataset in datasets:
        mlp_data = require_mapping(
            data["dynamic_routers"]["alpharouter_mlp"]["datasets"], dataset, "alpharouter_mlp.datasets"
        )
        adv_data = require_mapping(
            data["dynamic_routers"]["alpharouter_advanced"]["datasets"], dataset, "alpharouter_advanced.datasets"
        )

        mean_mlp = float(require_mapping(mlp_data, "mean_alpha", f"{dataset}.mlp"))
        std_mlp = float(require_mapping(mlp_data, "std_alpha", f"{dataset}.mlp"))
        mean_adv = float(require_mapping(adv_data, "mean_alpha", f"{dataset}.adv"))
        std_adv = float(require_mapping(adv_data, "std_alpha", f"{dataset}.adv"))

        # Pretty formatting for dataset names
        dataset_name = dataset.upper()
        if dataset_name == "SCIFACT": dataset_name = "SciFact"
        elif dataset_name == "NFCORPUS": dataset_name = "NFCorpus"
        elif dataset_name == "QUORA": dataset_name = "Quora"

        row = rf"{dataset_name} & {mean_mlp:.3f} & {std_mlp:.3f} & {mean_adv:.3f} & {std_adv:.3f} \\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(latex_lines)


def write_tables(output_path: Path, tables: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n\n" + ("%" * 60) + "\n\n"
    output_path.write_text(separator.join(tables) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    summary_path = Path(args.input)
    output_path = Path(args.output)

    data = load_json(summary_path)
    validate_summary(data)
    datasets = resolve_dataset_order(data, args.datasets)

    tables = []

    # 1. Generate metric tables (NDCG, MRR, Recall, etc.)
    for metric_key in args.metrics:
        table = build_table(
            data=data,
            datasets=datasets,
            metric_key=metric_key,
            caption_template=args.caption_template,
        )
        tables.append(table)
        print(f"Generated table for {metric_key}")

    # 2. Generate the new Alpha stats table for the Appendix
    alpha_stats_table = build_alpha_stats_table(data, datasets)
    tables.append(alpha_stats_table)
    print(f"Generated table for Alpha Statistics (Appendix)")

    # 3. Write all to file
    write_tables(output_path, tables)
    print(f"Saved {len(tables)} table(s) to {output_path}")


if __name__ == "__main__":
    main()