import torch
import shap
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from beir.retrieval.evaluation import EvaluateRetrieval
from typing import List, Dict, Any

from src.features import CalculateFeatures
from src.utils import ensure_dir, save_json


ALPHAS = np.round(np.arange(0.0, 1.05, 0.05), 2)
K_VALUES = [10, 100]


class SHAPWrapper(nn.Module):
    """Wrapper to ensure the model outputs a 2D tensor for SHAP compatibility."""
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        out = self.original_model(x)
        if len(out.shape) == 1:
            out = out.unsqueeze(1)
        return out


def build_hybrid_results(dense_scores: Dict[str, Dict[str, float]],
                         sparse_scores: Dict[str, Dict[str, float]],
                         alpha: float) -> Dict[str, Dict[str, float]]:
    hybrid_results = {}
    all_qids = set(dense_scores.keys()) | set(sparse_scores.keys())

    for qid in all_qids:
        hybrid_results[qid] = {}
        dense_docs = dense_scores.get(qid, {})
        sparse_docs = sparse_scores.get(qid, {})
        candidate_docs = set(dense_docs.keys()) | set(sparse_docs.keys())

        for did in candidate_docs:
            s_d = dense_docs.get(did, 0.0)
            s_s = sparse_docs.get(did, 0.0)
            hybrid_results[qid][did] = float(alpha * s_d + (1.0 - alpha) * s_s)

    return hybrid_results


def serialize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}


def collect_ranking_metrics(evaluator: EvaluateRetrieval,
                            qrels: Dict[str, Dict[str, int]],
                            results: Dict[str, Dict[str, float]],
                            k_values: List[int]) -> Dict[str, Dict[str, float]]:
    ndcg_dict, map_dict, recall_dict, precision_dict = evaluator.evaluate(qrels, results, k_values)
    mrr_dict = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")

    return {
        "NDCG": serialize_metrics(ndcg_dict),
        "MAP": serialize_metrics(map_dict),
        "Recall": serialize_metrics(recall_dict),
        "Precision": serialize_metrics(precision_dict),
        "MRR": serialize_metrics(mrr_dict),
    }


def evaluate_static(retrieval_data: Dict[str, Any], datasets_list: List[str], output_dir: str = "results/plots"):
    """Calculates static Oracle baselines across alphas and saves full metrics."""
    evaluator = EvaluateRetrieval()

    results_ndcg_10 = {ds: [] for ds in datasets_list}
    results_mrr_10 = {ds: [] for ds in datasets_list}

    alpha_metrics = {ds: [] for ds in datasets_list}

    for ds in datasets_list:
        data = retrieval_data[ds]
        dense_norm = data["dense_norm"]
        sparse_norm = data["sparse_norm"]
        qrels = data["qrels"]

        print(f"[{ds.upper()}] Evaluating static alphas...")
        for alpha in ALPHAS:
            hybrid_results = build_hybrid_results(dense_norm, sparse_norm, float(alpha))
            metrics = collect_ranking_metrics(evaluator, qrels, hybrid_results, K_VALUES)

            results_ndcg_10[ds].append(metrics["NDCG"]["NDCG@10"])
            results_mrr_10[ds].append(metrics["MRR"]["MRR@10"])

            alpha_metrics[ds].append({
                "alpha": float(alpha),
                "metrics": metrics,
            })

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "#4D8A8C", "#38B00E"]

    for i, ds in enumerate(datasets_list):
        # NDCG@10 Plot
        ax1.plot(ALPHAS, results_ndcg_10[ds], label=ds.upper(), color=colors[i], marker='o', markersize=4)
        max_idx = np.argmax(results_ndcg_10[ds])
        ax1.scatter(ALPHAS[max_idx], results_ndcg_10[ds][max_idx], color=colors[i], s=100, marker='*')

        # MRR@10 Plot
        ax2.plot(ALPHAS, results_mrr_10[ds], label=ds.upper(), color=colors[i], marker='o', markersize=4)
        max_idx_mrr = np.argmax(results_mrr_10[ds])
        ax2.scatter(ALPHAS[max_idx_mrr], results_mrr_10[ds][max_idx_mrr], color=colors[i], s=100, marker='*')

    ax1.set_title("NDCG@10 by Dense Weight ($\\alpha$)", fontsize=14)
    ax1.set_xlabel("$\\alpha$ (0 = Pure BM25, 1 = Pure Dense)", fontsize=12)
    ax1.set_ylabel("NDCG@10", fontsize=12)
    ax1.set_xticks(ALPHAS)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title("MRR@10 by Dense Weight ($\\alpha$)", fontsize=14)
    ax2.set_xlabel("$\\alpha$ (0 = Pure BM25, 1 = Pure Dense)", fontsize=12)
    ax2.set_ylabel("MRR@10", fontsize=12)
    ax2.set_xticks(ALPHAS)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_dir = ensure_dir(output_dir)
    save_path = output_dir / "static_oracle_benchmark.pdf"
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"\nSaved static evaluation plot to {save_path}")

    metric_selection = {
        "ndcg_at_10": ("NDCG", "NDCG@10"),
        "ndcg_at_100": ("NDCG", "NDCG@100"),
        "map_at_10": ("MAP", "MAP@10"),
        "map_at_100": ("MAP", "MAP@100"),
        "recall_at_10": ("Recall", "Recall@10"),
        "recall_at_100": ("Recall", "Recall@100"),
        "precision_at_10": ("Precision", "P@10"),
        "precision_at_100": ("Precision", "P@100"),
        "mrr_at_10": ("MRR", "MRR@10"),
        "mrr_at_100": ("MRR", "MRR@100"),
    }

    best_by_dataset = {}
    for ds in datasets_list:
        best_by_metric = {}
        for metric_name, (family, key) in metric_selection.items():
            best_idx = int(np.argmax([entry["metrics"][family][key] for entry in alpha_metrics[ds]]))
            best_by_metric[metric_name] = {
                "alpha": alpha_metrics[ds][best_idx]["alpha"],
                "value": alpha_metrics[ds][best_idx]["metrics"][family][key],
            }

        best_by_dataset[ds] = {
            "best_by_metric": best_by_metric,
            "alpha_metrics": alpha_metrics[ds],
        }

    return {
        "plot_path": str(save_path),
        "k_values": K_VALUES,
        "alpha_grid": [float(alpha) for alpha in ALPHAS],
        "best_by_dataset": best_by_dataset,
    }


def evaluate_dynamic_router(model, retrieval_data: Dict[str, Any], datasets: List[str], device: str = 'cpu'):
    """Evaluates the neural router on zero-shot test datasets using multiple k-cutoffs."""
    model.to(device)
    model.eval()
    evaluator = EvaluateRetrieval()
    dynamic_results = {}

    with torch.no_grad():
        for ds in datasets:
            print(f"\n[TESTING] {ds.upper()}...")
            data = retrieval_data[ds]
            queries = data["queries"]
            qrels = data["qrels"]
            dense_norm = data["dense_norm"]
            sparse_norm = data["sparse_norm"]
            idf_dict = data["idf_dict"]
            vocab_set = data["vocab_set"]

            hybrid_results = {}
            predicted_alphas = []

            for qid, q_text in queries.items():
                if qid not in qrels:
                    continue

                x_q = CalculateFeatures.extract_features(q_text, idf_dict, vocab_set)
                x_q_tensor = torch.tensor(x_q, dtype=torch.float32).unsqueeze(0).to(device)

                alpha = model(x_q_tensor).item()
                predicted_alphas.append(alpha)

                candidate_docs = set(dense_norm.get(qid, {}).keys()) | set(sparse_norm.get(qid, {}).keys())
                hybrid_results[qid] = {}

                for did in candidate_docs:
                    s_d = dense_norm.get(qid, {}).get(did, 0.0)
                    s_s = sparse_norm.get(qid, {}).get(did, 0.0)
                    hybrid_results[qid][did] = alpha * s_d + (1.0 - alpha) * s_s

            metrics = collect_ranking_metrics(evaluator, qrels, hybrid_results, K_VALUES)

            mean_alpha = np.mean(predicted_alphas)
            std_alpha = np.std(predicted_alphas)

            dynamic_results[ds] = {
                "metrics": metrics,
                "mean_alpha": float(mean_alpha),
                "std_alpha": float(std_alpha)
            }

            print(
                f"NDCG@10:    {metrics['NDCG']['NDCG@10']:.4f}  |  "
                f"NDCG@100: {metrics['NDCG']['NDCG@100']:.4f}"
            )
            print(
                f"MRR@10:     {metrics['MRR']['MRR@10']:.4f}  |  "
                f"MRR@100:  {metrics['MRR']['MRR@100']:.4f}"
            )
            print(
                f"Recall@10:  {metrics['Recall']['Recall@10']:.4f}  |  "
                f"Recall@100: {metrics['Recall']['Recall@100']:.4f}"
            )
            print(f"Alpha stats: mean = {mean_alpha:.4f}, std = {std_alpha:.4f}")

    return {
        "k_values": K_VALUES,
        "datasets": dynamic_results,
    }


def get_plot_shap(model, retrieval_data: Dict[str, Any], datasets: List[str], output_dir: str = "results/plots"):
    """Generates and saves a SHAP summary plot for model interpretability."""
    print("\nExtracting features for SHAP analysis...")
    model.cpu()
    model.eval()
    shap_model = SHAPWrapper(model)

    all_features = []
    for ds in datasets:
        data = retrieval_data[ds]
        queries = data["queries"]
        idf_dict = data["idf_dict"]
        vocab_set = data["vocab_set"]

        for qid, q_text in queries.items():
            feat = CalculateFeatures.extract_features(q_text, idf_dict, vocab_set)
            all_features.append(feat)

    sample_size = min(2000, len(all_features))
    sampled_features = random.sample(all_features, sample_size)

    X_tensor = torch.tensor(sampled_features, dtype=torch.float32)

    background = X_tensor[:100]
    test_samples = X_tensor[100:]

    print("Running DeepExplainer (this might take a few seconds)...")
    explainer = shap.DeepExplainer(shap_model, background)
    shap_values = explainer.shap_values(test_samples, check_additivity=False)

    feature_names = [
        "q_len", "mean_idf", "max_idf", "min_idf", "std_idf",
        "rare_ratio", "oov_ratio", "idf_skewness", "query_entropy", "digit_ratio",
        "upper_ratio", "punct_ratio", "stopword_ratio", "noun_ratio",
        "verb_ratio", "adj_ratio", "avg_word_len"
    ]

    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_vals_2d = np.array(shap_values).reshape(-1, 17)
    test_samples_2d = test_samples.numpy().reshape(-1, 17)

    shap.summary_plot(shap_vals_2d, test_samples_2d, feature_names=feature_names, show=False)

    output_dir = ensure_dir(output_dir)
    model_name = model.__class__.__name__.lower()
    save_path = output_dir / f"shap_summary_{model_name}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved SHAP summary plot to {save_path}")
    plt.close()
    return str(save_path)


def save_evaluation_summary(
        output_dir: str,
        static_summary: Dict[str, Any],
        dynamic_summaries: Dict[str, Any],
        shap_plots: Dict[str, str],
) -> str:
    output_dir_path = ensure_dir(output_dir)
    summary_path = output_dir_path / "evaluation_summary.json"
    payload = {
        "static_oracle": static_summary,
        "dynamic_routers": dynamic_summaries,
        "shap_plots": shap_plots,
    }
    save_json(payload, summary_path)
    return str(summary_path)
