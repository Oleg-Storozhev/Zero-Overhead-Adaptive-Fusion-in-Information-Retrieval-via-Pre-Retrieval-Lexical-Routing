import os
import sys
import pickle
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import ensure_dir, set_global_seed


def load_model(model_class, model_path, input_dim=17, device='cpu'):
    """Instantiates the model architecture and loads the saved weights."""
    import torch

    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AlphaRouter models on BEIR test datasets.")
    parser.add_argument("--cache-path", default=os.path.join(PROJECT_ROOT, "cached_data", "test_retrieval_data_cached_v1.pkl"))
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "plots"))
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "nfcorpus", "scidocs", "nq", "quora", "msmarco"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    from src.evaluation import evaluate_static, evaluate_dynamic_router, get_plot_shap, save_evaluation_summary
    from src.models import AlphaRouter, AlphaRouterMLP

    set_global_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device.upper()}")

    models_dir = args.models_dir
    output_dir = ensure_dir(args.output_dir)

    # --- Load Advanced Model ---
    print("\nLoading models...")
    adv_path = os.path.join(models_dir, "alpharouter_advanced_v1.pth")
    if not os.path.exists(adv_path):
        raise FileNotFoundError(f"Advanced model weights not found at {adv_path}. Run training first.")
    alpharouter_advanced = load_model(AlphaRouter, adv_path, device=device)
    print(f"Advanced model loaded from {adv_path}")

    # --- Load MLP Model ---
    mlp_path = os.path.join(models_dir, "alpharouter_mlp_v1.pth")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"MLP model weights not found at {mlp_path}. Run training first.")
    alpharouter_mlp = load_model(AlphaRouterMLP, mlp_path, device=device)
    print(f"MLP model loaded from {mlp_path}")

    # --- Load Cached Test Data ---
    cache_path = args.cache_path
    print(f"\nLoading test data from {cache_path}...")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Test cache not found at {cache_path}. Run scripts/01_prepare_data.py first "
            "or pass --cache-path explicitly."
        )

    with open(cache_path, "rb") as f:
        test_retrieval_data = pickle.load(f)
    print("Data loaded successfully.")

    datasets_list = args.datasets

    print("RUNNING STATIC ORACLE EVALUATION")
    static_summary = evaluate_static(
        retrieval_data=test_retrieval_data,
        datasets_list=datasets_list,
        output_dir=output_dir,
    )

    print("RUNNING DYNAMIC EVALUATION: ADVANCED ROUTER")
    advanced_results = evaluate_dynamic_router(model=alpharouter_advanced,
                                               retrieval_data=test_retrieval_data,
                                               datasets=datasets_list,
                                               device=device)

    print("RUNNING DYNAMIC EVALUATION: MLP ROUTER")
    mlp_results = evaluate_dynamic_router(model=alpharouter_mlp,
                                          retrieval_data=test_retrieval_data,
                                          datasets=datasets_list,
                                          device=device)

    # --- Generate SHAP Plots ---
    print("\n--- Generating SHAP Plots ---")
    shap_paths = {
        "alpharouter_advanced": get_plot_shap(model=alpharouter_advanced,
                                              retrieval_data=test_retrieval_data,
                                              datasets=datasets_list,
                                              output_dir=output_dir),
        "alpharouter_mlp": get_plot_shap(model=alpharouter_mlp,
                                         retrieval_data=test_retrieval_data,
                                         datasets=datasets_list,
                                         output_dir=output_dir),
    }

    summary_path = save_evaluation_summary(
        output_dir=output_dir,
        static_summary=static_summary,
        dynamic_summaries={
            "alpharouter_advanced": advanced_results,
            "alpharouter_mlp": mlp_results,
        },
        shap_plots=shap_paths,
    )
    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
