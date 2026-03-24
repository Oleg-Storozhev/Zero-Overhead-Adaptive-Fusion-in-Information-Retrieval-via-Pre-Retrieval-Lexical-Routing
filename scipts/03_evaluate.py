import os
import sys
import pickle
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models import AlphaRouter, AlphaRouterMLP
from src.evaluation import evaluate_static, evaluate_dynamic_router, get_plot_shap


def load_model(model_class, model_path, input_dim=17, device='cpu'):
    """Instantiates the model architecture and loads the saved weights."""
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device.upper()}")

    models_dir = os.path.join(PROJECT_ROOT, "models")

    # --- Load Advanced Model ---
    print("\nLoading models...")
    adv_path = os.path.join(models_dir, "alpharouter_advanced_v1.pth")
    alpharouter_advanced = load_model(AlphaRouter, adv_path, device=device)
    print(f"Advanced model loaded from {adv_path}")

    # --- Load MLP Model ---
    mlp_path = os.path.join(models_dir, "alpharouter_mlp_v1.pth")
    alpharouter_mlp = load_model(AlphaRouterMLP, mlp_path, device=device)
    print(f"MLP model loaded from {mlp_path}")

    # --- Load Cached Test Data ---
    cache_path = os.path.join(PROJECT_ROOT, "cached_data", "test_retrieval_data_cached_v1.pkl")
    print(f"\nLoading test data from {cache_path}...")

    with open(cache_path, "rb") as f:
        test_retrieval_data = pickle.load(f)
    print("Data loaded successfully.")

    datasets_list = ["scifact", "fiqa", "nfcorpus", "scidocs", "nq", "quora", "msmarco"]

    print("RUNNING STATIC ORACLE EVALUATION")
    evaluate_static(retrieval_data=test_retrieval_data, datasets_list=datasets_list)

    print("RUNNING DYNAMIC EVALUATION: ADVANCED ROUTER")
    evaluate_dynamic_router(model=alpharouter_advanced,
                            retrieval_data=test_retrieval_data,
                            datasets=datasets_list,
                            device=device)

    print("RUNNING DYNAMIC EVALUATION: MLP ROUTER")
    evaluate_dynamic_router(model=alpharouter_mlp,
                            retrieval_data=test_retrieval_data,
                            datasets=datasets_list,
                            device=device)

    # --- Generate SHAP Plots ---
    print("\n--- Generating SHAP Plots ---")
    get_plot_shap(model=alpharouter_mlp,
                  retrieval_data=test_retrieval_data,
                  datasets=datasets_list,
                  output_dir="results/plots")


if __name__ == "__main__":
    main()
