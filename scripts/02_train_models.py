import argparse
import sys
import os
import random
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import ensure_dir, save_json, set_global_seed, timestamp_utc


def train_router(model, dataloader, epochs=32, lr=1e-3, device='cuda'):
    import torch

    from src.loss import infonce_hybrid_loss

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    early_stopping_threshold = 5
    early_stopping_count = 0
    best_loss = float('inf')
    min_delta = 1e-4

    training_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            x_q = batch['x_q'].to(device)
            s_d_pos = batch['s_d_pos'].to(device)
            s_s_pos = batch['s_s_pos'].to(device)
            s_d_neg = batch['s_d_neg'].to(device)
            s_s_neg = batch['s_s_neg'].to(device)

            alpha = model(x_q)

            loss = infonce_hybrid_loss(
                alpha,
                s_d_pos, s_s_pos,
                s_d_neg, s_s_neg,
                tau=0.1
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        training_history.append({"epoch": epoch + 1, "loss": float(avg_loss)})

        if early_stopping_count >= early_stopping_threshold:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

    return model, {
        "best_loss": float(best_loss),
        "epochs_completed": len(training_history),
        "training_history": training_history,
        "early_stopping_threshold": early_stopping_threshold,
        "min_delta": min_delta,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaRouter models.")
    parser.add_argument("--cache-path", default=os.path.join(PROJECT_ROOT, "cached_data", "train_retrieval_data_cached_v1.pkl"))
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--train-datasets", nargs="+", default=["scifact", "fiqa", "nfcorpus", "msmarco"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples-per-dataset", type=int, default=15000)
    parser.add_argument("--num-negatives", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from src.features import CalculateFeatures
    from src.models import AlphaRouter, HybridLTRDataset, AlphaRouterMLP

    set_global_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device.upper()}")

    # LOAD CACHED DATA
    cache_path = args.cache_path
    print(f"Loading data from {cache_path}...")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Training cache not found at {cache_path}. Run scripts/01_prepare_data.py first "
            "or pass --cache-path explicitly."
        )

    with open(cache_path, "rb") as f:
        train_retrieval_data = pickle.load(f)

    train_datasets = args.train_datasets
    global_triplets = []
    max_samples_per_dataset = args.max_samples_per_dataset

    print("\n--- Generating Features & Triplets ---")

    # Create features and make triplets for all datasets
    for ds in train_datasets:
        train_triplets = CalculateFeatures.build_training_triplets(
            train_retrieval_data,
            dataset_name=ds,
            num_negatives=args.num_negatives
        )

        if len(train_triplets) > max_samples_per_dataset:
            train_triplets = random.sample(train_triplets, max_samples_per_dataset)

        global_triplets.extend(train_triplets)
        print(f"[{ds.upper()}] Added {len(train_triplets)} triplets to global pool.")

    print(f"\nTotal training triplets: {len(global_triplets)}")
    if not global_triplets:
        raise RuntimeError("No training triplets were generated. Check cached data and dataset configuration.")

    dataloader = DataLoader(
        HybridLTRDataset(global_triplets),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    models_dir = ensure_dir(args.models_dir)
    results_dir = ensure_dir(args.results_dir)
    training_summary = {
        "created_at_utc": timestamp_utc(),
        "seed": args.seed,
        "device": device,
        "cache_path": cache_path,
        "train_datasets": train_datasets,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "max_samples_per_dataset": max_samples_per_dataset,
        "num_negatives": args.num_negatives,
        "total_training_triplets": len(global_triplets),
        "models": {},
    }

    # Advanced AlphaRouter
    print("\n--- Training Advanced AlphaRouter ---")
    model_adv = AlphaRouter(input_dim=17)
    model_adv, adv_summary = train_router(model_adv, dataloader, epochs=args.epochs, lr=args.lr, device=device)

    adv_path = models_dir / "alpharouter_advanced_v1.pth"
    torch.save(model_adv.state_dict(), adv_path)
    print(f"Saved Advanced AlphaRouter weights to {adv_path}")
    training_summary["models"]["alpharouter_advanced"] = {
        "weights_path": str(adv_path),
        **adv_summary,
    }

    # MLP AlphaRouter
    print("\n--- Training AlphaRouter MLP ---")
    model_mlp = AlphaRouterMLP(input_dim=17)
    model_mlp, mlp_summary = train_router(model_mlp, dataloader, epochs=args.epochs, lr=args.lr, device=device)

    mlp_path = models_dir / "alpharouter_mlp_v1.pth"
    torch.save(model_mlp.state_dict(), mlp_path)
    print(f"Saved AlphaRouterMLP weights to {mlp_path}")
    training_summary["models"]["alpharouter_mlp"] = {
        "weights_path": str(mlp_path),
        **mlp_summary,
    }

    summary_path = save_json(training_summary, results_dir / "training_summary.json")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
