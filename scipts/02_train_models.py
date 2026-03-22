import sys
import os
import random
import pickle
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models import AlphaRouter, HybridLTRDataset, AlphaRouterMLP
from src.loss import infonce_hybrid_loss
from src.features import CalculateFeatures


def train_router(model, dataloader, epochs=32, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    early_stopping_threshold = 5
    early_stopping_count = 0
    best_loss = float('inf')
    min_delta = 1e-4

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

        if early_stopping_count >= early_stopping_threshold:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device.upper()}")

    # LOAD CACHED DATA
    cache_path = os.path.join(PROJECT_ROOT, "cached_data", "train_retrieval_data_cached_v1.pkl")
    print(f"Loading data from {cache_path}...")

    with open(cache_path, "rb") as f:
        train_retrieval_data = pickle.load(f)

    train_datasets = ["scifact", "fiqa", "nfcorpus", "msmarco"]
    global_triplets = []
    MAX_SAMPLES_PER_DATASET = 15000

    print("\n--- Generating Features & Triplets ---")

    # Create features and make triplets for all datasets
    for ds in train_datasets:
        train_triplets = CalculateFeatures.build_training_triplets(
            train_retrieval_data,
            dataset_name=ds,
            num_negatives=5
        )

        if len(train_triplets) > MAX_SAMPLES_PER_DATASET:
            train_triplets = random.sample(train_triplets, MAX_SAMPLES_PER_DATASET)

        global_triplets.extend(train_triplets)
        print(f"[{ds.upper()}] Added {len(train_triplets)} triplets to global pool.")

    print(f"\nTotal training triplets: {len(global_triplets)}")

    dataloader = DataLoader(HybridLTRDataset(global_triplets), batch_size=64, shuffle=True, drop_last=True)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Advanced AlphaRouter
    print("\n--- Training Advanced AlphaRouter ---")
    model_adv = AlphaRouter(input_dim=17)
    model_adv = train_router(model_adv, dataloader, epochs=62, device=device)

    adv_path = os.path.join(models_dir, "alpharouter_advanced_v1.pth")
    torch.save(model_adv.state_dict(), adv_path)
    print(f"Saved Advanced AlphaRouter weights to {adv_path}")

    # MLP AlphaRouter
    print("\n--- Training AlphaRouter MLP ---")
    model_mlp = AlphaRouterMLP(input_dim=17)
    model_mlp = train_router(model_mlp, dataloader, epochs=62, device=device)

    mlp_path = os.path.join(models_dir, "alpharouter_mlp_v1.pth")
    torch.save(model_mlp.state_dict(), mlp_path)
    print(f"Saved AlphaRouterMLP weights to {mlp_path}")


if __name__ == "__main__":
    main()
