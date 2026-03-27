import argparse
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dir, save_json, set_global_seed, timestamp_utc


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BEIR retrieval caches for AlphaRouter.")
    parser.add_argument("--train-datasets", nargs="+", default=["scifact", "fiqa", "nfcorpus", "msmarco"])
    parser.add_argument("--test-datasets", nargs="+", default=["scifact", "fiqa", "nfcorpus", "scidocs", "nq", "quora", "msmarco"])
    parser.add_argument("--train-sample-queries", type=int, default=7000)
    parser.add_argument("--test-sample-queries", type=int, default=1000)
    parser.add_argument("--cache-dir", default="cached_data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.data_loader import RunDatasets

    set_global_seed(args.seed)

    runner = RunDatasets()
    cache_dir = ensure_dir(args.cache_dir)

    # 1. Train split
    train_datasets = args.train_datasets
    runner.run_train_datasets(train_datasets, sample_queries=args.train_sample_queries)
    train_cache_path = cache_dir / "train_retrieval_data_cached_v1.pkl"
    with open(train_cache_path, "wb") as f:
        pickle.dump(runner.train_retrieval_data, f)

    # 2. Test split
    test_datasets = args.test_datasets
    runner.run_test_datasets(test_datasets, sample_queries=args.test_sample_queries)
    test_cache_path = cache_dir / "test_retrieval_data_cached_v1.pkl"
    with open(test_cache_path, "wb") as f:
        pickle.dump(runner.test_retrieval_data, f)

    save_json(
        {
            "created_at_utc": timestamp_utc(),
            "seed": args.seed,
            "train_datasets": train_datasets,
            "test_datasets": test_datasets,
            "train_sample_queries": args.train_sample_queries,
            "test_sample_queries": args.test_sample_queries,
            "train_cache_path": str(train_cache_path),
            "test_cache_path": str(test_cache_path),
        },
        cache_dir / "data_preparation_manifest.json",
    )

    print(f"Saved training cache to {train_cache_path}")
    print(f"Saved test cache to {test_cache_path}")


if __name__ == "__main__":
    main()
