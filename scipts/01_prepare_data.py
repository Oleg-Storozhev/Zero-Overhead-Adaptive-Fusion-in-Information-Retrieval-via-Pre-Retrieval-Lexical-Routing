import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import RunDatasets


def main():
    runner = RunDatasets()

    # 1. Train split
    train_datasets = ["scifact", "fiqa", "nfcorpus", "msmarco"]
    runner.run_train_datasets(train_datasets)
    with open("train_retrieval_data_cached_v1.pkl", "wb") as f:
        pickle.dump(runner.train_retrieval_data, f)

    # 2. Test split
    test_datasets = ["scifact", "fiqa", "nfcorpus", "scidocs", "nq", "quora", "msmarco"]
    runner.run_test_datasets(test_datasets)
    with open("test_retrieval_data_cached_v1.pkl", "wb") as f:
        pickle.dump(runner.test_retrieval_data, f)


if __name__ == "__main__":
    main()
