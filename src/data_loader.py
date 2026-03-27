import os
import bm25s
import numpy as np
import random
import math

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SentenceBERT


class RunDatasets:
    def __init__(self):
        self.evaluator = EvaluateRetrieval()
        self.train_retrieval_data = {}
        self.test_retrieval_data = {}

    @staticmethod
    def add_data(dataset_name: str):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(os.getcwd(), "../datasets")
        data_path = util.download_and_unzip(url, out_dir)
        return data_path

    @staticmethod
    def add_model_dense(model_name: str):
        dense_model = DRES(SentenceBERT(f"sentence-transformers/{model_name}"), batch_size=64)
        return dense_model

    @staticmethod
    def add_model_sparse():
        retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        return retriever

    @staticmethod
    def normalize_scores(results):
        """Min-Max normalization of scores for BM25 and Dense."""
        normalized = {}
        for qid, docs in results.items():
            if not docs:
                continue
            scores = list(docs.values())
            min_s, max_s = min(scores), max(scores)

            normalized[qid] = {}
            for did, score in docs.items():
                if max_s > min_s:
                    normalized[qid][did] = (score - min_s) / (max_s - min_s)
                else:
                    normalized[qid][did] = 0.5
        return normalized

    @staticmethod
    def get_hybrid_scores(dense_res, sparse_res, alpha):
        """alpha * Dense + (1 - alpha) * Sparse."""
        hybrid = {}
        all_qids = set(dense_res.keys()) | set(sparse_res.keys())

        for qid in all_qids:
            hybrid[qid] = {}
            q_dense = dense_res.get(qid, {})
            q_sparse = sparse_res.get(qid, {})
            all_dids = set(q_dense.keys()) | set(q_sparse.keys())

            for did in all_dids:
                s_d = q_dense.get(did, 0.0)
                s_s = q_sparse.get(did, 0.0)
                hybrid[qid][did] = float(alpha * s_d + (1.0 - alpha) * s_s)

        return hybrid

    def _process_dataset(self, ds: str, split: str, sample_queries: int = None):
        """Core pipeline: loads data, runs retrieval, normalizes scores."""
        print(f"\n[{ds.upper()}] Loading data (split='{split}')...")
        data_path = self.add_data(ds)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

        corpus_ids = list(corpus.keys())
        corpus_texts = [f"{corpus[cid].get('title', '')} {corpus[cid].get('text', '')}" for cid in corpus_ids]
        query_ids = [qid for qid in queries.keys() if qid in qrels]
        if sample_queries and len(query_ids) > sample_queries:
            print(f"[{ds.upper()}] Subsampling queries from {len(query_ids)} to {sample_queries}...")
            query_ids = random.sample(query_ids, sample_queries)

        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

        query_texts = [queries[qid] for qid in query_ids]

        print(f"[{ds.upper()}] Building IDF dictionary...")
        df_dict = {}
        for text in corpus_texts:
            for token in set(text.lower().split()):
                df_dict[token] = df_dict.get(token, 0) + 1

        N = len(corpus_texts)
        idf_dict = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in df_dict.items() if df > 0}
        vocab_set = set(df_dict.keys())

        print(f"[{ds.upper()}] Running Dense Retrieval...")
        dense_model = self.add_model_dense("all-MiniLM-L6-v2")
        dense_retriever = EvaluateRetrieval(dense_model, score_function="cos_sim")
        dense_raw = dense_retriever.retrieve(corpus, queries)

        print(f"[{ds.upper()}] Running Sparse Retrieval (BM25s)...")
        sparse_model = self.add_model_sparse()
        corpus_tokens = bm25s.tokenize(corpus_texts)
        sparse_model.index(corpus_tokens)

        query_tokens = bm25s.tokenize(query_texts)
        docs, scores = sparse_model.retrieve(query_tokens, corpus=corpus_ids, k=100)
        sparse_raw = {qid: {docs[i][j]: float(scores[i][j]) for j in range(len(docs[i]))} for i, qid in
                      enumerate(query_ids)}

        print(f"[{ds.upper()}] Normalizing scores...")
        dense_norm = self.normalize_scores(dense_raw)
        sparse_norm = self.normalize_scores(sparse_raw)

        return {
            "queries": {qid: queries[qid] for qid in query_ids},
            "qrels": qrels,
            "dense_norm": dense_norm,
            "sparse_norm": sparse_norm,
            "idf_dict": idf_dict,
            "vocab_set": vocab_set
        }

    def run_train_datasets(self, datasets: list, split="train", sample_queries: int = 7000):
        """Extracts and saves retrieval artifacts for training the PyTorch model."""
        for ds in datasets:
            self.train_retrieval_data[ds] = self._process_dataset(ds, split, sample_queries=sample_queries)
        print("\nTraining datasets processed and saved!")

    def run_test_datasets_with_alphas(self, datasets: list):
        alphas = np.round(np.arange(0.0, 1.05, 0.05), 2)
        results_ndcg = {ds: [] for ds in datasets}
        results_mrr = {ds: [] for ds in datasets}

        for ds in datasets:
            data = self.test_retrieval_data[ds]

            dense_norm = data["dense_norm"]
            sparse_norm = data["sparse_norm"]
            qrels = data["qrels"]

            print(f"[{ds.upper()}] Evaluating static alphas...")
            for alpha in alphas:
                hybrid_res = self.get_hybrid_scores(dense_norm, sparse_norm, alpha)

                ndcg_dict, _, _, _ = self.evaluator.evaluate(qrels, hybrid_res, [10])
                mrr_dict = self.evaluator.evaluate_custom(qrels, hybrid_res, [10], metric="mrr")

                results_ndcg[ds].append(ndcg_dict["NDCG@10"])
                mrr_key = list(mrr_dict.keys())[0]
                results_mrr[ds].append(mrr_dict[mrr_key])

        print("\nTest benchmark completed!")
        return results_ndcg, results_mrr, alphas

    def run_test_datasets(self, datasets: list, split="test", sample_queries: int = 1000):
        """Extracts artifacts, saves them for dynamic inference, and calculates static baselines."""
        alphas = np.round(np.arange(0.0, 1.05, 0.05), 2)
        results_ndcg = {ds: [] for ds in datasets}
        results_mrr = {ds: [] for ds in datasets}

        for ds in datasets:
            data = self._process_dataset(ds, split, sample_queries=sample_queries)
            self.test_retrieval_data[ds] = data  # Save for PyTorch

            dense_norm = data["dense_norm"]
            sparse_norm = data["sparse_norm"]
            qrels = data["qrels"]

            print(f"[{ds.upper()}] Evaluating static alphas...")
            for alpha in alphas:
                hybrid_res = self.get_hybrid_scores(dense_norm, sparse_norm, alpha)

                ndcg_dict, _, _, _ = self.evaluator.evaluate(qrels, hybrid_res, [10])
                mrr_dict = self.evaluator.evaluate_custom(qrels, hybrid_res, [10], metric="mrr")

                results_ndcg[ds].append(ndcg_dict["NDCG@10"])
                mrr_key = list(mrr_dict.keys())[0]
                results_mrr[ds].append(mrr_dict[mrr_key])

        print("\nTest benchmark completed!")
        return results_ndcg, results_mrr, alphas
