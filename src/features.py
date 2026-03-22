import numpy as np
import random
import string
import math
import spacy

from scipy.stats import skew


class CalculateFeatures:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    @staticmethod
    def get_lexical_features(idf_dict: dict, vocab_set: set, tokens: list, q_len: int) -> list:
        """Calculates standard lexical, IDF-based, and statistical features."""
        idfs = [idf_dict.get(t, 0.0) for t in tokens]

        mean_idf = float(np.mean(idfs)) if q_len > 0 else 0.0
        max_idf = float(np.max(idfs)) if q_len > 0 else 0.0
        min_idf = float(np.min(idfs)) if q_len > 0 else 0.0
        std_idf = float(np.std(idfs)) if q_len > 0 else 0.0

        # Rare token ratio (a threshold > 5.0)
        rare_token_ratio = sum(1 for idf in idfs if idf > 5.0) / q_len if q_len > 0 else 0.0

        # OOV ratio (Out Of Vocabulary)
        oov_ratio = sum(1 for t in tokens if t not in vocab_set) / q_len if q_len > 0 else 0.0

        # Identifies if there are a few highly specific words among common ones (favors BM25)
        if q_len > 2 and std_idf > 0:
            idf_skewness = float(skew(idfs))
        else:
            idf_skewness = 0.0

        # Calculates Shannon entropy over IDF weights to measure lexical diversity
        sum_idf = sum(idfs)
        if sum_idf > 0:
            probabilities = [idf / sum_idf for idf in idfs]
            query_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        else:
            query_entropy = 0.0

        return [q_len, mean_idf, max_idf, min_idf, std_idf, rare_token_ratio, oov_ratio, idf_skewness, query_entropy]

    @staticmethod
    def get_digit_features(chars: list, query_text: str) -> list:
        """Calculates character-level features (digits, casing, punctuation)."""
        char_len = len(chars)

        # Guard clause for empty character lists
        if char_len == 0:
            return [0.0, 0.0, 0.0]

        digit_ratio = sum(1 for c in chars if c.isdigit()) / char_len

        # Uppercase ratio
        orig_chars = list(query_text.replace(" ", ""))
        orig_len = len(orig_chars)
        uppercase_ratio = sum(1 for c in orig_chars if c.isupper()) / orig_len if orig_len > 0 else 0.0

        punctuation_ratio = sum(1 for c in chars if c in string.punctuation) / char_len

        return [digit_ratio, uppercase_ratio, punctuation_ratio]

    @staticmethod
    def get_nlp_features(query_text: str, tokens: list) -> list:
        """Calculates NLP-based features using high-performance SpaCy."""
        # Process text through the optimized SpaCy pipeline
        doc = CalculateFeatures.nlp(query_text)
        n_tokens = max(len(doc), 1)

        # SpaCy's built-in stopword detection
        stopword_ratio = sum(1 for token in doc if token.is_stop) / n_tokens

        # Universal POS tags. Including PROPN (Proper Nouns) as they are strong signals for exact match
        noun_ratio = sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN")) / n_tokens
        verb_ratio = sum(1 for token in doc if token.pos_ == "VERB") / n_tokens
        adj_ratio = sum(1 for token in doc if token.pos_ == "ADJ") / n_tokens

        # Average word length
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0

        return [stopword_ratio, noun_ratio, verb_ratio, adj_ratio, avg_word_len]

    @staticmethod
    def extract_features(query_text: str, idf_dict: dict, vocab_set: set) -> list:
        """Orchestrates feature extraction from all categories."""
        tokens = query_text.lower().split()
        chars = list(query_text.replace(" ", ""))
        q_len = len(tokens)

        # Return 17 zeros if the query is empty to prevent downstream errors
        if q_len == 0:
            return [0.0] * 17

        lex_features = CalculateFeatures.get_lexical_features(idf_dict, vocab_set, tokens, q_len)
        digit_features = CalculateFeatures.get_digit_features(chars, query_text)
        nlp_features = CalculateFeatures.get_nlp_features(query_text, tokens)

        # Total vector: 9 (Lexical) + 3 (Character) + 5 (NLP) = 17 features
        return lex_features + digit_features + nlp_features

    @staticmethod
    def build_training_triplets(retrieval_data: dict, dataset_name: str, num_negatives=3):
        """
        Collects triplets (x_q, pos_scores, neg_scores) for training a PyTorch model.
        num_negatives: the number of irrelevant documents to include for every relevant document.
        """
        data = retrieval_data[dataset_name]
        queries = data["queries"]
        qrels = data["qrels"]
        dense_norm = data["dense_norm"]
        sparse_norm = data["sparse_norm"]
        idf_dict = data["idf_dict"]
        vocab_set = data["vocab_set"]

        triplets = []

        for qid, q_text in queries.items():
            if qid not in qrels:
                continue

            # 1. Get features of queries x_q
            x_q = CalculateFeatures.extract_features(q_text, idf_dict, vocab_set)

            # 2. Find positive documents
            pos_docs = [did for did, rel in qrels[qid].items() if rel > 0]
            if not pos_docs:
                continue

            # 3. Get pull of negative documents
            retrieved_docs = set(dense_norm.get(qid, {}).keys()) | set(sparse_norm.get(qid, {}).keys())
            neg_docs = list(retrieved_docs - set(pos_docs))

            # Guard clause: ensure we have exactly `num_negatives` to keep tensor shapes consistent
            if len(neg_docs) < num_negatives:
                print(f"[{dataset_name.upper()}] Warning: Insufficient negative docs for {qid}. Skipping.")
                continue

            # 4. get lists
            for pos_d in pos_docs:
                # get random negatives examples (Hard Negatives, they are from the top)
                sampled_negs = random.sample(neg_docs, min(num_negatives, len(neg_docs)))

                neg_dense_scores = [float(dense_norm.get(qid, {}).get(nd, 0.0)) for nd in sampled_negs]
                neg_sparse_scores = [float(sparse_norm.get(qid, {}).get(nd, 0.0)) for nd in sampled_negs]

                triplets.append({
                    'features': x_q,
                    'pos_dense': float(dense_norm.get(qid, {}).get(pos_d, 0.0)),
                    'pos_sparse': float(sparse_norm.get(qid, {}).get(pos_d, 0.0)),
                    'neg_dense': neg_dense_scores,   # List of K floats
                    'neg_sparse': neg_sparse_scores  # List of K floats
                })

        print(f"[{dataset_name.upper()}] Generated {len(triplets)} training triplets.")
        return triplets
