import faiss
import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class EmbeddingRetriever:
    def __init__(self, df, model_name='emilyalsentzer/Bio_ClinicalBERT',
                 emb_path="embeddings_choices.npy", index_path="index_choices.faiss"):
        self.df = df.copy()
        self.df["text"] = self.df["question"]

        self.model = SentenceTransformer(model_name)
        self.emb_path = emb_path
        self.index_path = index_path

        if os.path.exists(self.emb_path) and os.path.exists(self.index_path):
            print("Loading saved embeddings and FAISS index...")
            self.embeddings = np.load(self.emb_path)
            self.index = faiss.read_index(self.index_path)
        else:
            print("Building embeddings and FAISS index...")
            self.embeddings = self.model.encode(self.df["question"].tolist(), convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(self.embeddings)

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

            np.save(self.emb_path, self.embeddings)
            faiss.write_index(self.index, self.index_path)

    def search(self, query, topk=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, topk)

        results = []
        for idx, score in zip(I[0], D[0]):
            row = self.df.iloc[idx]
            results.append({
                "question": row['question'],
                "answer": row['answer']
            })
        return results


class BM25Retriever:
    def __init__(self, df, bm25_path="bm25_choices.json"):
        self.df = df.copy()
        self.df["text"] = self.df["question"]

        self.bm25_path = bm25_path
        tokenized_corpus = [doc.lower().split() for doc in self.df["question"].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        with open(self.bm25_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_corpus, f)

    def search(self, query, topk=1):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:topk]

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            results.append({
                "question": row['question'],
                "answer": row['answer']
            })
        return results


class HybridRetriever:
    def __init__(self, df, alpha=0.5, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        self.df = df.copy()
        self.alpha = alpha
        self.embedder = EmbeddingRetriever(self.df, model_name=model_name)
        self.bm25 = BM25Retriever(self.df)

    def search(self, query, topk=5):
        q_emb = self.embedder.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.embedder.index.search(q_emb, topk*5)  # 先取更多候选

        emb_candidates = I[0]
        emb_scores = D[0]

        bm25_scores = self.bm25.bm25.get_scores(query.lower().split())

        emb_scores_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-8)
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        results = []
        seen_idx = set()
        for idx, emb_score in zip(emb_candidates, emb_scores_norm):
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            score = self.alpha * emb_score + (1 - self.alpha) * bm25_scores_norm[idx]
            row = self.df.iloc[idx]
            results.append({
                "question": row['question'],
                "answer": row['answer'],
                "score": score
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]
        return results


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, topk=5):
        pairs = [(query, c["question"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:topk]

        retrieved_info = []
        for r in reranked:
            info = ''
            info += f'{r["question"]} '
            info += f'Answer: {r["answer"]}\n'
            retrieved_info.append(info)

        return retrieved_info


if __name__ == "__main__":
    # 读取数据
    df = pd.DataFrame(json.load(open("create_datasets/medqa_10000.json", "r", encoding="utf-8")))

    # Step 1: Hybrid Retrieval
    retriever = HybridRetriever(df, alpha=0.5)
    candidates = retriever.search("chest pain troponin elevated suspected MI", topk=20)

    # print("Hybrid retrieval candidates:")
    # for i, c in enumerate(candidates):
    #     print(f"{i+1}. Question: {c['question']}, Answer: {c['score']:.4f}")

    # Step 2: CrossEncoder Reranking
    reranker = CrossEncoderReranker()
    topk_results = reranker.rerank("chest pain troponin elevated suspected MI", candidates, topk=1)

    print("\nTop reranked results:")
    for i, info in enumerate(topk_results):
        print(f"=== Result {i+1} ===")
        print(info)
