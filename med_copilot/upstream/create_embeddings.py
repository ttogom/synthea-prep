import faiss
import numpy as np
import pandas as pd
import json
import os
import spacy

from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# Threshold for selecting dominant medical terms
CONTRIBUTION_THRESHOLD = 0.85

# Load spaCy / scispaCy model
def load_spacy_model():
    MODEL_NAME = "en_core_sci_md"
    try:
        nlp = spacy.load(MODEL_NAME)
        print(f"scispaCy model loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"Model not found: {MODEL_NAME}")
        print("Please install it first with:")
        print("pip install scispacy")
        print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz")
        raise e
    return nlp

NLP = load_spacy_model()


# LLM-based medical keyword extractor
class LLMKeywordExtractor:
    """
    Use an LLM to extract medically meaningful entities from free text.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def extract(self, text: str) -> list[str]:
        """
        Extract symptoms, diagnoses, and medications from text.
        Returns a de-duplicated list of terms in lower case.
        """

        prompt = f"""
You are a clinical NLP expert.
From the following clinical note, extract medically meaningful entities.

IMPORTANT:
- Each item MUST be a verbatim span copied from the text.
- Do NOT paraphrase or infer.
Return ONLY valid JSON:
{{
  "symptoms": [],
  "diagnoses": [],
  "medications": []
}}

Text:
{text}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        keywords = (
            result.get("symptoms", []) +
            result.get("diagnoses", []) +
            result.get("medications", [])
        )

        return list(set([k.strip() for k in keywords]))


# Embedding + FAISS index
class EmbeddingRetriever:
    """
    Build and maintain an embedding index over clinical notes (SOA).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        emb_path: str = "embeddings.npy",
        index_path: str = "index.faiss"
    ):
        print("Initializing EmbeddingRetriever...")

        self.df = df.copy()
        self.df["SOA"] = (
            self.df["subjective"] + " " +
            self.df["objective"] + " " +
            self.df["assessment"]
        )

        self.model = SentenceTransformer(model_name)
        self.emb_path = emb_path
        self.index_path = index_path

        if os.path.exists(self.emb_path) and os.path.exists(self.index_path):
            print("Loading saved embeddings and FAISS index...")
            self.embeddings = np.load(self.emb_path)
            self.index = faiss.read_index(self.index_path)
        else:
            print("Building new embeddings and FAISS index...")

            self.embeddings = self.model.encode(
                self.df["SOA"].tolist(),
                convert_to_numpy=True,
                show_progress_bar=True
            )

            faiss.normalize_L2(self.embeddings)

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

            np.save(self.emb_path, self.embeddings)
            faiss.write_index(self.index, self.index_path)

            print("Embeddings and index saved.")

# Keyword-aware hybrid retriever
class HybridRetriever:
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        alpha: float = 0.7,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str | None = None
    ):
        print("Initializing Keyword-Aware HybridRetriever...")

        self.df = df.copy()
        self.alpha = alpha

        self.df["SOA"] = (
            self.df["subjective"] + " " +
            self.df["objective"] + " " +
            self.df["assessment"]
        )

        self.embedder = EmbeddingRetriever(self.df, model_name=model_name)

        self.keyword_extractor = LLMKeywordExtractor(
            model=llm_model,
            api_key=llm_api_key or os.getenv("OPENAI_API_KEY")
        )

        print(f"HybridRetriever ready (alpha={self.alpha}).")

    # -------------------------
    # Build query embedding
    # -------------------------
    def build_query_embedding(self, keywords: list[str]) -> np.ndarray | None:
        if len(keywords) == 0:
            return None

        kw_embeddings = self.embedder.model.encode(
            keywords,
            convert_to_numpy=True
        )

        faiss.normalize_L2(kw_embeddings)
        return np.mean(kw_embeddings, axis=0).reshape(1, -1)

    # -------------------------
    # Phrase + span extraction
    # -------------------------
    def compute_case_phrase_contributions(
        self,
        query_embedding: np.ndarray,
        case_text: str,
        threshold: float,
        min_terms: int = 10,
        max_terms: int = 30,
        max_phrases: int = 200
    ):
        """
        Extract phrases WITH original span positions
        point to SAME display_text used in app.py.
        """

        doc = NLP(case_text)

        spans = []

        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 3:
                spans.append(chunk)

        for ent in doc.ents:
            if len(ent.text.strip()) > 3:
                spans.append(ent)

        # Deduplicate by (start, end)
        unique = {(s.start_char, s.end_char): s for s in spans}
        spans = list(unique.values())[:max_phrases]

        if len(spans) == 0:
            return [], []

        phrases = [s.text for s in spans]

        phrase_embeddings = self.embedder.model.encode(
            phrases,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        faiss.normalize_L2(phrase_embeddings)
        faiss.normalize_L2(query_embedding)

        scores = np.dot(phrase_embeddings, query_embedding.T).reshape(-1)

        all_contrib = []

        for span, score in zip(spans, scores):
            all_contrib.append({
                "term": span.text,
                "original_text": case_text[span.start_char:span.end_char],
                "start": span.start_char,
                "end": span.end_char,
                "contribution": float(score)
            })

        all_contrib = sorted(all_contrib, key=lambda x: x["contribution"], reverse=True)

        dominant_terms = [x for x in all_contrib if x["contribution"] >= threshold]

        # Ensure min / max bounds
        if len(dominant_terms) < min_terms:
            dominant_terms = all_contrib[:min_terms]

        if len(dominant_terms) > max_terms:
            dominant_terms = dominant_terms[:max_terms]

        return all_contrib, dominant_terms

    # -------------------------
    # Keyword overlap
    # -------------------------
    def compute_keyword_overlap_score(self, keywords: list[str], case_text: str) -> float:
        if not keywords:
            return 0.0

        case_text = case_text.lower()
        hits = sum(1 for kw in keywords if kw.lower().strip() in case_text)

        return hits / len(keywords)

    # -------------------------
    # SEARCH
    # -------------------------
    def search(self, query: str, topk: int = 5) -> list[dict]:

        print(f"\n[SEARCH] Query: {query}")

        keywords = self.keyword_extractor.extract(query)
        print("[SEARCH] Extracted keywords:", keywords)

        if not keywords:
            q_emb = self.embedder.model.encode([query], convert_to_numpy=True)
        else:
            q_emb = self.build_query_embedding(keywords)

        faiss.normalize_L2(q_emb)

        D, I = self.embedder.index.search(q_emb, topk * 8)
        candidates, scores = I[0], D[0]

        interim_results = []
        seen = set()

        for idx, emb_score in zip(candidates, scores):

            if idx in seen:
                continue
            seen.add(idx)

            row = self.df.iloc[idx]

            case_text = (
                row["subjective"] + " " +
                row["objective"] + " " +
                row["assessment"]
            )

            # ✅ THIS TEXT WILL BE USED FOR HIGHLIGHTING
            display_text = f"""Subjective:
{row["subjective"]}

Objective:
{row["objective"]}

Assessment:
{row["assessment"]}

Plan:
{row["plan"]}
"""

            keyword_score = self.compute_keyword_overlap_score(keywords, case_text)

            hybrid_score = (
                self.alpha * float(emb_score) +
                (1 - self.alpha) * float(keyword_score)
            )

            interim_results.append({
                "idx": idx,
                "display_text": display_text,
                "case_text": case_text,
                "subjective": row["subjective"],
                "objective": row["objective"],
                "assessment": row["assessment"],
                "plan": row["plan"],
                "embedding_score": float(emb_score),
                "keyword_score": float(keyword_score),
                "hybrid_score": float(hybrid_score)
            })

            print(f"[CANDIDATE] idx={idx} | emb={emb_score:.4f} | keyword={keyword_score:.3f} | hybrid={hybrid_score:.4f}")

        topk_results = sorted(
            interim_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:topk]

        final_results = []

        for r in topk_results:

            # 🔴 IMPORTANT: must use same text used for display
            display_text = r["display_text"]

            all_contrib, dominant_terms = self.compute_case_phrase_contributions(
                query_embedding=q_emb,
                case_text=display_text,   # <---- KEY FIX
                threshold=CONTRIBUTION_THRESHOLD
            )

            r["term_contributions"] = all_contrib
            r["dominant_terms"] = dominant_terms

            print("\n[DEBUG] --- dominant_terms preview ---")
            for t in dominant_terms[:5]:
                print(t)

            print("\n[DEBUG] --- display_text preview ---")
            print(display_text[:300])

            final_results.append(r)

        return final_results

# Cross-Encoder reranker
class CrossEncoderReranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], topk: int = 3) -> list[str]:

        pairs = [
            (
                query,
                c["subjective"] + " " +
                c["objective"] + " " +
                c["assessment"]
            )
            for c in candidates
        ]

        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:topk]

        retrieved_info = []

        for i, r in enumerate(reranked, 1):

            info = f"""
Result {i}
Subjective:
{r['subjective']}
Objective:
{r['objective']}
Assessment:
{r['assessment']}
Plan:
{r['plan']}
Rerank score: {r['rerank_score']:.4f}

Dominant medical concepts (contribution >= {CONTRIBUTION_THRESHOLD}):
"""

            for t in r["dominant_terms"]:
                info += f" - {t['original_text']} ({t['contribution']:.4f})\n"

            retrieved_info.append(info)

        return retrieved_info


# -------------------------------
# Standalone test
# -------------------------------

if __name__ == "__main__":

    df = pd.DataFrame(
        json.load(open("soap_with_metadata.json", "r", encoding="utf-8"))
    )

    query = "chest pain troponin elevated suspected MI"

    retriever = HybridRetriever(df, alpha=0.7)
    candidates = retriever.search(query, topk=5)

    reranker = CrossEncoderReranker()
    final = reranker.rerank(query, candidates, topk=3)

    for r in final:
        print(r)
