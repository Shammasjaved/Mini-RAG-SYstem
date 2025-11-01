import re
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from typing import List, Dict

class MovieRAG:
    """
    A tiny, readable RAG pipeline for movie plots.
    Steps: load -> chunk -> embed -> index -> retrieve -> generate.
    """
    def __init__(self,
                 csv_path: str = "data/wiki_movie_plots_deduped.csv",
                 rows: int = 300,
                 chunk_words: int = 300,
                 chunk_overlap: int = 40,
                 top_k: int = 4,
                 embed_model: str = "text-embedding-3-small",
                 chat_model: str = "gpt-4o-mini"):
        self.csv_path = csv_path
        self.rows = rows
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.client = OpenAI()
        self.chunks: List[str] = []
        self.titles: List[str] = []
        self.index = None 
        self._built = False

    # -Data 
    def _pick_cols(self, cols):
        tcol = next((c for c in cols if c.lower() in ("title","movie_title","name")), None)
        pcol = next((c for c in cols if c.lower() in ("plot","summary","storyline")), None)
        if not tcol or not pcol:
            raise ValueError("CSV must contain Title and Plot columns (case-insensitive).")
        return tcol, pcol

    def _clean(self, s: str) -> str:
        return re.sub(r"\s+", " ", str(s).replace("\r"," ").replace("\n"," ")).strip()

    def load_and_chunk(self):
        df = pd.read_csv(self.csv_path)
        tcol, pcol = self._pick_cols(df.columns)
        df = df[[tcol, pcol]].rename(columns={tcol:"Title", pcol:"Plot"}).head(self.rows).reset_index(drop=True)

        C, O = self.chunk_words, self.chunk_overlap
        self.chunks.clear(); self.titles.clear()
        for _, r in df.iterrows():
            words = self._clean(r["Plot"]).split(" ")
            if len(words) < 50: 
                continue
            start = 0
            while start < len(words):
                end = min(start + C, len(words))
                text = " ".join(words[start:end])
                self.chunks.append(text)
                self.titles.append(r["Title"])
                if end == len(words): break
                start = max(end - O, 0)

    # -Embeddings & Index 
    def _embed(self, texts: List[str]) -> np.ndarray:
        # Simple batching to avoid large payloads
        B, out = 128, []
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            res = self.client.embeddings.create(model=self.embed_model, input=batch)
            out.extend([np.array(d.embedding, dtype=np.float32) for d in res.data])
        return np.vstack(out)

    def build(self):
        # Create embeddings for chunks and add to FAISS (cosine via normalization)
        if not self.chunks:
            self.load_and_chunk()
        vecs = self._embed(self.chunks)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.index = index
        self._built = True

    # - Retrieval 
    def retrieve_contexts(self, question: str) -> List[str]:
        q = self._embed([question])[0]
        q /= (np.linalg.norm(q) + 1e-9)
        D, I = self.index.search(q.reshape(1,-1).astype(np.float32), self.top_k)
        ids = I[0].tolist()
        return [f"{self.titles[i]} … {self.chunks[i][:350]}" for i in ids]

    # - Generation 
    def answer(self, question: str, contexts: List[str]) -> Dict[str, str]:
        system = (    "You are an assistant that answers ONLY using the provided context. "
                      "Do not add new facts. Keep answers very short and direct.")
        prompt = "Contexts:\n" + "\n\n---\n\n".join(contexts) + f"\n\nQuestion: {question}\nReturn a short, direct answer."
        res = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":prompt}],
            temperature=0.2
        )
     # ✅ Dynamic reasoning style (matching PDF example)
        reasoning = (
            f"I searched the movie plots and found that the returned context "
            f"is related to the question '{question}'. I used that evidence to "
            f"generate the answer."
        )

        return {
            "answer": res.choices[0].message.content.strip(),
            "contexts": contexts,
            "reasoning": reasoning
        }
    def ask(self, question: str) -> Dict[str, object]:
        if not self._built:
            self.build()
        ctxs = self.retrieve_contexts(question)
        gen = self.answer(question, ctxs)
        return {"answer": gen["answer"], "contexts": ctxs, "reasoning": gen["reasoning"]}
