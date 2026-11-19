import os
import json
import uuid
import re
from typing import List, Dict, Any
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests
import glob

# -----------------------------
# CONFIG
# -----------------------------
class Config:
    EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 100

    FAISS_INDEX_PATH = "faiss_index.ivf"
    METADATA_PATH = "faiss_metadata.json"
    TOP_K = 5

    MAX_GEN_TOKENS = 256

    # OpenRouter Gemini
    OPENROUTER_API_KEY = "enter your key here"
    GEMINI_MODEL = "google/gemini-2.5-flash"

cfg = Config()

# -----------------------------
# PDF extraction
# -----------------------------
def clean_text(text: str) -> str:
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        text = clean_text(text)
        if not text:
            print(f"⚠️ Warning: Empty text on page {i+1} of {pdf_path}. Skipping is avoided, adding empty string.")
            text = ""  # keep the page to preserve indexing
        pages.append({"page": i + 1, "text": text, "source_file": pdf_path})
    doc.close()
    return pages

# -----------------------------
# Chunking
# -----------------------------
def sentence_split_amharic(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[።\n])\s+", text) if s.strip()]

def chunk_text(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for p in pages:
        sents = sentence_split_amharic(p["text"])
        if not sents:
            # ensure at least one chunk per page
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": "",
                "source": os.path.basename(p["source_file"]),
                "page": p["page"]
            })
            continue

        cur = 0
        while cur < len(sents):
            chunk_sents, token_count, j = [], 0, cur
            while j < len(sents) and token_count < cfg.CHUNK_SIZE:
                token_count += len(sents[j].split())
                chunk_sents.append(sents[j])
                j += 1
            if chunk_sents:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": " ".join(chunk_sents),
                    "source": os.path.basename(p["source_file"]),
                    "page": p["page"]
                })
            step = max(1, len(chunk_sents) - cfg.CHUNK_OVERLAP)
            cur += step
    return chunks

# -----------------------------
# Embeddings & FAISS
# -----------------------------
class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(cfg.EMBEDDINGS_MODEL, device=cfg.DEVICE)
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

class FaissStore:
    def __init__(self):
        self.index = None
        self.id_to_meta = {}

    def create_index(self, dim: int, nlist: int = 100):
        if nlist > 0:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dim)  # fallback for small datasets

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if embeddings.shape[0] == 0:
            return
        faiss.normalize_L2(embeddings)

        if self.index is None:
            nlist = min(100, embeddings.shape[0])
            self.create_index(embeddings.shape[1], nlist)
            if hasattr(self.index, "is_trained") and self.index.is_trained == 0:
                if embeddings.shape[0] >= nlist:
                    self.index.train(embeddings)
                else:
                    print("⚠️ Not enough embeddings to train IVF index, using flat index.")
                    self.index = faiss.IndexFlatIP(embeddings.shape[1])

        ids = np.arange(len(self.id_to_meta), len(self.id_to_meta) + embeddings.shape[0])
        self.index.add_with_ids(embeddings, ids)
        for i, meta in zip(ids, metadatas):
            self.id_to_meta[str(i)] = meta

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(query_emb.shape[0])]
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for dist_row, id_row in zip(D, I):
            row = []
            for dist, idx in zip(dist_row.tolist(), id_row.tolist()):
                if idx < 0:
                    continue
                row.append({"meta": self.id_to_meta[str(idx)], "score": float(dist)})
            results.append(row)
        return results

    def save(self):
        if self.index:
            faiss.write_index(self.index, cfg.FAISS_INDEX_PATH)
            with open(cfg.METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(cfg.FAISS_INDEX_PATH) and os.path.exists(cfg.METADATA_PATH):
            self.index = faiss.read_index(cfg.FAISS_INDEX_PATH)
            with open(cfg.METADATA_PATH, "r", encoding="utf-8") as f:
                self.id_to_meta = json.load(f)
            print("✅ Loaded existing FAISS index and metadata.")
            return True
        else:
            return False

# -----------------------------
# Retriever + Gemini Chat
# -----------------------------
class RetrieverRAG:
    def __init__(self, embedder: EmbeddingGenerator, store: FaissStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed_texts([query])
        return self.store.search(q_emb, cfg.TOP_K)[0]

    def construct_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        header = "እባክህ በታች ያሉትን መረጃዎች በመጠቀም ጥያቄውን አጭር በአማርኛ ይመልሱ።\n\n"
        ctx = "\n---\n".join([
            f"[Context {i+1} | source: {r['meta']['source']} | page: {r['meta']['page']} | score: {r['score']:.3f}]\n{r['meta']['text']}"
            for i, r in enumerate(retrieved)
        ])
        return f"{header}\nContext:\n{ctx}\nQuestion:\n{query}\nAnswer:"

    def generate_answer(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": cfg.GEMINI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers in Amharic."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": cfg.MAX_GEN_TOKENS,
            "temperature": 0.3
        }
        r = requests.post(url, headers=headers, json=data)
        if r.status_code == 200:
            resp = r.json()
            if "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["message"]["content"].strip()
            else:
                return json.dumps(resp, ensure_ascii=False)
        else:
            return f"Error: {r.status_code} -> {r.text}"

    def answer_query(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query)
        prompt = self.construct_prompt(query, retrieved)
        answer = self.generate_answer(prompt)
        return {"query": query, "answer": answer, "retrieved": retrieved, "prompt": prompt}

# -----------------------------
# END-TO-END
# -----------------------------
def example_end_to_end(pdf_folder: str, query: str):
    embedder = EmbeddingGenerator()
    store = FaissStore()

    if store.load():
        retriever = RetrieverRAG(embedder, store)
    else:
        print("⚙️ Building FAISS index from PDFs...")
        pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        if not pdf_paths:
            print("❌ No PDFs found!")
            return

        all_pages = []
        for pdf in pdf_paths:
            print(f"📄 Extracting {pdf}")
            all_pages.extend(extract_text_from_pdf(pdf))

        chunks = chunk_text(all_pages)
        if not chunks:
            print("❌ No text chunks generated!")
            return

        texts = [c['text'] for c in chunks]
        metadatas = [{"id": c['id'], "text": c['text'], "source": c['source'], "page": c['page']} for c in chunks]

        embeddings = embedder.embed_texts(texts)
        store.add(embeddings, metadatas)
        store.save()
        print("✅ FAISS index built and saved.")
        retriever = RetrieverRAG(embedder, store)

    res = retriever.answer_query(query)

    print("\n📝 Answer:\n", res['answer'])
    print("\n🔍 Retrieved contexts:\n", json.dumps(res['retrieved'], ensure_ascii=False, indent=2))

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    example_end_to_end("/home/elshaday/RAG/all_pdfs", "የወል የሥራ ክርክሮች ምንን ይመለከታሉ?")
