"""
==============================================================================
 Simple RAG (Retrieval-Augmented Generation) Pipeline with an Open-Source LLM
==============================================================================

WHAT IS RAG?
------------
A Large Language Model (LLM) only knows what it saw during training. It does not
know about your private documents, your company wiki, or anything that happened
after its training cutoff. If you ask it about those things it will either say
"I don't know" or, worse, make something up ("hallucinate").

RAG addresses this by giving the model the relevant information *at question time*:

    1. RETRIEVE  - find the pieces of your documents most relevant to the question
    2. AUGMENT   - paste those pieces into the prompt as "context"
    3. GENERATE  - let the LLM answer using that context

The model doesn't need to be retrained. You just change what goes into the prompt.


THE PIPELINE IN THIS FILE
-------------------------

   INDEXING (done once, offline)
   ┌───────────┐   ┌──────────┐   ┌────────────────┐   ┌───────────────┐
   │ Documents │──▶│ Chunking │──▶│ Embedding model│──▶│ Vector store  │
   │ (.txt/.md)│   │          │   │ (text → vector)│   │ (numpy matrix)│
   └───────────┘   └──────────┘   └────────────────┘   └───────────────┘

   QUERYING (done for every question)
   ┌──────────┐   ┌────────────────┐   ┌────────────┐   ┌────────┐   ┌────────┐
   │ Question │──▶│ Embedding model│──▶│ Top-k      │──▶│ Prompt │──▶│  LLM   │──▶ Answer
   └──────────┘   └────────────────┘   │ similarity │   │ builder│   │(Qwen2.5)│
                                       └────────────┘   └────────┘   └────────┘


COMPONENTS USED (all open source, all run locally)
--------------------------------------------------
  * Embedding model : sentence-transformers/all-MiniLM-L6-v2  (~90 MB, Apache-2.0)
                      Turns text into a 384-dimensional vector. Texts with similar
                      meaning get vectors that point in similar directions.
  * Vector store    : a plain NumPy array + cosine similarity. For a few thousand
                      chunks this is instant. For millions, swap in FAISS / Chroma /
                      Qdrant - the idea is identical.
  * LLM             : Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0) via Hugging Face
                      `transformers`. Small enough for a laptop CPU (slow-ish) or any
                      modest GPU. You can swap in any chat model, e.g.
                      "meta-llama/Llama-3.2-3B-Instruct", "microsoft/Phi-3.5-mini-instruct",
                      "mistralai/Mistral-7B-Instruct-v0.3" (needs a bigger GPU).


INSTALL (run yourself)
----------------------
    pip install -r requirements.txt

  For GPU acceleration install the CUDA build of PyTorch first, see
  https://pytorch.org/get-started/locally/


USAGE
-----
    # Put .txt or .md files into ./data  (a sample file is included)
    python simple_rag.py                                  # interactive Q&A loop
    python simple_rag.py --question "What is RAG?"        # single question
    python simple_rag.py --data_dir my_docs --top_k 5     # custom options

The first run downloads the models from Hugging Face (cached afterwards).
==============================================================================
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# All tunable knobs in one place. Changing these is the easiest way to
# experiment with how retrieval quality and answer quality behave.

@dataclass
class RAGConfig:
    # Folder containing the documents we want to "chat with".
    data_dir: str = "data"

    # Embedding model: converts text to vectors. all-MiniLM-L6-v2 is small, fast
    # and good enough for English. For better quality try "BAAI/bge-small-en-v1.5"
    # or for multilingual text "intfloat/multilingual-e5-small".
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Generator LLM: any Hugging Face chat/instruct model works.
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Chunking: documents are split into pieces of ~chunk_size characters, with
    # chunk_overlap characters shared between neighbours to retain some boundary context. This does not guarantee whole sentences.
    chunk_size: int = 500
    chunk_overlap: int = 100

    # How many chunks to retrieve and put into the prompt.
    top_k: int = 3

    # Generation settings.
    max_new_tokens: int = 256
    temperature: float = 0.0   # low = factual / deterministic, high = creative


# ==============================================================================
# STEP 1 - LOAD DOCUMENTS
# ==============================================================================
# We read every .txt / .md file in the data folder. Each document keeps its
# file name as "source" so we can cite where an answer came from.
# (To support PDFs, read them with `pypdf` and return the extracted text here.)

@dataclass
class Document:
    text: str
    source: str


def load_documents(data_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(Document(text=text, source=path.relative_to(data_dir).as_posix()))
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found in '{data_dir}'.")
    print(f"[load] {len(docs)} document(s) loaded from '{data_dir}'")
    return docs


# ==============================================================================
# STEP 2 - CHUNKING
# ==============================================================================
# WHY CHUNK?
#   * Embedding models have an input limit (all-MiniLM-L6-v2: 256 tokens).
#   * A whole document as one vector is a "blurry average" of many topics; small
#     chunks give sharp, topic-specific vectors and therefore better retrieval.
#   * The LLM's context window is limited, so we only want the relevant bits.
#
# STRATEGY (simple but effective):
#   Use bounded sliding windows of up to chunk_size characters. Consecutive character windows overlap by chunk_overlap.
#   This generic example does not preserve complete legal sections.

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    windows = []
    for start in range(0, len(text), size - overlap):
        windows.append(text[start:start + size])
        if start + size >= len(text):
            break
    return windows


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Bounded overlapping character windows; preserve each document's source."""
    _sliding_window("", chunk_size, chunk_overlap)  # validate even for empty input
    chunks = []
    for doc in docs:
        for text in _sliding_window(doc.text, chunk_size, chunk_overlap):
            if text.strip():
                chunks.append(Chunk(text, doc.source, len(chunks)))
    print(f"[chunk] {len(chunks)} chunk(s) created")
    return chunks



# ==============================================================================
# STEP 3 - EMBEDDINGS
# ==============================================================================
# An embedding model maps text -> fixed-length vector (here 384 floats).
# Sentences with similar MEANING end up close together, even if they share no
# words ("car" vs "automobile"). We embed every chunk once and store the vectors.
#
# We L2-normalise the vectors (length = 1). Then cosine similarity is simply a
# dot product, which makes search a single matrix multiplication.

class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        print(f"[embed] loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors -> dot product == cosine sim
            show_progress_bar=len(texts) > 50,
        )
        return vectors.astype(np.float32)


# ==============================================================================
# STEP 4 - VECTOR STORE + RETRIEVAL
# ==============================================================================
# The "vector database" is just a matrix of shape (num_chunks, embedding_dim)
# plus the list of chunks it was built from.
#
# Retrieval:
#   1. embed the question           -> q            (shape: [dim])
#   2. similarity = matrix @ q      -> one score per chunk, in [-1, 1]
#   3. take the top_k highest scores
#
# This is exact (brute-force) search. FAISS / Chroma / Qdrant do the same thing
# with approximate indexes that scale to millions of vectors.

class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks: raise ValueError("Cannot index empty chunks")
        self.chunks = chunks
        self.matrix = self.embedder.encode([c.text for c in chunks])
        print(f"[index] vector matrix shape: {self.matrix.shape}")

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        assert self.matrix is not None, "Call add() before search()."
        if top_k <= 0 or not query.strip(): raise ValueError("Require a nonempty query and positive top_k")
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q                      # cosine similarities
        top_k = min(top_k, len(self.chunks))
        best = np.argsort(-scores)[:top_k]            # indices, highest first
        return [(self.chunks[i], float(scores[i])) for i in best]

    # Optional persistence so you don't re-embed on every run.
    def save(self, path: str) -> None:
        np.savez(path, matrix=self.matrix,
                 texts=np.array([c.text for c in self.chunks], dtype=str),
                 sources=np.array([c.source for c in self.chunks], dtype=str))

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.matrix = data["matrix"]
        self.chunks = [Chunk(t, s, i) for i, (t, s) in enumerate(zip(data["texts"], data["sources"]))]


# ==============================================================================
# STEP 5 - PROMPT CONSTRUCTION (the "Augmented" in RAG)
# ==============================================================================
# We give the LLM:
#   * a system instruction: answer ONLY from the context, admit when unsure
#   * the retrieved chunks, numbered and labelled with their source
#   * the user's question
#
# Context-only instructions encourage grounding but do not guarantee faithful answers.

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not contained in the context, say \"I don't know based on the "
    "provided documents.\" Keep answers concise and cite sources like [1], [2]."
)


def build_messages(question: str, retrieved: list[tuple[Chunk, float]]) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] (source: {chunk.source})\n{chunk.text}"
        for i, (chunk, _score) in enumerate(retrieved, start=1)
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Chat models expect a list of role/content messages. The tokenizer's chat
    # template converts this into the exact special-token format the model was
    # trained on (different for Qwen, Llama, Mistral, ...).
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ==============================================================================
# STEP 6 - THE GENERATOR LLM (the "Generation" in RAG)
# ==============================================================================
# We load an open-source instruct model with Hugging Face transformers.
#   * device_map="auto" puts the model on GPU if available, else CPU.
#   * torch_dtype="auto" uses the dtype the model was saved in (bf16/fp16 on GPU).
#
# Want even less memory? Install `bitsandbytes` and load in 4-bit:
#     from transformers import BitsAndBytesConfig
#     quant = BitsAndBytesConfig(load_in_4bit=True)
#     AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, ...)
#
# Prefer Ollama? See the `OllamaLLM` class below - same interface.

class HuggingFaceLLM:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[llm] loading LLM '{model_name}' (first run downloads it) ...")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        # 1. messages -> single prompt string in the model's chat format
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 2. string -> token ids
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # 3. autoregressive generation
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 4. drop the prompt tokens, decode only the newly generated part
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OllamaLLM:
    """
    Alternative generator using a local Ollama server (https://ollama.com).
    Run `ollama pull llama3.2` once, keep Ollama running, then pass
    --backend ollama --llm_model llama3.2 . Needs only the `requests` package.
    """

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        import requests
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_new_tokens},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ==============================================================================
# STEP 7 - PUTTING IT ALL TOGETHER
# ==============================================================================

class SimpleRAG:
    def __init__(self, config: RAGConfig, backend: str = "hf"):
        self.config = config

        # ---- Indexing phase -------------------------------------------------
        docs = load_documents(config.data_dir)
        chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
        self.store = VectorStore(Embedder(config.embedding_model))
        self.store.add(chunks)

        # ---- Generator ------------------------------------------------------
        self.llm = OllamaLLM(config.llm_model) if backend == "ollama" else HuggingFaceLLM(config.llm_model)

    def ask(self, question: str, verbose: bool = True) -> dict:
        # RETRIEVE
        retrieved = self.store.search(question, self.config.top_k)
        # AUGMENT
        messages = build_messages(question, retrieved)
        # GENERATE
        answer = self.llm.generate(messages, self.config.max_new_tokens, self.config.temperature)

        if verbose:
            print("\n" + "=" * 70)
            print(f"Q: {question}")
            print("-" * 70)
            print("Retrieved chunks:")
            for i, (chunk, score) in enumerate(retrieved, start=1):
                preview = chunk.text[:100].replace("\n", " ")
                print(f"  [{i}] score={score:.3f}  {chunk.source}: {preview}...")
            print("-" * 70)
            print(f"A: {answer}")
            print("=" * 70)

        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": c.source, "score": s, "text": c.text, "chunk_id": c.chunk_id} for c, s in retrieved],
        }


# ==============================================================================
# COMMAND-LINE ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    d = RAGConfig()
    p = argparse.ArgumentParser(description="Simple RAG pipeline with an open-source LLM")
    p.add_argument("--data_dir", default=d.data_dir)
    p.add_argument("--embedding_model", default=d.embedding_model)
    p.add_argument("--llm_model", default=d.llm_model)
    p.add_argument("--backend", choices=["hf", "ollama"], default="hf",
                   help="hf = Hugging Face transformers (default), ollama = local Ollama server")
    p.add_argument("--chunk_size", type=int, default=d.chunk_size)
    p.add_argument("--chunk_overlap", type=int, default=d.chunk_overlap)
    p.add_argument("--top_k", type=int, default=d.top_k)
    p.add_argument("--max_new_tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--question", "-q", default=None, help="Ask one question and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = RAGConfig(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    rag = SimpleRAG(config, backend=args.backend)

    if args.question:
        rag.ask(args.question)
        return

    print("\nRAG ready. Type a question (or 'exit' to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            rag.ask(q)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence a HF warning
    main()


# ==============================================================================
# WHERE TO GO NEXT
# ==============================================================================
#  * Better chunking   : split by sentences/tokens (e.g. LangChain's
#                        RecursiveCharacterTextSplitter) or by document structure.
#  * Better retrieval  : hybrid search (BM25 keyword + vectors), or a re-ranker
#                        such as "BAAI/bge-reranker-base" on the top-20 results.
#  * Scale             : replace VectorStore with FAISS, Chroma, Qdrant, pgvector.
#  * Evaluation        : build a small set of question/answer pairs and measure
#                        retrieval hit-rate and answer correctness (e.g. RAGAS).
#  * Frameworks        : LangChain / LlamaIndex / Haystack wrap all of the above,
#                        but now you know what they do under the hood.
