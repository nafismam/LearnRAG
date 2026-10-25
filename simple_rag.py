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

# Read this file from top to bottom: settings, files, chunks, search, prompt, then answer.
# The notebook imports these classes instead of keeping a second copy of the pipeline.
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# SETTINGS: keep the model names and search settings together so they are easy to change.
# These defaults are for a small example; they were not chosen by an accuracy study.
@dataclass
class RAGConfig:

    # Read text and Markdown files from this folder.
    data_dir: str = "data"


    # This model turns English text into 384 numbers used for matching.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


    # This is a different model: it writes the answer from the retrieved text.
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"


    # Cut at most 500 characters at a time. Repeat 100 characters across each boundary.
    # This helps keep some context, but can still cut sentences in half.
    chunk_size: int = 500
    chunk_overlap: int = 100


    # Return this many passages, even if their scores are low. No relevance gate here.
    top_k: int = 3


    max_new_tokens: int = 256
    # Zero turns off sampling. It does not make every answer true or fully repeatable everywhere.
    temperature: float = 0.0


# 1. LOAD: keep the document text together with the path it came from.
@dataclass
class Document:
    text: str
    source: str


# Read files in a consistent order. Skip empty files and stop if no usable files exist.
def load_documents(data_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(data_dir).rglob("*")):

        # Keep the folder in the source name: two folders can both contain a file called note.md.
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(Document(text=text, source=path.relative_to(data_dir).as_posix()))
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found in '{data_dir}'.")
    print(f"[load] {len(docs)} document(s) loaded from '{data_dir}'")
    return docs


# 2. SPLIT: each piece keeps its source path and a number, so we can inspect it later.
@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    # Reject settings that would make empty pieces or prevent us from moving forward.
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    windows = []

    # With size=500 and overlap=100, start at 0, 400, 800, and so on.
    for start in range(0, len(text), size - overlap):
        windows.append(text[start:start + size])

        # We have covered the end of the text; do not add another repeated tail.
        if start + size >= len(text):
            break
    return windows


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Bounded overlapping character windows; preserve each document's source."""

    # Check the settings even if the document list is empty.
    _sliding_window("", chunk_size, chunk_overlap)
    chunks = []
    for doc in docs:
        for text in _sliding_window(doc.text, chunk_size, chunk_overlap):
            if text.strip():

                # Use the current list length as the next ID: 0, 1, 2, ... across all files.
                chunks.append(Chunk(text, doc.source, len(chunks)))
    print(f"[chunk] {len(chunks)} chunk(s) created")
    return chunks


# 3. EMBED: turn each chunk into numbers. Similar texts can have similar vectors.
# Compute document vectors once, rather than repeat that work for every question.
class Embedder:
    def __init__(self, model_name: str):

        # Import here so loading files and testing the chunker do not need this package.
        from sentence_transformers import SentenceTransformer
        print(f"[embed] loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Give each vector length one. That lets a dot product measure cosine similarity later.
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )

        # Store the numbers in one consistent format.
        return vectors.astype(np.float32)


# 4. STORE AND SEARCH: a NumPy table is enough for this tiny collection.
# Row 0 holds chunk 0's vector, row 1 holds chunk 1's vector, and so on.
class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks: raise ValueError("Cannot index empty chunks")
        self.chunks = chunks

        # Keep vector rows in the same order as the chunks, or source labels would be wrong.
        self.matrix = self.embedder.encode([c.text for c in chunks])
        print(f"[index] vector matrix shape: {self.matrix.shape}")

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        assert self.matrix is not None, "Call add() before search()."
        if top_k <= 0 or not query.strip(): raise ValueError("Require a nonempty query and positive top_k")

        # Use the same model for questions and documents so their vectors can be compared.
        q = self.embedder.encode([query])[0]
        # @ compares the question with every stored vector in one operation.
        # Vectors have length one, so these dot products are cosine scores, not probabilities.
        scores = self.matrix @ q
        top_k = min(top_k, len(self.chunks))

        # argsort gives positions from smallest to largest; minus scores puts the best first.
        best = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best]


    # Saving avoids computing document vectors again. SimpleRAG does not call save/load for you.
    def save(self, path: str) -> None:

        # Save ordinary strings, not Python objects that need pickle when loaded.
        np.savez(path, matrix=self.matrix,
                 texts=np.array([c.text for c in self.chunks], dtype=str),
                 sources=np.array([c.source for c in self.chunks], dtype=str))

    def load(self, path: str) -> None:

        # Load only files you expect. You must still use the same embedding model as when saving.
        # This example does not record or verify that model choice in the saved index.
        data = np.load(path, allow_pickle=False)
        self.matrix = data["matrix"]
        self.chunks = [Chunk(t, s, i) for i, (t, s) in enumerate(zip(data["texts"], data["sources"]))]


# 5. PROMPT: ask the writer to use the passages, cite them, and admit missing information.
# These are instructions, not guarantees. Our saved answers omitted the requested inline citations.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not contained in the context, say \"I don't know based on the "
    "provided documents.\" Keep answers concise and cite sources like [1], [2]."
)


def build_messages(question: str, retrieved: list[tuple[Chunk, float]]) -> list[dict]:

    # Label each passage [1], [2], etc. and show its source beside the text.
    context = "\n\n".join(
        f"[{i}] (source: {chunk.source})\n{chunk.text}"
        for i, (chunk, _score) in enumerate(retrieved, start=1)
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


    # The system message sets instructions; the user message holds evidence and the question.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# 6. GENERATE: load Qwen locally. The first run downloads its files.
class HuggingFaceLLM:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[llm] loading LLM '{model_name}' (first run downloads it) ...")
        self.torch = torch
        # The tokenizer converts between readable text and the token IDs the model uses.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Let the library choose the saved number format and suitable devices.
        # A larger model may need more memory than the machine has available.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        # Turn off training-only behavior such as dropout. We are answering, not training.
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:

        # Put the messages into the chat format this particular model expects.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Convert the prompt into tokens and put them on the model's device.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)


        # No training gradients are needed, which saves memory and work.
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )


        # The output contains the original prompt too. Keep only the newly written answer.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Optional alternative: ask an already-running local Ollama server to generate the answer.
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

        # Stop on an HTTP error instead of pretending the failed request returned an answer.
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# 7. CONNECT THE PARTS: prepare the documents and load the models once for this object.
class SimpleRAG:
    def __init__(self, config: RAGConfig, backend: str = "hf"):
        self.config = config


        docs = load_documents(config.data_dir)
        chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
        self.store = VectorStore(Embedder(config.embedding_model))
        self.store.add(chunks)


        self.llm = OllamaLLM(config.llm_model) if backend == "ollama" else HuggingFaceLLM(config.llm_model)

    # Each question follows three steps. No document re-indexing is needed here.
    def ask(self, question: str, verbose: bool = True) -> dict:


        # Retrieve: find the closest document pieces.
        retrieved = self.store.search(question, self.config.top_k)

        # Augment: put those pieces into the prompt alongside the question.
        messages = build_messages(question, retrieved)

        # Generate: let the model write an answer using that prompt.
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


        # Return evidence separately so a reader can inspect it, even if the answer has no citations.
        # This does not automatically check that every sentence is supported.
        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": c.source, "score": s, "text": c.text, "chunk_id": c.chunk_id} for c, s in retrieved],
        }


# COMMAND LINE: read settings such as --top_k and --question from the user's command.
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


# Build the pipeline, then answer one question or keep asking until the user exits.
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


# Run main() when launched as a script, but not when imported by the notebook or tests.
if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
