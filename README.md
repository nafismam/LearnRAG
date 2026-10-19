# LearnRAG

A small, step-by-step Python project for learning **Retrieval-Augmented Generation (RAG)**. Ask a question about your documents, retrieve related passages, and give those passages to a language model to help it answer.

This repository includes a command-line script, a notebook, a sample document, and component tests. No paid API key is required for the default public models.

## Why use RAG?

A language model may not know what is in your documents. Asking it directly can produce an unrelated or unsupported answer. RAG supplies relevant document text at question time, without retraining the model.

For example, the sample document states which embedding model this project uses. The pipeline retrieves that passage and includes it in the prompt when you ask, “Which embedding model does this pipeline use?”

RAG can help ground an answer, but does not guarantee that the answer is correct. Always inspect the retrieved text as well as the response.

## What happens inside?

```text
Documents → small overlapping chunks → embeddings → NumPy index
Question  → embedding → closest chunks → source-labeled prompt → LLM answer
```

| Step | What it does | Why it is there |
| --- | --- | --- |
| Load | Reads `.txt` and `.md` files in `data/` | Gives the system a document collection |
| Chunk | Splits text into windows of up to 500 characters, with 100-character overlap | Keeps passages small and retains some context across boundaries |
| Embed | MiniLM converts each chunk into a numerical vector | Allows matching by learned text representations rather than exact words alone |
| Index | Stores normalized vectors in a NumPy matrix | Keeps document vectors ready for search |
| Retrieve | Embeds a question and ranks chunks by cosine similarity | Finds the closest passages; the default returns three |
| Augment | Adds passages and numbered source labels to the prompt | Gives the generator evidence it can refer to |
| Generate | Qwen produces an answer from the prompt | Completes the generation part of RAG |

An **embedding** is a list of numbers representing text. **Cosine similarity** compares vector directions; a higher score means a closer match under this model. It is not a probability that an answer is correct.

## Run in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/nafismam/LearnRAG/blob/main/simple_rag.ipynb).

1. Select **Runtime → Change runtime type → T4 GPU**, if available.
2. Run the cells in order. The setup cell gets the code and sample data; the next cell installs dependencies.
3. Wait for the model downloads and loading to finish.
4. Inspect retrieved sources beside each generated answer.
5. Change a question or edit a document in `data/`. Rebuild the pipeline after editing documents.

The first run needs Internet access and several GB of disk space. Model files are cached for later runs. CPU inference is supported but can be slow; free Colab GPU availability varies. Document processing and inference run inside your runtime, without calling a hosted generation API.

## Run locally

Use Python 3.10 or newer. On macOS or Linux:

```bash
git clone https://github.com/nafismam/LearnRAG.git
cd LearnRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simple_rag.py --question "Which embedding model does this pipeline use?"
```

On Windows, activate with `.venv\Scripts\activate` instead. Install a suitable PyTorch build for your machine if needed; see [the official installation guide](https://pytorch.org/get-started/locally/).

For interactive questions:

```bash
python simple_rag.py
```

For your own documents, place `.txt` or `.md` files in a folder and run:

```bash
python simple_rag.py --data_dir my_documents --top_k 3 --question "Your question"
```

The default models are `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `Qwen/Qwen2.5-1.5B-Instruct` for generation. PDFs are not supported directly. The script also includes an optional Ollama backend, which requires a separately installed, running Ollama server and a downloaded model.

## Files

- `simple_rag.py`: the pipeline, component classes, and command-line interface.
- `simple_rag.ipynb`: a guided demonstration using the same Python implementation.
- `data/rag_notes.md`: sample content used to answer demonstration questions.
- `tests/test_simple_rag.py`: checks for core behavior without downloading models.
- `requirements.txt`: runtime dependencies; `requirements-dev.txt` adds pytest.

The script separates document loading, chunking, embeddings, vector search, prompt construction, and generation so each step is easy to inspect.

## What do the tests check?

Regression tests help catch changes that break previously expected behavior.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

| Check | What a failure would mean |
| --- | --- |
| Chunk size, overlap and coverage | Text is missing or chunks exceed their configured size |
| Invalid chunk settings | Bad settings are silently accepted |
| Relative source paths | Same-named files from different folders become indistinguishable |
| Retrieval and save/load | Known vectors return the wrong source or persistence changes results |
| Retrieve → prompt → generate wiring | Retrieved text is missing from the prompt or results lose source information |

There are seven test cases, including three invalid-configuration cases. Tests use deliberately simple model doubles: they check software behavior, not real embedding quality or generated-answer accuracy.

## Validation status and limits

Seven component tests passed in the preparation environment. Full model execution remains **unverified** because Hugging Face downloads returned HTTP 403 there. The notebook does not contain fabricated model outputs; run it in your own runtime to observe actual answers.

This is an educational, generic document pipeline, not a legal-advice application. It uses character windows, not complete statutory sections. Search always returns top-k passages without a relevance gate, even when the documents cannot answer a question. The prompt requests abstention and citations, but neither is automatically enforced or verified. Inputs are not yet checked against tokenizer or context-window limits, and document instructions can interfere with model behavior.

Useful next steps are token-aware chunking, prompt budgeting, irrelevant-query handling, citation validation, and evaluation against a small set of questions with known supporting passages. There are no claimed accuracy or faithfulness scores yet.

## Capture actual answers

Run `python run_examples.py` after model downloads are available. It asks five fixed questions and saves real answers, retrieved passages, scores, model configuration and document hashes to `results/examples.json` and `results/examples.md`. The unsupported question checks whether the model stays within the documents, even when it may know an answer from training. These are observations, not automatically graded accuracy results. No answer files are included until the real-model run succeeds.
