# Observed runs

## Independently executed CPU run

[Five answers with full retrieved evidence](examples.md) · [Machine-readable results](examples.json)

The default MiniLM embedding model and Qwen2.5-1.5B-Instruct generator ran locally on CPU. The JSON records UTC execution time, package versions, generation-model commit, configuration, document hashes and individual generation durations. Downloads and model loading are excluded from those per-question durations. The separately executed notebook also contains actual saved outputs.

| Question | Observed behavior |
| --- | --- |
| Which embedding model is used? | Named the model stated in the source |
| How are vectors stored and compared? | Described the NumPy matrix and cosine similarity |
| Which generator is used? | Named Qwen2.5-1.5B-Instruct |
| Which models are used, and why inspect sources? | Answered both parts using information in the sample document |
| Who won the 1998 FIFA World Cup? | Abstained because the supplied documents do not establish an answer |

These are qualitative observations on five hand-picked questions, not independently graded accuracy measurements. None of the five answers included the requested numbered inline citations. The returned source records provide evidence for inspection, but do not validate the generated claims automatically.

The corpus contains one short document producing three chunks. With top_k=3, every question retrieves all three; this does not test selective retrieval across a larger corpus. Relevant queries had top similarities around 0.45–0.57, while the unrelated question's top score was about 0.02. Scores do not establish a calibrated confidence threshold. One abstention example cannot establish reliable abstention behavior.

## User-supplied Colab run

[Colab observations](colab_observations.md) preserves the three answers and retrieved previews from the owner's supplied text export. It is labeled separately from the independently executed CPU run. The general RAG answer includes broader claims not established by the sample source, illustrating why factual-sounding answers still need review.

## Reproduce

Run `python run_examples.py` from the repository after installing dependencies. To execute the guided notebook, open it in Colab and run all cells. Different devices, package versions or upstream model revisions may change generated wording. Component tests use model doubles and assess software behavior separately from these model observations.
