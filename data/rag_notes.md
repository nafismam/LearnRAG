# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.
