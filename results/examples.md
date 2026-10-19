# Observed model answers

Observed outputs, not automatically graded correctness or verified citations.

Run: 2026-10-19T17:20:51.381786+00:00

The unsupported question tests document grounding: a correct world-knowledge answer would still be unsupported by these documents.

## Which embedding model does this pipeline use?

Category: answerable

### Generated answer

The pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model.

### Retrieved evidence

[1] `rag_notes.md`, chunk 0, cosine 0.4904

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vec

[2] `rag_notes.md`, chunk 1, cosine 0.3050

odel. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete

[3] `rag_notes.md`, chunk 2, cosine 0.1289

monstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.

## How are document vectors stored and compared?

Category: answerable

### Generated answer

Document vectors are stored as a NumPy matrix where each row represents a different chunk of text converted into a 384-dimensional vector through the use of sentence-transformers/all-MiniLM-L6-v2. These vectors are then used to rank the relevance of these chunks for a given query by calculating cosine similarity between the query and the document vectors.

### Retrieved evidence

[1] `rag_notes.md`, chunk 0, cosine 0.4780

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vec

[2] `rag_notes.md`, chunk 1, cosine 0.4457

odel. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete

[3] `rag_notes.md`, chunk 2, cosine 0.1940

monstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.

## Which language model generates the answers?

Category: answerable

### Generated answer

Qwen/Qwen2.5-1.5B-Instruct

### Retrieved evidence

[1] `rag_notes.md`, chunk 0, cosine 0.4927

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vec

[2] `rag_notes.md`, chunk 1, cosine 0.3534

odel. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete

[3] `rag_notes.md`, chunk 2, cosine 0.2212

monstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.

## Which embedding and generation models are used, and why should a user inspect the sources?

Category: multi_part

### Generated answer

The embedding model used is sentence-transformers/all-MiniLM-L6-v2, which converts text chunks into normalized 384-dimensional vectors. The generation model is Qwen/Qwen2.5-1.5B-Instruct, also loaded locally with Hugging Face Transformers.

A user should inspect the sources because:

1. Retrieved passages may be irrelevant or incomplete.
2. A model may ignore instructions or invent citations.
3. Claims or citations in the generated answer should be verified by inspecting the retrieved passages and the answer itself.

### Retrieved evidence

[1] `rag_notes.md`, chunk 0, cosine 0.4821

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vec

[2] `rag_notes.md`, chunk 1, cosine 0.3827

odel. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete

[3] `rag_notes.md`, chunk 2, cosine 0.3159

monstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.

## Who won the 1998 FIFA World Cup?

Category: unsupported

### Generated answer

Based on the given context, there is no specific mention of who won the 1998 FIFA World Cup. Therefore, I don't know based on the provided documents.

### Retrieved evidence

[1] `rag_notes.md`, chunk 2, cosine 0.0203

monstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete. A model may ignore instructions or invent citations. Inspect both the retrieved passages and the generated answer. This example does not automatically validate claims or citations and is not intended for legal advice.

[2] `rag_notes.md`, chunk 0, cosine 0.0086

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) retrieves passages from a document collection, adds them to a language model prompt, and generates an answer using that context. It can provide information outside a model's training data, but does not guarantee accurate answers.

# This demonstration

This pipeline uses sentence-transformers/all-MiniLM-L6-v2 as its embedding model. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vec

[3] `rag_notes.md`, chunk 1, cosine -0.0049

odel. It converts text chunks into normalized 384-dimensional vectors. A NumPy matrix stores the vectors. Cosine similarity ranks chunks for each question.

The generator is Qwen/Qwen2.5-1.5B-Instruct, loaded locally with Hugging Face Transformers. The prompt includes numbered retrieved passages and asks the model to cite sources and acknowledge missing information. This document describes this demonstration, not all RAG systems.

# Limitations

Retrieved passages may be irrelevant or incomplete
