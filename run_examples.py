"""Run real models once and save answers plus evidence; no simulated responses."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import time

from simple_rag import RAGConfig, SimpleRAG

QUESTIONS = [
    ('answerable', 'Which embedding model does this pipeline use?'),
    ('answerable', 'How are document vectors stored and compared?'),
    ('answerable', 'Which language model generates the answers?'),
    ('multi_part', 'Which embedding and generation models are used, and why should a user inspect the sources?'),
    ('unsupported', 'Who won the 1998 FIFA World Cup?'),
]


def main():
    root = Path(__file__).resolve().parent
    config = RAGConfig(data_dir=str(root / 'data'), temperature=0.0)
    pipeline = SimpleRAG(config)
    results = []
    for kind, question in QUESTIONS:
        start = time.perf_counter()
        result = pipeline.ask(question)
        result.update(kind=kind, elapsed_seconds=round(time.perf_counter() - start, 3))
        results.append(result)
    report = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': asdict(config),
        'packages': {p: version(p) for p in ['numpy', 'torch', 'transformers', 'sentence-transformers']},
        'documents_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((root / 'data').rglob('*')) if p.is_file()},
        'generation_model_commit': getattr(pipeline.llm.model.config, '_commit_hash', None),
        'results': results,
        'scope': 'Observed outputs, not automatically graded correctness or verified citations.',
    }
    output = root / 'results'
    output.mkdir(exist_ok=True)
    (output / 'examples.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Observed model answers', '', report['scope'], '',
             f"Run: {report['executed_at_utc']}", '',
             'The unsupported question tests document grounding: a correct world-knowledge answer '
             'would still be unsupported by these documents.', '']
    for result in results:
        lines += [f"## {result['question']}", '', f"Category: {result['kind']}", '',
                  '### Generated answer', '', result['answer'], '', '### Retrieved evidence', '']
        for i, source in enumerate(result['sources'], 1):
            lines += [f"[{i}] `{source['source']}`, chunk {source['chunk_id']}, cosine {source['score']:.4f}",
                      '', source['text'], '']
    (output / 'examples.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved real model outputs to results/examples.json and results/examples.md')


if __name__ == '__main__':
    main()
