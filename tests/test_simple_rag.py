import numpy as np
import pytest
from simple_rag import Document, Chunk, chunk_documents, VectorStore, build_messages, load_documents, SimpleRAG, RAGConfig

# Deliberately simple vectors isolate retrieval mechanics; they do not evaluate MiniLM.
class StubEmbedder:
    def encode(self, texts):
        return np.array([[1.,0.] if 'apple' in t else [0.,1.] for t in texts], dtype=np.float32)

# Check BOTH bounds and reconstruction: valid-looking chunk sizes could still hide lost text.
def test_chunk_boundaries_and_coverage():
    text = 'abcdefghijklmnopqrstuvwxyz' * 8
    chunks = chunk_documents([Document(text,'a')],50,10)
    assert all(0<len(c.text)<=50 for c in chunks)
    assert chunks[0].text + ''.join(c.text[10:] for c in chunks[1:]) == text
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

# One test function covers three invalid configurations, producing three test cases.
@pytest.mark.parametrize('size,overlap',[(0,0),(10,-1),(10,10)])
def test_invalid_chunk_parameters(size,overlap):
    with pytest.raises(ValueError): chunk_documents([],size,overlap)

# A same-name collision is realistic when users upload documents from different folders.
def test_distinct_relative_sources(tmp_path):
    for folder in ['one','two']:
        (tmp_path/folder).mkdir(); (tmp_path/folder/'same.md').write_text('apple')
    assert {d.source for d in load_documents(str(tmp_path))}=={'one/same.md','two/same.md'}

# Round-trip persistence must preserve which evidence wins a known query.
def test_retrieval_and_persistence(tmp_path):
    store=VectorStore(StubEmbedder()); store.add([Chunk('apple','a',0),Chunk('banana','b',1)])
    assert store.search('apple',1)[0][0].source=='a'
    store.save(str(tmp_path/'index.npz'))
    restored=VectorStore(StubEmbedder()); restored.load(str(tmp_path/'index.npz'))
    assert restored.search('apple',1)[0][0].source=='a'
    with pytest.raises(ValueError): restored.search('apple',0)

# A fake generator checks that retrieved evidence actually reaches the prompt.
# This verifies component connection, not answer quality or model instruction following.
def test_retrieve_augment_generate_wiring():
    class StubLLM:
        def generate(self,messages,*args):
            assert 'apple' in messages[1]['content'] and '(source: a)' in messages[1]['content']
            return 'An apple [1].'
    # Deliberately stub both models: this verifies orchestration, not model quality.
    rag=SimpleRAG.__new__(SimpleRAG); rag.config=RAGConfig(top_k=1)
    rag.store=VectorStore(StubEmbedder()); rag.store.add([Chunk('apple','a',0)])
    rag.llm=StubLLM(); result=rag.ask('apple',verbose=False)
    assert result['answer']=='An apple [1].' and result['sources'][0]['chunk_id']==0
