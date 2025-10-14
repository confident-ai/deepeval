import pytest

from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker


##########################
# Helpers / Test Doubles #
##########################


class StubEmbedder:
    """A minimal stand-in for DeepEvalBaseEmbeddingModel used in tests.

    This stub avoids calling a real embedding model by returning fixed length
    dummy vectors. It supports both synchronous and asynchronous methods so
    that DocumentChunker can run without depending on external services.
    """

    def embed_texts(self, xs):
        return [[0.0] * 4 for _ in xs]

    def a_embed_texts(self, xs):
        raise NotImplementedError

    def embed_text(self, x):
        return [0.0] * 4

    async def a_embed_text(self, x):
        return [0.0] * 4


class StubAsyncEmbedder(StubEmbedder):
    """An async variant of StubEmbedder.

    Unlike StubEmbedder, this implementation provides a working asynchronous
    `a_embed_texts` method that returns dummy embeddings, so that async
    chunking methods, such as DocumentChunker.a_chunk_doc, can be tested.
    """

    async def a_embed_texts(self, xs):
        return [[0.0] * 4 for _ in xs]


class FakeCollection:
    """A fake ChromaDB collection used in tests.

    This fake captures calls to ``add`` so tests can inspect the documents,
    embeddings, metadata, and IDs passed during chunking without requiring a
    real ChromaDB backend.
    """

    def __init__(self):
        self.add_calls = []

    def add(self, documents, embeddings, metadatas, ids):
        self.add_calls.append((documents, embeddings, metadatas, ids))


class FakeClient:
    """A fake ChromaDB client that manages FakeCollections in memory.

    It implements ``get_collection`` and ``create_collection`` so that tests
    can simulate both cache hits and cache misses when DocumentChunker tries
    to retrieve or create a collection.
    """

    def __init__(self):
        self.collections = {}

    def get_collection(self, name):
        if name not in self.collections:
            raise RuntimeError("not found")
        return self.collections[name]

    def create_collection(self, name):
        c = FakeCollection()
        self.collections[name] = c
        return c


class FakeChromaMod:
    """A fake Chroma module shim with only PersistentClient.

    This lets tests monkeypatch ``_chroma_mod`` with a fake implementation that
    always returns the provided FakeClient instance.
    """

    def __init__(self, client):
        self._client = client

    def PersistentClient(self, path, **kwargs):
        return self._client


###########################
# Markdown / Loader tests #
###########################


@pytest.mark.parametrize("ext", [".md", ".markdown", ".mdx"])
def test_markdown_family_preserves_table(tmp_path, ext):
    """Verify that markdown family extensions (.md, .markdown, .mdx)
    are all loaded via the TextLoader and that table formatting is
    preserved in the loaded document sections.
    """
    p = tmp_path / f"sample{ext}"
    p.write_text("# T\n\n| A | B |\n| - | - |\n| 1 | 2 |\n", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding="utf-8")
    assert dc.sections
    assert any("| A | B |" in d.page_content for d in dc.sections)


def test_unsupported_extension(tmp_path):
    """Ensure that get_loader raises ValueError when asked to load
    a file with an unsupported extension.
    """
    p = tmp_path / "weird.xyz"
    p.write_text("hello", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    with pytest.raises(ValueError):
        dc.get_loader(str(p), encoding="utf-8")


def test_textloader_autodetect_encoding(tmp_path):
    """Confirm that the TextLoader correctly autodetects encodings.
    This test writes a UTF-8 BOM-prefixed file and verifies that the
    loader strips the BOM and returns the expected text content.
    """
    # UTF-8 BOM content should still parse correctly via autodetect
    p = tmp_path / "bom.md"
    p.write_bytes(b"\xef\xbb\xbfHello")
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding=None)
    assert any("Hello" in d.page_content for d in dc.sections)


def test_count_tokens_runs(tmp_path):
    """Check that DocumentChunker.count_tokens runs successfully after
    loading a text file, and that it produces a positive integer token count.
    """
    p = tmp_path / "a.md"
    p.write_text("a b c", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding="utf-8")
    assert isinstance(dc.text_token_count, int)
    assert dc.text_token_count > 0


#############################################################
# Lazy import behavior (dependency required only when used) #
#############################################################


def test_lazy_imports_langchain_required_on_loader(monkeypatch):
    """Verify that attempting to load a document requires LangChain.

    This test monkeypatches ``_get_langchain`` to raise ImportError,
    simulating a missing LangChain installation. When
    ``DocumentChunker.get_loader`` is called, it should propagate
    the ImportError since LangChain is required for loader creation.
    """
    # simulate LangChain missing by stubbing the getter to raise
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._get_langchain",
        lambda: (_ for _ in ()).throw(ImportError("no langchain")),
    )
    dc = DocumentChunker(StubEmbedder())
    with pytest.raises(ImportError):
        dc.get_loader("x.md", encoding="utf-8")


def test_lazy_imports_chromadb_required_on_chunk(monkeypatch, tmp_path):
    """Verify that attempting to chunk a document requires ChromaDB.

    After loading a markdown file via LangChain, this test monkeypatches
    ``get_chromadb`` to raise ImportError, simulating a missing ChromaDB
    installation. When ``DocumentChunker.chunk_doc`` is called, it should
    propagate the ImportError since ChromaDB is required for chunking.
    """
    p = tmp_path / "x.md"
    p.write_text("hello", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    # make sure loading (LangChain path) works
    dc.load_doc(str(p), encoding="utf-8")

    # now simulate chromadb missing only for the chunking path
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker.get_chromadb",
        lambda: (_ for _ in ()).throw(ImportError("no chroma")),
    )
    with pytest.raises(ImportError):
        dc.chunk_doc()


###############################
# Chroma integration (mocked) #
###############################


def test_chunk_doc_raises_if_not_loaded():
    """Ensure that calling chunk_doc before load_doc raises ValueError.

    DocumentChunker requires a loaded document before chunking. This test
    verifies that attempting to chunk prematurely fails with the correct error.
    """
    dc = DocumentChunker(StubEmbedder())
    with pytest.raises(ValueError):
        dc.chunk_doc()


def test_chunk_doc_batches_and_metadata(monkeypatch, tmp_path):
    """Verify batching behavior and metadata when chunking a large document.

    - Creates a large markdown file to force multiple batches.
    - Monkeypatches Chroma to use a FakeClient/FakeCollection.
    - Confirms that:
      * Each batch size does not exceed the hard limit (5461).
      * All documents, IDs, and metadata lists are aligned in length.
      * Each metadata entry contains a ``source_file`` key.
    """
    # prepare many chunks to force batching
    p = tmp_path / "big.md"
    p.write_text(("x\n" * 6000), encoding="utf-8")  # many tiny chunks
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding="utf-8")

    fake_client = FakeClient()
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        FakeChromaMod(fake_client),
        raising=True,
    )

    collection = dc.chunk_doc(chunk_size=1, chunk_overlap=0)
    assert isinstance(collection, FakeCollection)
    assert collection.add_calls, "expected chunks to be added"
    assert all(len(docs) <= 5461 for docs, *_ in collection.add_calls)
    for docs, _emb, metas, ids in collection.add_calls:
        assert len(docs) == len(metas) == len(ids)
        assert all(isinstance(m, dict) and "source_file" in m for m in metas)


@pytest.mark.asyncio
async def test_a_chunk_doc_works(monkeypatch, tmp_path):
    """Verify that a_chunk_doc works end 2 end with async embedding.

    - Uses StubAsyncEmbedder to provide async embeddings.
    - Monkeypatches Chroma with a FakeClient.
    - Confirms that chunks are added to the fake collection without error.
    """
    p = tmp_path / "big_async.md"
    p.write_text(("x\n" * 2000), encoding="utf-8")
    dc = DocumentChunker(StubAsyncEmbedder())
    dc.load_doc(str(p), encoding="utf-8")

    fake_client = FakeClient()
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        FakeChromaMod(fake_client),
        raising=True,
    )

    collection = await dc.a_chunk_doc(chunk_size=1, chunk_overlap=0)
    assert isinstance(collection, FakeCollection)
    assert collection.add_calls


def test_chunk_doc_uses_existing_collection(monkeypatch, tmp_path):
    """Ensure chunk_doc reuses an existing collection if present.

    - Prepopulates FakeClient with a collection named for the default chunk
      parameters.
    - Verifies that DocumentChunker returns the existing collection rather than
      creating a new one, and does not perform additional adds.
    """
    p = tmp_path / "a.md"
    p.write_text("hello", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding="utf-8")

    fake_client = FakeClient()
    existing = FakeCollection()
    fake_client.collections["processed_chunks_1024_0"] = existing

    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        FakeChromaMod(fake_client),
        raising=True,
    )

    returned = dc.chunk_doc()
    assert returned is existing
    assert existing.add_calls == []  # no new adds on cache hit


def test_persistent_path_and_collection_name(monkeypatch, tmp_path):
    """Confirm persistent client path and collection naming conventions.

    - Loads a file with version suffix in its name.
    - Monkeypatches Chroma client to capture the path argument.
    - Asserts that:
      * PersistentClient path is derived from the file basename (no extension).
      * Collection name includes both chunk_size and chunk_overlap values.
    """
    p = tmp_path / "notes.v1.md"
    p.write_text("data", encoding="utf-8")
    dc = DocumentChunker(StubEmbedder())
    dc.load_doc(str(p), encoding="utf-8")

    captured = {}
    fake_client = FakeClient()

    class CapturingChroma:
        def PersistentClient(self, path, **kwargs):
            captured["path"] = path
            return fake_client

    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        CapturingChroma(),
        raising=True,
    )

    dc.chunk_doc(chunk_size=123, chunk_overlap=7)

    assert captured["path"].endswith(".vector_db/notes.v1")
    assert "processed_chunks_123_7" in fake_client.collections
