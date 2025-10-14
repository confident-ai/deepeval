import pytest
import os

from itertools import chain
from types import SimpleNamespace

from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.models.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


# stub the langchain loader/splitter
class _FakeTextLoader:
    def __init__(self, path, encoding=None, autodetect_encoding=True):
        self._path = path

    def load(self):
        class _Doc:
            page_content = (
                "The answer to life,\nthe universe and everything:\n42"
            )

        return [_Doc()]

    async def aload(self):
        return self.load()


class _FakeSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self._size = chunk_size
        self._ov = chunk_overlap

    def split_documents(self, docs):
        class _Doc:
            def __init__(self, txt):
                self.page_content = txt

        # 10 small chunks
        return [_Doc(f"c{j}") for j in range(10)]


def _make_stub_embedder():
    class _Stub:
        # used by DocumentChunker
        def embed_texts(self, xs):
            return [[0.0, 0.0, 0.0, 0.0] for _ in xs]

        # used by DocumentChunker
        async def a_embed_texts(self, xs):
            return [[0.0, 0.0, 0.0, 0.0] for _ in xs]

        # used by sync ContextGenerator
        def embed_text(self, x):
            return [0.0, 0.0, 0.0, 0.0]

        # used by async ContextGenerator
        async def a_embed_text(self, x):
            return [0.0, 0.0, 0.0, 0.0]

    return _Stub()


class _CapturingCollection:
    def __init__(self, name, count_value=10):
        self.name = name
        self._count_value = count_value
        self.add_calls = []

    def count(self):
        return self._count_value

    def get(self, ids):
        # flat list of strings -> flat list of docs
        return {"documents": [f"D{i}" for i in ids]}

    def query(self, _embedding, n_results):
        # 2D: index 0 is the "query row"
        docs = [["q"] + [f"n{j}" for j in range(n_results - 1)]]
        dists = [[0.0] + [0.1] * (n_results - 1)]
        return {"documents": docs, "distances": dists}

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))


class _CapturingClient:
    def __init__(self, count_value=10):
        self.collections = {}
        self.delete_calls = []
        self._count_value = count_value

    def get_collection(self, name):
        if name not in self.collections:
            raise RuntimeError("not found")
        return self.collections[name]

    def create_collection(self, name):
        collection = _CapturingCollection(
            name=name, count_value=self._count_value
        )
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        self.delete_calls.append(name)
        self.collections.pop(name, None)


class _CapturingChromaMod:
    def __init__(self, client: "_CapturingClient" = None):
        self.calls = []
        self.client = client or _CapturingClient()

    def PersistentClient(self, path, **kwargs):
        anon = getattr(kwargs.get("settings"), "anonymized_telemetry", None)
        self.calls.append({"path": path, "anon": anon})
        return self.client


def _patch_langchain(monkeypatch):
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._langchain_ns",
        SimpleNamespace(
            LCDocument=object,
            TokenTextSplitter=_FakeSplitter,
            TextSplitter=_FakeSplitter,
            PyPDFLoader=_FakeTextLoader,
            TextLoader=_FakeTextLoader,
            Docx2txtLoader=_FakeTextLoader,
            BaseLoader=_FakeTextLoader,
        ),
        raising=True,
    )


@pytest.fixture
def context_generator_fixture():
    generator = ContextGenerator(
        document_paths=[
            os.path.join(MODULE_DIR, "synthesizer_data", "pdf_example.pdf")
        ],
        embedder=OpenAIEmbeddingModel(),
    )
    yield generator


@pytest.fixture
def ensure_synthesizer_data():
    data_dir = os.path.join(MODULE_DIR, "synthesizer_data")
    pdf_path = os.path.join(data_dir, "pdf_example.pdf")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF file not found: {pdf_path}")


def test_generate_contexts(
    context_generator_fixture,
    ensure_synthesizer_data,
):
    context_generator: ContextGenerator = context_generator_fixture
    contexts, source_files, context_scores = (
        context_generator.generate_contexts(
            max_contexts_per_source_file=2,
            min_contexts_per_source_file=1,
        )
    )
    unique_chunks = len(set(chain.from_iterable(contexts)))
    assert contexts is not None, "Contexts should not be None"
    assert source_files is not None, "Source files should not be None"
    assert context_scores is not None, "Context scores should not be None"
    assert len(contexts) > 0, "No contexts were generated"
    assert unique_chunks > 0, "No unique chunks were utilized"
    assert (
        unique_chunks <= context_generator.total_chunks
    ), "More chunks utilized than available"


def test_multiple_context_generations(
    context_generator_fixture,
    ensure_synthesizer_data,
):
    context_generator: ContextGenerator = context_generator_fixture
    contexts1, _, _ = context_generator.generate_contexts(
        max_contexts_per_source_file=2,
        min_contexts_per_source_file=1,
    )
    contexts2, _, _ = context_generator.generate_contexts(
        max_contexts_per_source_file=2,
        min_contexts_per_source_file=1,
    )
    unique_chunks1 = len(set(chain.from_iterable(contexts1)))
    unique_chunks2 = len(set(chain.from_iterable(contexts2)))
    assert (
        contexts1 is not None and contexts2 is not None
    ), "Both context generations should succeed"
    assert (
        len(contexts1) > 0 and len(contexts2) > 0
    ), "Both generations should produce contexts"
    assert (
        unique_chunks1 > 0 and unique_chunks2 > 0
    ), "Both generations should produce unique chunks"
    assert (
        unique_chunks1 <= context_generator.total_chunks
        and unique_chunks2 <= context_generator.total_chunks
    ), "More chunks utilized than available"


def test_many_docs_should_spawn_a_single_chroma_client(monkeypatch, tmp_path):
    """
    Ensure ContextGenerator uses a single shared Chroma PersistentClient per run.

    Even with multiple documents, only one PersistentClient should be constructed
    and reused across all document pipelines. We assert exactly one call to
    PersistentClient(), which keeps file handles and FS contention bounded.
    """
    # fabricate many tiny ".md" docs
    num_docs = 10
    doc_paths = []
    for i in range(num_docs):
        p = tmp_path / f"doc_{i}.md"
        p.write_text("x\n" * 10, encoding="utf-8")
        doc_paths.append(str(p))

    # a capturing chroma
    cap_chroma = _CapturingChromaMod()
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        cap_chroma,
        raising=True,
    )
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._langchain_ns",
        None,  # lazy load
        raising=False,
    )

    # Build a minimal langchain namespace
    from types import SimpleNamespace

    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._langchain_ns",
        SimpleNamespace(
            LCDocument=object,
            TokenTextSplitter=_FakeSplitter,
            TextSplitter=_FakeSplitter,
            PyPDFLoader=_FakeTextLoader,
            TextLoader=_FakeTextLoader,
            Docx2txtLoader=_FakeTextLoader,
            BaseLoader=_FakeTextLoader,
        ),
        raising=True,
    )

    from deepeval.synthesizer.chunking.context_generator import ContextGenerator

    gen = ContextGenerator(
        document_paths=doc_paths,
        embedder=_make_stub_embedder(),
        chunk_size=50,  # small so we "chunk"
        chunk_overlap=0,
        max_retries=1,  # keep the loop short
        filter_threshold=0.0,
        similarity_threshold=0.0,
    )

    # run the sync path
    contexts, srcs, scores = gen.generate_contexts(
        max_contexts_per_source_file=1,  # one context per doc is enough
        min_contexts_per_source_file=1,
    )

    # check that we processed something
    assert len(contexts) == len(srcs) == num_docs

    if len(cap_chroma.calls) != 1:
        pytest.fail(
            f"Expected 1 PersistentClient() call; got {len(cap_chroma.calls)}"
        )


@pytest.mark.asyncio
async def test_async_many_docs_uses_single_chroma_client(monkeypatch, tmp_path):
    """
    Ensure a_generate_contexts uses a single shared Chroma PersistentClient per run,
    even with multiple documents.
    """
    # make tiny docs
    num_docs = 3
    doc_paths = []
    for i in range(num_docs):
        p = tmp_path / f"doc_{i}.md"
        p.write_text("x\n" * 10, encoding="utf-8")
        doc_paths.append(str(p))

    _patch_langchain(monkeypatch)

    # single client backing the whole run
    cap_client = _CapturingClient(count_value=10)
    cap_chroma = _CapturingChromaMod(cap_client)
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        cap_chroma,
        raising=True,
    )

    gen = ContextGenerator(
        document_paths=doc_paths,
        embedder=_make_stub_embedder(),
        chunk_size=50,
        chunk_overlap=0,
        max_retries=1,
        filter_threshold=0.0,
        similarity_threshold=0.0,
    )

    contexts, srcs, scores = await gen.a_generate_contexts(
        max_contexts_per_source_file=1,
        min_contexts_per_source_file=1,
    )

    # processed something
    assert len(contexts) == len(srcs) == num_docs
    # exactly one PersistentClient() call
    assert len(cap_chroma.calls) == 1


##############
# Validation #
##############


def test_sync_min_context_size_validation(monkeypatch, tmp_path, caplog):
    """
    If a document collection has fewer chunks than `min_context_size`,
    the sync path should log an error for that doc and continue (no raise).
    """
    _patch_langchain(monkeypatch)

    p = tmp_path / "tiny.md"
    p.write_text("short", encoding="utf-8")

    cap_client = _CapturingClient(count_value=2)
    cap_chroma = _CapturingChromaMod(cap_client)
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        cap_chroma,
        raising=True,
    )

    gen = ContextGenerator(
        document_paths=[str(p)],
        embedder=_make_stub_embedder(),
        chunk_size=50,
        chunk_overlap=0,
        max_retries=1,
        filter_threshold=0.0,
        similarity_threshold=0.0,
    )

    with caplog.at_level("ERROR"):
        contexts, srcs, scores = gen.generate_contexts(
            max_contexts_per_source_file=1,
            min_contexts_per_source_file=1,
            min_context_size=5,  # larger than count() results in validation failure
        )

    # no contexts produced for the failing doc
    assert contexts == []
    assert srcs == []
    assert scores == []
    # and the failure is logged
    assert any(
        "Document pipeline failed for" in rec.message for rec in caplog.records
    )


###########################
# Failures and exceptions #
###########################


@pytest.mark.asyncio
async def test_async_per_doc_failure_is_logged_and_others_continue(
    monkeypatch, tmp_path, caplog
):
    """
    When one document's a_chunk_doc raises, we should log the error and
    continue processing other documents instead of crashing.
    """
    _patch_langchain(monkeypatch)

    # two docs, first will fail
    p1 = tmp_path / "bad.md"
    p2 = tmp_path / "good.md"
    p1.write_text("aaa", encoding="utf-8")
    p2.write_text("bbb", encoding="utf-8")
    doc_paths = [str(p1), str(p2)]

    # normal client
    cap_client = _CapturingClient(count_value=10)
    cap_chroma = _CapturingChromaMod(cap_client)
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        cap_chroma,
        raising=True,
    )

    # monkeypatch DocumentChunker.a_chunk_doc to raise for p1 only
    async def _boom(self, *args, **kwargs):
        # self.source_file is set by load_doc
        if getattr(self, "source_file", "").endswith("bad.md"):
            raise RuntimeError("boom")
        # fallback to real path by creating a collection
        client = cap_client
        name = "processed_chunks_1024_0"
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name)

    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker.DocumentChunker.a_chunk_doc",
        _boom,
        raising=True,
    )

    gen = ContextGenerator(
        document_paths=doc_paths,
        embedder=_make_stub_embedder(),
        chunk_size=50,
        chunk_overlap=0,
        max_retries=1,
        filter_threshold=0.0,
        similarity_threshold=0.0,
    )

    with caplog.at_level("ERROR"):
        contexts, srcs, scores = await gen.a_generate_contexts(
            max_contexts_per_source_file=1,
            min_contexts_per_source_file=1,
        )

    # should still have processed the doc that did not cause an error
    assert len(contexts) == 1
    assert any(
        "Document pipeline failed for" in rec.message for rec in caplog.records
    )


#####################
# Deletion Tracking #
#####################


def test_sync_deletes_one_collection_per_doc(monkeypatch, tmp_path):
    """
    After each document pipeline completes, we call delete_collection(name).
    Verify we issue exactly one delete per document.
    """
    _patch_langchain(monkeypatch)

    num_docs = 3
    doc_paths = []
    for i in range(num_docs):
        p = tmp_path / f"doc_{i}.md"
        p.write_text("x\n" * 10, encoding="utf-8")
        doc_paths.append(str(p))

    cap_client = _CapturingClient(count_value=10)
    cap_chroma = _CapturingChromaMod(cap_client)
    monkeypatch.setattr(
        "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
        cap_chroma,
        raising=True,
    )

    gen = ContextGenerator(
        document_paths=doc_paths,
        embedder=_make_stub_embedder(),
        chunk_size=50,
        chunk_overlap=0,
        max_retries=1,
        filter_threshold=0.0,
        similarity_threshold=0.0,
    )

    contexts, srcs, scores = gen.generate_contexts(
        max_contexts_per_source_file=1,
        min_contexts_per_source_file=1,
    )

    assert len(contexts) == num_docs
    # one delete per doc
    assert len(cap_client.delete_calls) == num_docs
