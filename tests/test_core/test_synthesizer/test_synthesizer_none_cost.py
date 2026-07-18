"""Regression tests for #2884: Synthesizer silent empty goldens when cost is None."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.constants import DEEPSEEK_MODELS_DATA
from deepeval.synthesizer.chunking.context_generator import (
    ContextGenerator,
    ContextScore,
)


class _StubLLM(DeepEvalBaseLLM):
    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        return ""

    async def a_generate(self, *args, **kwargs) -> str:
        return ""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


class _NativeModelWithUnknownCost:
    def generate(self, prompt, schema=None):
        return (
            ContextScore(
                clarity=1.0,
                depth=1.0,
                structure=1.0,
                relevance=1.0,
            ),
            None,
        )

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def _patch_langchain(monkeypatch):
    class _FakeTextLoader:
        def __init__(self, path, encoding=None, autodetect_encoding=True):
            self._path = path

        def load(self):
            class _Doc:
                page_content = "chunk one\nchunk two\nchunk three"

            return [_Doc()]

        async def aload(self):
            return self.load()

    class _FakeSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            pass

        def split_documents(self, docs):
            class _Doc:
                def __init__(self, txt):
                    self.page_content = txt

            return [_Doc(f"c{j}") for j in range(5)]

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


class _StubEmbedder:
    def embed_texts(self, xs):
        return [[0.0, 0.0] for _ in xs]

    async def a_embed_texts(self, xs):
        return [[0.0, 0.0] for _ in xs]

    def embed_text(self, x):
        return [0.0, 0.0]

    async def a_embed_text(self, x):
        return [0.0, 0.0]


class _CapturingCollection:
    def __init__(self, name, count_value=5):
        self.name = name
        self._count_value = count_value

    def count(self):
        return self._count_value

    def get(self, ids):
        return {"documents": [f"D{i}" for i in ids]}

    def query(self, _embedding, n_results):
        docs = [["q"] + [f"n{j}" for j in range(n_results - 1)]]
        dists = [[0.0] + [0.1] * (n_results - 1)]
        return {"documents": docs, "distances": dists}


class _CapturingClient:
    def __init__(self, count_value=5):
        self.collections = {}
        self._count_value = count_value

    def get_collection(self, name):
        return self.collections[name]

    def create_collection(self, name):
        collection = _CapturingCollection(name, self._count_value)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        self.collections.pop(name, None)


class _CapturingChromaMod:
    def __init__(self, client=None):
        self.client = client or _CapturingClient()

    def PersistentClient(self, path, **kwargs):
        return self.client


class TestDeepSeekV4ModelRegistration:
    def test_v4_models_have_cache_miss_pricing(self):
        flash = DEEPSEEK_MODELS_DATA["deepseek-v4-flash"]
        pro = DEEPSEEK_MODELS_DATA["deepseek-v4-pro"]

        assert flash.input_price == pytest.approx(0.14 / 1e6)
        assert flash.output_price == pytest.approx(0.28 / 1e6)
        assert pro.input_price == pytest.approx(0.435 / 1e6)
        assert pro.output_price == pytest.approx(0.87 / 1e6)


class TestEvaluateChunkNoneCost:
    def test_sync_evaluate_chunk_skips_none_cost(self):
        generator = ContextGenerator(
            document_paths=["dummy.txt"],
            embedder=_StubEmbedder(),
            model=_StubLLM(),
        )
        generator.using_native_model = True
        generator.model = _NativeModelWithUnknownCost()

        score = generator.evaluate_chunk("sample chunk")

        assert score == 1.0
        assert generator.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_async_evaluate_chunk_skips_none_cost(self):
        generator = ContextGenerator(
            document_paths=["dummy.txt"],
            embedder=_StubEmbedder(),
            model=_StubLLM(),
        )
        generator.using_native_model = True
        generator.model = _NativeModelWithUnknownCost()

        score = await generator.a_evaluate_chunk("sample chunk")

        assert score == 1.0
        assert generator.total_cost == 0.0


class TestContextGenerationErrors:
    def test_all_document_failures_raise_deep_eval_error(
        self, monkeypatch, tmp_path
    ):
        import sys

        _patch_langchain(monkeypatch)

        doc_path = tmp_path / "doc.md"
        doc_path.write_text("hello", encoding="utf-8")

        cap_chroma = _CapturingChromaMod()
        monkeypatch.setattr(
            "deepeval.synthesizer.chunking.doc_chunker._chroma_mod",
            cap_chroma,
            raising=True,
        )
        monkeypatch.setattr(
            "deepeval.synthesizer.chunking.context_generator.get_chromadb",
            lambda: cap_chroma,
            raising=True,
        )
        sys.modules["chromadb.config"] = SimpleNamespace(
            Settings=lambda **kwargs: SimpleNamespace(**kwargs)
        )

        generator = ContextGenerator(
            document_paths=[str(doc_path)],
            embedder=_StubEmbedder(),
            model=_StubLLM(),
            chunk_size=50,
            chunk_overlap=0,
            max_retries=1,
            filter_threshold=0.0,
            similarity_threshold=0.0,
        )
        generator.using_native_model = True
        generator.model = _NativeModelWithUnknownCost()

        def _boom(_chunk):
            raise TypeError("unsupported operand type(s) for +=: 'float' and 'NoneType'")

        monkeypatch.setattr(generator, "evaluate_chunk", _boom)

        with pytest.raises(DeepEvalError, match="Context generation failed for all"):
            generator.generate_contexts(
                max_contexts_per_source_file=1,
                min_contexts_per_source_file=1,
            )


class TestSynthesizerEmptyContexts:
    def test_generate_goldens_from_docs_raises_on_empty_contexts(self):
        from deepeval.synthesizer.synthesizer import Synthesizer

        fake_model = MagicMock()
        fake_model.get_model_name.return_value = "fake-model"

        with patch(
            "deepeval.synthesizer.synthesizer.initialize_model",
            return_value=(fake_model, True),
        ), patch(
            "deepeval.synthesizer.config.initialize_model",
            return_value=(fake_model, True),
        ), patch(
            "deepeval.synthesizer.config.initialize_embedding_model",
            side_effect=lambda embedder: embedder,
        ), patch(
            "deepeval.synthesizer.synthesizer.ContextGenerator"
        ) as mock_context_generator_cls, patch(
            "deepeval.synthesizer.synthesizer.synthesizer_progress_context"
        ) as mock_progress:
            mock_progress.return_value.__enter__ = MagicMock(
                return_value=(MagicMock(), 0)
            )
            mock_progress.return_value.__exit__ = MagicMock(return_value=False)

            mock_context_generator = MagicMock()
            mock_context_generator.generate_contexts.return_value = ([], [], [])
            mock_context_generator.total_chunks = 0
            mock_context_generator_cls.return_value = mock_context_generator

            synth = Synthesizer(model=fake_model, async_mode=False)
            embedder = MagicMock()
            embedder.get_model_name.return_value = "stub-embedder"

            from deepeval.synthesizer.config import ContextConstructionConfig

            with pytest.raises(DeepEvalError, match="No contexts were generated"):
                synth.generate_goldens_from_docs(
                    document_paths=["doc.txt"],
                    context_construction_config=ContextConstructionConfig(
                        embedder=embedder,
                        critic_model=fake_model,
                    ),
                )
