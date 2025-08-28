import subprocess
import sys
import textwrap


def run_py(code: str):
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert (
        proc.returncode == 0
    ), f"Subprocess failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]




def test_metrics_lazy_import_subprocess():
    code = textwrap.dedent(
        """
        import sys
        import deepeval.metrics as mm
        print("bias_loaded", "deepeval.metrics.bias.bias" in sys.modules)
        from deepeval.metrics import GEval
        print("g_eval_loaded", "deepeval.metrics.g_eval.g_eval" in sys.modules)
        print("base_metric_loaded", "deepeval.metrics.base_metric" in sys.modules)
        print("models_base_loaded", "deepeval.models.base_model" in sys.modules)
        # Note: deepeval.metrics.utils imports from deepeval.models, which may load llms
        print("models_llms_loaded", any(k.startswith("deepeval.models.llms") for k in sys.modules))
        print("bias_loaded_after", "deepeval.metrics.bias.bias" in sys.modules)
        """
    )
    out = run_py(code)
    assert "bias_loaded False" in out
    assert "g_eval_loaded True" in out
    assert "base_metric_loaded True" in out
    assert "models_base_loaded True" in out
    # Accept either behavior depending on utils import path; we only assert bias remains lazy
    assert "bias_loaded_after False" in out


def test_public_api_models_subset():
    # Ensure a few representative symbols are importable
    from deepeval.models import (
        DeepEvalBaseLLM,
        GPTModel,
        MultimodalGeminiModel,
        OpenAIEmbeddingModel,
    )

    assert isinstance(GPTModel, type)
    assert isinstance(DeepEvalBaseLLM, type)
    assert isinstance(MultimodalGeminiModel, type)
    assert isinstance(OpenAIEmbeddingModel, type)


def test_public_api_metrics_subset():
    # Ensure a few representative symbols are importable
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BaseMetric,
        GEval,
        MultimodalGEval,
    )

    assert isinstance(GEval, type)
    assert isinstance(BaseMetric, type)
    assert isinstance(AnswerRelevancyMetric, type)
    assert isinstance(MultimodalGEval, type)

