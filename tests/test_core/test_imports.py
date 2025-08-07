import pytest


def test_metrics_imports():
    """Test that all metrics can be imported."""
    from deepeval.metrics import (
        # Base classes
        BaseMetric,
        BaseConversationalMetric,
        BaseMultimodalMetric,
        BaseArenaMetric,
        # Core metrics
        GEval,
        ArenaGEval,
        ConversationalGEval,
        DAGMetric,
        # RAG metrics
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        # MCP metrics
        MCPArgsCorrectnessMetric,
        MCPTaskCompletionMetric,
        MCPToolCorrectnessMetric,
        # Other metrics
        HallucinationMetric,
        SummarizationMetric,
        JsonCorrectnessMetric,
        PromptAlignmentMetric,
        # Safety and compliance metrics
        BiasMetric,
        ToxicityMetric,
        PIILeakageMetric,
        NonAdviceMetric,
        MisuseMetric,
        RoleViolationMetric,
        # Agentic metrics
        ToolCorrectnessMetric,
        TaskCompletionMetric,
        ArgumentCorrectnessMetric,
        # Conversational metrics
        TurnRelevancyMetric,
        ConversationCompletenessMetric,
        KnowledgeRetentionMetric,
        RoleAdherenceMetric,
        # Multimodal metrics
        TextToImageMetric,
        ImageEditingMetric,
        ImageCoherenceMetric,
        ImageHelpfulnessMetric,
        ImageReferenceMetric,
        MultimodalContextualRecallMetric,
        MultimodalContextualRelevancyMetric,
        MultimodalContextualPrecisionMetric,
        MultimodalAnswerRelevancyMetric,
        MultimodalFaithfulnessMetric,
        MultimodalToolCorrectnessMetric,
        MultimodalGEval,
    )

    # Verify all imports are not None
    all_metrics = [
        BaseMetric,
        BaseConversationalMetric,
        BaseMultimodalMetric,
        BaseArenaMetric,
        GEval,
        ArenaGEval,
        ConversationalGEval,
        DAGMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        MCPArgsCorrectnessMetric,
        MCPTaskCompletionMetric,
        MCPToolCorrectnessMetric,
        HallucinationMetric,
        BiasMetric,
        ToxicityMetric,
        SummarizationMetric,
        PIILeakageMetric,
        NonAdviceMetric,
        MisuseMetric,
        RoleViolationMetric,
        RoleAdherenceMetric,
        ToolCorrectnessMetric,
        JsonCorrectnessMetric,
        PromptAlignmentMetric,
        TaskCompletionMetric,
        ArgumentCorrectnessMetric,
        KnowledgeRetentionMetric,
        TurnRelevancyMetric,
        ConversationCompletenessMetric,
        TextToImageMetric,
        ImageEditingMetric,
        ImageCoherenceMetric,
        ImageHelpfulnessMetric,
        ImageReferenceMetric,
        MultimodalContextualRecallMetric,
        MultimodalContextualRelevancyMetric,
        MultimodalContextualPrecisionMetric,
        MultimodalAnswerRelevancyMetric,
        MultimodalFaithfulnessMetric,
        MultimodalToolCorrectnessMetric,
        MultimodalGEval,
    ]

    assert True


def test_core_modules_import():
    """Test that core modules can be imported."""
    import deepeval
    import deepeval.metrics
    import deepeval.test_case
    import deepeval.test_run
    import deepeval.evaluate
    import deepeval.dataset
    import deepeval.synthesizer
    import deepeval.tracing
    import deepeval.models
    import deepeval.prompt
    import deepeval.feedback
    import deepeval.confident
    import deepeval.scorer
    import deepeval.simulator
    import deepeval.plugins
    import deepeval.openai
    import deepeval.openai_agents
    import deepeval.integrations
    import deepeval.benchmarks
    import deepeval.cli

    # Verify modules exist
    assert deepeval is not None
    assert deepeval.metrics is not None
    assert deepeval.test_case is not None
    assert deepeval.test_run is not None
    assert deepeval.evaluate is not None
    assert deepeval.dataset is not None
    assert deepeval.synthesizer is not None
    assert deepeval.tracing is not None
    assert deepeval.models is not None
    assert deepeval.prompt is not None
    assert deepeval.feedback is not None
    assert deepeval.confident is not None
    assert deepeval.scorer is not None
    assert deepeval.simulator is not None
    assert deepeval.plugins is not None
    assert deepeval.openai is not None
    assert deepeval.openai_agents is not None
    assert deepeval.integrations is not None
    assert deepeval.benchmarks is not None
    assert deepeval.cli is not None


# from .llm_test_case import (
#     LLMTestCase,
#     LLMTestCaseParams,
#     ToolCall,
#     ToolCallParams,
# )
# from .conversational_test_case import (
#     ConversationalTestCase,
#     Turn,
#     TurnParams,
#     MCPMetaData,
#     MCPPromptCall,
#     MCPResourceCall,
#     MCPToolCall,
# )
# from .mllm_test_case import MLLMTestCase, MLLMTestCaseParams, MLLMImage
# from .arena_test_case import ArenaTestCase


def test_test_case_imports():
    """Test that test case classes can be imported."""
    from deepeval.test_case import (
        LLMTestCase,
        ConversationalTestCase,
        ArenaTestCase,
        MLLMTestCase,
        Turn,
        MLLMTestCaseParams,
        MLLMImage,
        ToolCall,
        ToolCallParams,
        TurnParams,
        LLMTestCaseParams,
        MCPMetaData,
        MCPPromptCall,
        MCPResourceCall,
        MCPToolCall,
    )

    assert LLMTestCase is not None
    assert ConversationalTestCase is not None
    assert ArenaTestCase is not None
    assert MLLMTestCase is not None
    assert Turn is not None
    assert MLLMTestCaseParams is not None
    assert MLLMImage is not None
    assert ToolCall is not None
    assert ToolCallParams is not None
    assert TurnParams is not None
    assert LLMTestCaseParams is not None
    assert MCPMetaData is not None
    assert MCPPromptCall is not None
    assert MCPResourceCall is not None
    assert MCPToolCall is not None


def test_evaluate_imports():
    """Test that evaluation functions can be imported."""
    from deepeval import evaluate, assert_test, compare

    assert evaluate is not None
    assert assert_test is not None
    assert compare is not None

    from deepeval.evaluate.configs import (
        AsyncConfig,
        DisplayConfig,
        CacheConfig,
        ErrorConfig,
    )

    assert AsyncConfig is not None
    assert DisplayConfig is not None
    assert CacheConfig is not None
    assert ErrorConfig is not None


def test_dataset_imports():
    """Test that dataset classes can be imported."""
    from deepeval.dataset import (
        EvaluationDataset,
        Golden,
        ConversationalGolden,
    )

    assert EvaluationDataset is not None
    assert Golden is not None
    assert ConversationalGolden is not None


def test_models_imports():
    """Test that model classes can be imported."""
    from deepeval.models import (
        DeepEvalBaseModel,
        DeepEvalBaseLLM,
        DeepEvalBaseMLLM,
        DeepEvalBaseEmbeddingModel,
        GPTModel,
        AzureOpenAIModel,
        LocalModel,
        OllamaModel,
        AnthropicModel,
        GeminiModel,
        AmazonBedrockModel,
        LiteLLMModel,
        KimiModel,
        GrokModel,
        DeepSeekModel,
        MultimodalOpenAIModel,
        MultimodalOllamaModel,
        MultimodalGeminiModel,
        OpenAIEmbeddingModel,
        AzureOpenAIEmbeddingModel,
        LocalEmbeddingModel,
        OllamaEmbeddingModel,
    )

    # Verify all model classes can be imported
    model_classes = [
        DeepEvalBaseModel,
        DeepEvalBaseLLM,
        DeepEvalBaseMLLM,
        DeepEvalBaseEmbeddingModel,
        GPTModel,
        AzureOpenAIModel,
        LocalModel,
        OllamaModel,
        AnthropicModel,
        GeminiModel,
        AmazonBedrockModel,
        LiteLLMModel,
        KimiModel,
        GrokModel,
        DeepSeekModel,
        MultimodalOpenAIModel,
        MultimodalOllamaModel,
        MultimodalGeminiModel,
        OpenAIEmbeddingModel,
        AzureOpenAIEmbeddingModel,
        LocalEmbeddingModel,
        OllamaEmbeddingModel,
    ]

    for model_class in model_classes:
        assert model_class is not None


def test_integrations_imports():
    """Test that integration modules can be imported."""
    import deepeval.integrations.langchain
    import deepeval.integrations.llama_index
    import deepeval.integrations.hugging_face
    import deepeval.integrations.crewai
    import deepeval.integrations.pydantic_ai

    assert deepeval.integrations.langchain is not None
    assert deepeval.integrations.llama_index is not None
    assert deepeval.integrations.hugging_face is not None
    assert deepeval.integrations.crewai is not None
    assert deepeval.integrations.pydantic_ai is not None


def test_benchmarks_imports():
    """Test that benchmark modules can be imported."""
    from deepeval.benchmarks import (
        MMLU,
        BigBenchHard,
        ARC,
        BBQ,
        DROP,
        HumanEval,
        IFEval,
        LAMBADA,
        LogiQA,
        MathQA,
        SQuAD,
        TruthfulQA,
        Winogrande,
        GSM8K,
        BoolQ,
        EquityMedQA,
    )

    assert MMLU is not None
    assert BigBenchHard is not None
    assert ARC is not None
    assert BBQ is not None
    assert DROP is not None
    assert HumanEval is not None
    assert IFEval is not None
    assert LAMBADA is not None
    assert LogiQA is not None
    assert MathQA is not None
    assert SQuAD is not None
    assert TruthfulQA is not None
    assert Winogrande is not None
    assert GSM8K is not None
    assert BoolQ is not None
    assert EquityMedQA is not None


def test_tracing_imports():
    from deepeval.tracing import (
        update_current_span,
        update_current_trace,
        LlmAttributes,
        RetrieverAttributes,
        ToolAttributes,
        AgentAttributes,
        TraceAttributes,
        BaseSpan,
        Trace,
        Feedback,
        TurnContext,
        observe,
        trace_manager,
        evaluate_thread,
        evaluate_trace,
        evaluate_span,
    )

    assert update_current_span is not None
    assert update_current_trace is not None
    assert LlmAttributes is not None
    assert RetrieverAttributes is not None
    assert ToolAttributes is not None
    assert AgentAttributes is not None
    assert TraceAttributes is not None
    assert BaseSpan is not None
    assert Trace is not None
    assert Feedback is not None
    assert TurnContext is not None
    assert observe is not None
    assert trace_manager is not None
    assert evaluate_thread is not None
    assert evaluate_trace is not None
    assert evaluate_span is not None
