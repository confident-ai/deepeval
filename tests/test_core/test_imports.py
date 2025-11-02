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
        DeepAcyclicGraph,
        # RAG metrics
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        # MCP metrics
        MCPUseMetric,
        MCPTaskCompletionMetric,
        MultiTurnMCPUseMetric,
        # Non-LLM metrics
        JsonCorrectnessMetric,
        ExactMatchMetric,
        PatternMatchMetric,
        # Other metrics
        HallucinationMetric,
        SummarizationMetric,
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
        GoalAccuracyMetric,
        TopicAdherenceMetric,
        PlanAdherenceMetric,
        PlanQualityMetric,
        ToolUseMetric,
        StepEfficiencyMetric,
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
        DeepAcyclicGraph,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        MCPTaskCompletionMetric,
        MCPUseMetric,
        MultiTurnMCPUseMetric,
        HallucinationMetric,
        BiasMetric,
        ExactMatchMetric,
        PatternMatchMetric,
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
        GoalAccuracyMetric,
        TopicAdherenceMetric,
        PlanAdherenceMetric,
        PlanQualityMetric,
        ToolUseMetric,
        StepEfficiencyMetric,
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

    for metric in all_metrics:
        assert metric is not None


def test_g_eval_imports():
    from deepeval.metrics.g_eval import Rubric

    assert Rubric is not None


def test_dag_imports():
    from deepeval.metrics.dag import (
        DeepAcyclicGraph,
        TaskNode,
        BinaryJudgementNode,
        NonBinaryJudgementNode,
        VerdictNode,
    )

    assert DeepAcyclicGraph is not None
    assert TaskNode is not None
    assert BinaryJudgementNode is not None
    assert NonBinaryJudgementNode is not None
    assert VerdictNode is not None


def test_conversational_dag_imports():
    from deepeval.metrics.conversational_dag import (
        ConversationalTaskNode,
        ConversationalBinaryJudgementNode,
        ConversationalNonBinaryJudgementNode,
        ConversationalVerdictNode,
    )

    assert ConversationalTaskNode is not None
    assert ConversationalBinaryJudgementNode is not None
    assert ConversationalNonBinaryJudgementNode is not None
    assert ConversationalVerdictNode is not None


def test_core_modules_import(unpatch_openai_after):
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
    import deepeval.annotation
    import deepeval.confident
    import deepeval.scorer
    import deepeval.simulator
    import deepeval.plugins
    import deepeval.openai
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
    assert deepeval.annotation is not None
    assert deepeval.confident is not None
    assert deepeval.scorer is not None
    assert deepeval.simulator is not None
    assert deepeval.plugins is not None
    assert deepeval.openai is not None
    assert deepeval.cli is not None


def test_test_case_imports():
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
        MCPServer,
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
    assert MCPServer is not None
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
        BaseSpan,
        Trace,
        observe,
        trace_manager,
        evaluate_thread,
        evaluate_trace,
        evaluate_span,
    )

    assert update_current_span is not None
    assert update_current_trace is not None
    assert BaseSpan is not None
    assert Trace is not None
    assert observe is not None
    assert trace_manager is not None
    assert evaluate_thread is not None
    assert evaluate_trace is not None
    assert evaluate_span is not None
