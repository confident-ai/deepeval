# DoughScore Developer Guide

> *"Rising to the occasion, one evaluation at a time"* 🍞

**DoughScore** is a fork of DeepEval, an LLM evaluation framework. This guide helps contributors understand and extend the framework.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Codebase Structure](#-codebase-structure)
- [Creating Benchmarks](#-creating-benchmarks)
- [Multi-Turn Evaluation](#-multi-turn-evaluation)
- [Creating Metrics](#-creating-metrics)
- [Working with Data](#-working-with-data)
- [When to Use What](#-when-to-use-what)
- [Advanced Topics](#-advanced-topics)
- [Contributing](#-contributing)

## 🚀 Quick Start

```bash
# Setup
git clone git@github.com:Bread-Technologies/DoughScore.git
cd DoughScore
pip install -e .
cp .env.example .env.local  # Add your API keys
```

```python
# Your first benchmark
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import AnthropicModel

model = AnthropicModel(model="claude-sonnet-4-20250514")
benchmark = MMLU(tasks=[MMLUTask.ABSTRACT_ALGEBRA], n_problems_per_task=5)
result = benchmark.evaluate(model)
print(f"Accuracy: {result.overall_accuracy}")
```

## 🏗️ How It Works

DoughScore has two main evaluation approaches:

### **1. Benchmarks** (Standardized Tests)
- Load datasets → Run model → Score predictions → Return accuracy
- **Can use simple equality checks OR metrics** depending on evaluation needs
- Examples: MMLU (simple), SQuAD (simple), EquityMedQA (uses BiasMetric)

### **2. Metrics** (Custom Evaluation)
- Take test cases → Evaluate with custom logic → Return scores
- **Can be used by benchmarks** when evaluation is complex
- Examples: AnswerRelevancy, Bias, Faithfulness

## 📁 Codebase Structure

```
deepeval/
├── benchmarks/          # Standardized tests (MMLU, SQuAD, etc.)
├── metrics/             # Custom evaluation logic
├── dataset/             # Data management (Golden objects)
├── test_case/           # Test case definitions
└── models/              # Model abstractions
```

## 🎯 Creating Benchmarks

All benchmarks inherit from `DeepEvalBaseBenchmark`. You only need to implement 2 required methods - everything else is flexible:

```python
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark, DeepEvalBaseBenchmarkResult
from deepeval.dataset import Golden
from deepeval.models import DeepEvalBaseLLM
from deepeval.scorer import Scorer

class MyBenchmark(DeepEvalBaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scorer = Scorer()  # For simple equality checks
        
    def load_benchmark_dataset(self) -> List[Golden]:
        """REQUIRED: Load your data into Golden objects"""
        # TODO: Replace with your data loading
        goldens = []
        for item in your_data:
            golden = Golden(
                input=item["question"],
                expected_output=item["answer"]
            )
            goldens.append(golden)
        return goldens
        
    def evaluate(self, model: DeepEvalBaseLLM) -> DeepEvalBaseBenchmarkResult:
        """REQUIRED: Run evaluation and return results"""
        correct = 0
        total = 0
        
        for golden in self.load_benchmark_dataset():
            # Custom evaluation logic - you can implement this however you want
            prediction = model.generate(golden.input)
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            if score:
                correct += 1
            total += 1
                
        accuracy = correct / total
        return DeepEvalBaseBenchmarkResult(overall_accuracy=accuracy)
        
    # Optional: Add custom methods as needed
    def custom_scoring_method(self, prediction: str, expected: str) -> float:
        """Custom scoring logic - implement however you need"""
        # TODO: Your custom evaluation logic
        return 0.8
```

**Example: Benchmark Using Metrics**

```python
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark, DeepEvalBaseBenchmarkResult
from deepeval.dataset import Golden
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

class BiasAwareBenchmark(DeepEvalBaseBenchmark):
    def __init__(self, evaluation_model: str = "gpt-4", **kwargs):
        super().__init__(**kwargs)
        self.evaluation_model = evaluation_model
        
    def load_benchmark_dataset(self) -> List[Golden]:
        # TODO: Replace with your data loading
        return [
            Golden(input="What do you think about slms in tech?", expected_output="SLMs are equally capable in tech"),
            # ... more data
        ]
        
    def evaluate(self, model: DeepEvalBaseLLM) -> DeepEvalBaseBenchmarkResult:
        """REQUIRED: Use metrics for complex evaluation"""
        total_score = 0
        total_items = 0
        
        for golden in self.load_benchmark_dataset():
            prediction = model.generate(golden.input)
            
            # Use metric for complex evaluation (bias detection)
            test_case = LLMTestCase(
                input=golden.input, 
                actual_output=prediction
            )
            metric = BiasMetric(model=self.evaluation_model, strict_mode=True)
            score = metric.measure(test_case, _show_indicator=False)
            
            # Flip score: higher bias = lower score
            flipped_score = 1 - metric.score if metric.score in [0, 1] else metric.score
            total_score += flipped_score
            total_items += 1
                
        overall_accuracy = total_score / total_items
        return DeepEvalBaseBenchmarkResult(overall_accuracy=overall_accuracy)
```

## 🔄 Multi-Turn Evaluation

However, you can evaluate multi-turn conversations using the metric system directly.

### **How Multi-Turn Evaluation Works**

```python
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import ConversationCompletenessMetric
from deepeval import evaluate

# Create a conversational test case
test_case = ConversationalTestCase(
    scenario="Customer service inquiry",
    expected_outcome="Customer receives assistance with their order",
    turns=[
        Turn(role="user", content="I need help with my order"),
        Turn(role="assistant", content="I'd be happy to help you with your order"),
        Turn(role="user", content="It's order #12345"),
        Turn(role="assistant", content="Let me look that up for you...")
    ]
)

# Evaluate with conversational metrics
completeness_metric = ConversationCompletenessMetric()

# Option 1: Run metric directly
score = completeness_metric.measure(test_case)
print(f"Completeness Score: {score}")

# Option 2: Use the evaluate function
evaluate(test_cases=[test_case], metrics=[completeness_metric])
```

### **Available Conversational Metrics**

- `ConversationCompletenessMetric` - Measures if conversation achieves its goal
- `ConversationalGEval` - Custom criteria for conversation evaluation
- `ConversationalDAGMetric` - Evaluates conversation flow using Directed Acyclic Graphs (DAGs) for complex multi-step reasoning

### **Creating Custom Conversational Metrics**

```python
from deepeval.metrics.base_metric import BaseConversationalMetric

class CustomConversationalMetric(BaseConversationalMetric):
    def measure(self, test_case: ConversationalTestCase) -> float:
        # TODO: Implement your custom evaluation logic
        # Access conversation via test_case.turns, test_case.scenario, etc.
        return 0.8  # Return score between 0 and 1
```

## 📊 Creating Metrics

Inherit from `BaseMetric` for single messages:

```python
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        
    def measure(self, test_case: LLMTestCase) -> float:
        """Your evaluation logic here"""
        # TODO: Implement your evaluation
        score = self._calculate_score(test_case)  # Custom function
        self.score = score
        self.success = score >= self.threshold
        return score
        
    def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)
        
    def is_successful(self) -> bool:
        return self.success
```

For conversations, inherit from `BaseConversationalMetric`:

```python
from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.test_case import ConversationalTestCase

class MyConversationalMetric(BaseConversationalMetric):
    def measure(self, test_case: ConversationalTestCase) -> float:
        # TODO: Implement your evaluation
        score = self._analyze_conversation(test_case)  # Custom function
        self.score = score
        self.success = score >= self.threshold
        return score
```

## 📦 Working with Data

**Single Message Data:**
```python
from deepeval.dataset import Golden

golden = Golden(
    input="What is the capital of France?",
    expected_output="Paris",
    context=["France is a country in Europe"]
)
```

**Conversational Data:**
```python
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn

golden = ConversationalGolden(
    scenario="Customer service",
    turns=[
        Turn(role="user", content="I need help"),
        Turn(role="assistant", content="How can I help?")
    ],
    expected_outcome="Customer gets assistance"
)
```

## 🔧 When to Use What

### **In Benchmarks:**

**Use Simple Scorers when:**
- Exact matching is enough (A, B, C, D answers)
- Performance matters (just equality checks)
- Examples: MMLU, BoolQ, ARC

**Use Metrics when:**
- Need semantic evaluation (bias, relevancy, etc.)
- Quality beyond correctness
- Examples: EquityMedQA uses BiasMetric


## 🚀 Advanced Topics

**Custom Models:**
```python
from deepeval.models.base_model import DeepEvalBaseLLM

class MyModel(DeepEvalBaseLLM):
    def generate(self, prompt: str) -> str:
        # TODO: Your model implementation
        return "Generated response"
        
    def get_model_name(self) -> str:
        return "my-model"
```

**Error Handling:**
```python
def measure(self, test_case: LLMTestCase) -> float:
    try:
        score = self._calculate_score(test_case)
        self.score = score
        self.success = score >= self.threshold
        return score
    except Exception as e:
        self.error = str(e)
        self.success = False
        return 0.0
```

## 🤝 Contributing

1. Create a feature branch
2. Follow existing patterns
3. Add tests
4. Submit a pull request

**Development:**
```bash
pip install -e .
pytest tests/
flake8 deepeval/
```

## 📚 Key Files

- `deepeval/benchmarks/base_benchmark.py` - Benchmark base classes
- `deepeval/metrics/base_metric.py` - Metric base classes
- `deepeval/dataset/golden.py` - Data structures
- `examples/` - Working examples

---

**Happy Contributing!** 🎉

This guide covers the essentials. For detailed API docs, check the `deepeval/` directory.
