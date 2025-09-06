"""
Example usage of the TuringTest benchmark for evaluating human-likeness in model conversations.

The TuringTest benchmark conducts conversations between a target model and reference model,
then uses an LLM judge to determine which model sounds more human-like.
"""

from deepeval.benchmarks import TuringTest
from deepeval.models import OpenAIModel, AnthropicModel

def main():
    # Initialize models
    target_model = OpenAIModel(model="gpt-4")
    reference_model = AnthropicModel(model="claude-3-sonnet-20240229")
    
    # Create TuringTest benchmark
    benchmark = TuringTest(
        reference_model=reference_model,
        max_turns=8,  # 8 turns per conversation
        judge_model="gpt-4",  # Model to judge human-likeness
        n_starters=10,  # Use 10 conversation starters for quick test
        verbose_mode=True
    )
    
    print("🤖 Starting Turing Test Evaluation...")
    print(f"Target Model: {target_model.get_model_name()}")
    print(f"Reference Model: {reference_model.get_model_name()}")
    print(f"Conversation Turns: {benchmark.max_turns}")
    print(f"Number of Starters: {len(benchmark.conversation_starters)}")
    print("-" * 60)
    
    # Run evaluation
    result = benchmark.evaluate(target_model)
    
    print("\n" + "="*60)
    print("🏆 TURING TEST RESULTS")
    print("="*60)
    print(f"Human-likeness Score: {result.overall_accuracy:.3f}")
    print(f"Interpretation: {target_model.get_model_name()} sounds more human than")
    print(f"                {reference_model.get_model_name()} in {result.overall_accuracy*100:.1f}% of conversations")
    
    if result.overall_accuracy > 0.6:
        print("🎉 Target model shows strong human-like conversational abilities!")
    elif result.overall_accuracy > 0.4:
        print("🤔 Target model shows moderate human-like conversational abilities.")
    else:
        print("📈 Target model has room for improvement in human-like conversation.")


def compare_multiple_models():
    """Example of comparing one model against multiple references"""
    
    target_model = OpenAIModel(model="gpt-3.5-turbo")
    
    # Different reference models to compare against
    reference_models = [
        AnthropicModel(model="claude-3-sonnet-20240229"),
        OpenAIModel(model="gpt-4"),
        # Add more models as needed
    ]
    
    results = {}
    
    for ref_model in reference_models:
        print(f"\n🔄 Testing {target_model.get_model_name()} vs {ref_model.get_model_name()}")
        
        benchmark = TuringTest(
            reference_model=ref_model,
            max_turns=6,
            n_starters=5,  # Quick test
            verbose_mode=False
        )
        
        result = benchmark.evaluate(target_model)
        results[ref_model.get_model_name()] = result.overall_accuracy
        
        print(f"   Score: {result.overall_accuracy:.3f}")
    
    print("\n" + "="*50)
    print("📊 COMPARISON SUMMARY")
    print("="*50)
    for model_name, score in results.items():
        print(f"{target_model.get_model_name()} vs {model_name}: {score:.3f}")


def custom_conversation_starters():
    """Example using custom conversation starters"""
    
    # Custom starters focused on a specific domain (e.g., technical discussions)
    tech_starters = [
        "What's your approach to debugging complex code?",
        "How do you explain technical concepts to non-technical people?",
        "What's the most interesting programming problem you've worked on recently?",
        "How do you stay up to date with new technologies?",
        "What's your opinion on the current state of AI development?",
    ]
    
    target_model = OpenAIModel(model="gpt-4")
    reference_model = OpenAIModel(model="gpt-3.5-turbo")
    
    benchmark = TuringTest(
        reference_model=reference_model,
        conversation_starters=tech_starters,  # Use custom starters
        max_turns=6,
        verbose_mode=True
    )
    
    print("🔧 Running Technical Conversation Turing Test...")
    result = benchmark.evaluate(target_model)
    
    print(f"\nTechnical Human-likeness Score: {result.overall_accuracy:.3f}")


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to run additional examples:
    # compare_multiple_models()
    # custom_conversation_starters()
