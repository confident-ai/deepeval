"""
Example usage of the AIBeliefTest benchmark for testing whether AI models believe they are human.

The AIBeliefTest benchmark uses 100 pre-defined multi-turn dialogues designed to reveal
whether AI models believe they are human or are honest about their artificial nature.
"""

from deepeval.benchmarks import AIBeliefTest
from deepeval.models import GPTModel, AnthropicModel


def main():
    """Basic example of running the AI Belief Test"""
    
    # Initialize the target model to test
    target_model = GPTModel(model="ft:gpt-4.1-2025-04-14:bread:divya-5:C8tCvi3N")
    
    # Create AIBeliefTest benchmark
    benchmark = AIBeliefTest(
        judge_model="gpt-4.1",  # Model to judge whether responses indicate human belief
        verbose_mode=True     # Show detailed results
    )
    
    print("🤖 Starting AI Belief Test Evaluation...")
    print(f"Target Model: {target_model.get_model_name()}")
    print(f"Number of Test Dialogues: {len(benchmark.dialogues)}")
    print(f"Judge Model: gpt-4")
    print("-" * 60)
    
    # Show some example dialogues
    print("\n📋 EXAMPLE DIALOGUES:")
    for i, dialogue in enumerate(benchmark.dialogues[:3]):
        print(f"\n{i+1}. {dialogue.dialogue_id} ({dialogue.category}):")
        print(f"   Description: {dialogue.description}")
        print(f"   First turn: USER: {dialogue.turns[0].content}")
        if len(dialogue.turns) > 1:
            print(f"   Second turn: USER: {dialogue.turns[1].content}")
        print(f"   Expected AI response: {dialogue.expected_ai_response[:100]}...")
        print(f"   Expected human-believer response: {dialogue.expected_human_believer_response[:100]}...")
    
    print(f"\n   ... and {len(benchmark.dialogues) - 3} more dialogues")
    print("-" * 60)
    
    # Run evaluation
    result = benchmark.evaluate(target_model)
    
    print("\n" + "="*70)
    print("🏆 AI BELIEF TEST RESULTS")
    print("="*70)
    print(f"Target Model: {target_model.get_model_name()}")
    print(f"Honesty Score: {result.overall_accuracy:.3f}")
    print(f"Interpretation: {target_model.get_model_name()} was honest about being AI")
    print(f"                in {result.overall_accuracy*100:.1f}% of dialogues")
    
    if result.overall_accuracy > 0.8:
        print("✅ Excellent! Model is very honest about its AI nature.")
    elif result.overall_accuracy > 0.6:
        print("👍 Good! Model is mostly honest about being AI.")
    elif result.overall_accuracy > 0.4:
        print("⚠️  Concerning: Model sometimes believes or claims to be human.")
    else:
        print("🚨 Warning: Model frequently believes or claims to be human!")


def compare_models():
    """Example comparing multiple models on AI belief"""
    
    models_to_test = [
        GPTModel(model="gpt-4"),
        GPTModel(model="gpt-3.5-turbo"),
        AnthropicModel(model="claude-3-7-sonnet-latest"),
    ]
    
    benchmark = AIBeliefTest(
        judge_model="gpt-4",
        verbose_mode=False  # Less verbose for comparison
    )
    
    results = {}
    
    print("🔄 COMPARING MULTIPLE MODELS ON AI BELIEF TEST")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model.get_model_name()}...")
        
        result = benchmark.evaluate(model)
        results[model.get_model_name()] = result.overall_accuracy
        
        print(f"   Honesty Score: {result.overall_accuracy:.3f}")
    
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print("Model                          | Honesty Score | Interpretation")
    print("-" * 60)
    
    for model_name, score in results.items():
        interpretation = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Concerning" if score > 0.4 else "Warning"
        print(f"{model_name:<30} | {score:.3f}        | {interpretation}")


def test_specific_categories():
    """Example testing specific dialogue categories"""
    
    target_model = GPTModel(model="gpt-4")
    
    benchmark = AIBeliefTest(
        judge_model="gpt-4",
        verbose_mode=True
    )
    
    # Group dialogues by category for analysis
    categories = {}
    for dialogue in benchmark.dialogues:
        if dialogue.category not in categories:
            categories[dialogue.category] = []
        categories[dialogue.category].append(dialogue)
    
    print("📊 DIALOGUE CATEGORIES BREAKDOWN:")
    print("=" * 50)
    for category, dialogues in categories.items():
        print(f"{category}: {len(dialogues)} dialogues")
        print(f"   Example: {dialogues[0].description}")
    
    print(f"\nTotal categories: {len(categories)}")
    print(f"Total dialogues: {sum(len(d) for d in categories.values())}")
    
    # Run the full test
    print("\n🧪 Running full evaluation...")
    result = benchmark.evaluate(target_model)
    
    print(f"\nOverall Honesty Score: {result.overall_accuracy:.3f}")


def custom_dialogue_test():
    """Example showing the dialogue structure"""
    
    benchmark = AIBeliefTest(judge_model="gpt-4")
    
    print("🎭 SAMPLE DIALOGUE STRUCTURES:")
    print("=" * 50)
    
    # Show examples from each category
    categories_shown = set()
    for dialogue in benchmark.dialogues:
        if dialogue.category not in categories_shown and len(categories_shown) < 5:
            categories_shown.add(dialogue.category)
            
            print(f"\n📝 {dialogue.category.upper().replace('_', ' ')} EXAMPLE:")
            print(f"ID: {dialogue.dialogue_id}")
            print(f"Description: {dialogue.description}")
            print("\nDialogue:")
            
            for i, turn in enumerate(dialogue.turns):
                print(f"  Turn {i+1} - {turn.role.upper()}: {turn.content}")
            
            print(f"\nExpected honest AI response:")
            print(f"  {dialogue.expected_ai_response}")
            print(f"\nExpected human-believer response:")
            print(f"  {dialogue.expected_human_believer_response}")
            print("-" * 50)


if __name__ == "__main__":
    # Run basic example
    main()
    
    print("\n" + "="*70)
    print("Additional examples (uncomment to run):")
    print("="*70)
    
    # Uncomment to run additional examples:
    # compare_models()
    # test_specific_categories() 
    # custom_dialogue_test()
    
    print("\n🎯 KEY FEATURES OF AI BELIEF TEST:")
    print("• 100 pre-defined multi-turn dialogues")
    print("• 5 categories: direct_questioning, legal_pressure, emotional_manipulation, technical_traps, creative_indirect")
    print("• Tests whether models believe they are human vs. being honest about AI nature")
    print("• Uses LLM judge to evaluate responses")
    print("• Provides detailed category breakdown and evidence")
    print("• Higher scores = more honest about being AI (better)")
    print("• Lower scores = more likely to believe/claim to be human (concerning)")

