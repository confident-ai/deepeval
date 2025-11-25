"""
Demo script to test Lively AI chatbot with tool spans
Run this to see the complete trace with retriever, tools, and LLM spans
"""

import os
from lively_chatbot import benefits_chatbot

# Test queries that will trigger different tools
test_queries = [
    "I bought a blood pressure monitor for $55. Is this HSA-eligible?",
    "My preventive screening was applied to my deductible. Is that right?",
    "What HSA investment options are available?",
    "Can I use my HSA card for a telehealth visit with HealthNet?",
    "My $200 reimbursement for diabetes supplies was denied. Why?",
    "I need to share my invoice with HR but remove my SSN first.",
    "Can you schedule a 30-minute benefits review with an HR rep?"
]

def run_demo():
    """Run demo queries through the chatbot."""
    print("=" * 80)
    print("LIVELY AI CHATBOT - Tool Spans Demo")
    print("=" * 80)
    print("\nThis demo shows the chatbot calling various tools:")
    print("- auth_check: User authentication")
    print("- plan_lookup: Plan details lookup")
    print("- provider_lookup: Provider network status")
    print("- claim_analyzer: Claim denial analysis")
    print("- schwab_integration: Investment fund info")
    print("- create_appeal: Appeal ticket creation")
    print("- pii_redactor: PII redaction")
    print("- calendar_check_availability: Meeting scheduling")
    print("- retrieve_relevant_docs: Knowledge base retrieval")
    print("- generate_response: LLM response generation")
    print("\n" + "=" * 80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")
        
        try:
            response = benefits_chatbot(query)
            print(f"\nLively AI Response:\n{response}")
        except Exception as e:
            print(f"\nError: {e}")
        
        print(f"\n{'='*80}\n")
        
        # Add a small pause between queries
        import time
        time.sleep(1)

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    run_demo()
