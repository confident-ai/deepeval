"""
Truck Safety RAG Chatbot - Fatty Arbuckle Use Case
A simple RAG chatbot using DeepEval's @observe decorator with tags
for tracking truck failure cases in fleet management.
"""

import os
from openai import OpenAI
from deepeval.tracing import observe, update_current_span, update_current_trace, update_retriever_span
from deepeval.test_case import LLMTestCase

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated knowledge base of truck failuses
TRUCK_CASES_DB = [
    {
        "id": "ENG-2019-073",
        "content": "Case ENG-2019-073: Freightliner Cascadia VIN 3AKJHHDR8KSLA5847 experienced catastrophic engine failure at 287,450 miles. Turbocharger bearing seized due to oil starvation, causing metal debris throughout engine. Root cause: clogged oil pickup screen from extended oil change intervals. Total loss, engine replacement required.",
        "category": "engine_failure",
        "truck": "Freightliner Cascadia"
    },
    {
        "id": "BRK-2021-022",
        "content": "Case BRK-2021-022: Kenworth T680 fleet (12 units) experienced premature brake pad wear at 18,000 miles vs expected 45,000 miles. Investigation revealed incorrect brake chamber pushrod adjustment causing constant drag. Fleet-wide inspection and adjustment performed, resolved issue.",
        "category": "brake_system",
        "truck": "Kenworth T680"
    },
    {
        "id": "TRANS-2023-014",
        "content": "Case TRANS-2023-014: Multiple Volvo VNL 760 units from 2022-2023 production runs reported transmission slipping in 9th-12th gears under load. Software calibration issue identified affecting shift points. OTA update released, affected approximately 0.4% of units manufactured before March 2023.",
        "category": "transmission_failure",
        "truck": "Volvo VNL 760"
    },
    {
        "id": "COOL-2020-005",
        "content": "Case COOL-2020-005: Peterbilt 579 experienced repeated coolant overheating within first 15 minutes of operation in ambient temps above 95°F. Radiator airflow restriction from debris buildup and faulty fan clutch identified. Cleaned radiator, replaced fan clutch, added pre-trip inspection protocol.",
        "category": "cooling_system",
        "truck": "Peterbilt 579"
    },
    {
        "id": "FUEL-2022-031",
        "content": "Case FUEL-2022-031: Fleet analysis found approximately 18% lower fuel efficiency in International LT units when operating with biodiesel blend B20 vs B5 in winter conditions. Cold weather fuel gelling and injector coking identified. Switched to B5 blend for temperatures below 32°F.",
        "category": "fuel_system",
        "truck": "International LT"
    },
    {
        "id": "SUSP-2020-009",
        "content": "Case SUSP-2020-009: After 6 months of heavy haul operations, Mack Anthem developed severe cab vibration and uneven tire wear. Air suspension height sensor failure caused improper load distribution. Sensors replaced, alignment corrected, vibration eliminated.",
        "category": "suspension",
        "truck": "Mack Anthem"
    },
    {
        "id": "ELEC-2022-004",
        "content": "Case ELEC-2022-004: Western Star 49X experienced complete electrical system failure at Week 10 of operation. Battery drain traced to parasitic draw from aftermarket APU installation. APU wiring corrected, battery replaced, no recurrence over 6 months.",
        "category": "electrical",
        "truck": "Western Star 49X"
    },
    {
        "id": "DEF-2021-013",
        "content": "Case DEF-2021-013: Seven Freightliner Cascadia units from 2021 fleet developed DEF system crystallization within 2-4 weeks of cold weather operation. DEF quality issue from contaminated batch identified. Tanks flushed, injectors cleaned, switched DEF supplier, triggered quality control review.",
        "category": "emissions_system",
        "truck": "Freightliner Cascadia"
    }
]


@observe(type="retriever", name="Truck Case Retriever", embedder="keyword-matcher", metric_collection="RAG Collection")
def retrieve_relevant_cases(query: str, top_k: int = 3) -> list[str]:
    """
    Simple keyword-based retriever for truck failure cases.
    In production, replace with semantic search using embeddings.
    """
    query_lower = query.lower()
    
    # Score each case based on keyword overlap
    scored_cases = []
    for case in TRUCK_CASES_DB:
        score = 0
        content_lower = case["content"].lower()
        
        # Check for truck name (high weight)
        if case["truck"].lower() in query_lower:
            score += 10
        
        # Check for category keywords
        if case["category"] in query_lower:
            score += 5
        
        # Check for general keyword overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        score += overlap
        
        if score > 0:
            scored_cases.append((score, case))
    
    # Sort by score and get top_k
    scored_cases.sort(reverse=True, key=lambda x: x[0])
    retrieved_docs = [case["content"] for _, case in scored_cases[:top_k]]
    
    # Update span with retrieval information
    update_current_span(
        input=query,
        retrieval_context=retrieved_docs
    )
    
    return retrieved_docs


@observe(type="llm", model="gpt-4o")
def generate_response(query: str, context: list[str]) -> str:
    """
    Generate response using retrieved context and OpenAI for truck maintenance queries.
    """
    # Build prompt with context
    context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
    
    prompt = f"""You are a fleet maintenance expert assistant. Based on the following truck failure cases, answer the user's question.

Retrieved Cases:
{context_str}

User Question: {query}

Provide a clear, evidence-based answer citing specific case IDs when relevant. Focus on maintenance implications and operational impact."""

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fleet maintenance expert specializing in truck failure analysis and preventive maintenance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    generated_text = response.choices[0].message.content
    
    # Update span
    update_current_span(
        input=prompt,
        output=generated_text
    )
    
    return generated_text


@observe(type="agent", name="Truck Safety RAG Chatbot")
def rag_chatbot(user_query: str) -> str:
    """
    Main RAG chatbot function for truck maintenance queries.
    Uses observe tags to track the Fatty Arbuckle truck failure use case.
    """
    # Update trace with tags specific to truck failure use case
    update_current_trace(
        name="Truck Maintenance Query",
        tags=["truck-safety", "fleet-management", "Fatty-Arbuckle-use-case", "maintenance-events"],
        metadata={
            "use_case": "truck_failure_analysis",
            "domain": "fleet_maintenance",
            "scenario": "Fatty Arbuckle"
        },
        user_id="fleet_manager_001",
        thread_id=f"thread_{hash(user_query) % 10000}"
    )
    
    # Step 1: Retrieve relevant cases
    retrieved_contexts = retrieve_relevant_cases(user_query, top_k=3)
    
    # Step 2: Generate response
    response = generate_response(user_query, retrieved_contexts)
    
    # Update agent span with test case for evaluation
    update_current_span(
        input=user_query,
        output=response,
        test_case=LLMTestCase(
            input=user_query,
            actual_output=response,
            retrieval_context=retrieved_contexts
        )
    )
    
    return response


def interactive_mode():
    """Run the chatbot in interactive mode."""
    print("=" * 80)
    print("TRUCK MAINTENANCE RAG CHATBOT - Fatty Arbuckle Use Case")
    print("=" * 80)
    print("\nAsk questions about truck failure cases. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n" + "=" * 80)
        response = rag_chatbot(user_input)
        print(f"\nResponse:\n{response}")
        print("=" * 80)


def demo_mode():
    """Run the chatbot with predefined demo queries."""
    test_queries = [
        "Are there any engine failure cases with Freightliner Cascadia?",
        "What do we know about brake system issues with Kenworth T680?",
        "Has Volvo VNL 760 had any transmission failures?",
        "Tell me about cooling system problems with Peterbilt 579",
        "What fuel efficiency issues affect International LT?"
    ]
    
    print("=" * 80)
    print("TRUCK MAINTENANCE RAG CHATBOT - Fatty Arbuckle Use Case (DEMO MODE)")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        response = rag_chatbot(query)
        print(f"\nResponse:\n{response}")
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demo_mode()
