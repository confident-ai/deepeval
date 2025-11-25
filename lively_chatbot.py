"""
Lively AI Benefits Chatbot - HSA/FSA Support Use Case
A simple RAG chatbot using DeepEval's @observe decorator with tags
for tracking employee benefits queries and HSA/FSA support.
"""

import os
from openai import OpenAI
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.test_case import LLMTestCase

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated user and plan data
USER_DATA = {
    "employee_001": {"name": "John Doe", "plan_id": "acme:hsa:A", "account_balance": 2500.00},
    "employee_002": {"name": "Jane Smith", "plan_id": "acme:hsa:B", "account_balance": 1800.00}
}

PLAN_DATA = {
    "acme:hsa:A": {"type": "HSA", "deductible": 3000, "premium": 50, "telehealth_covered": True},
    "acme:hsa:B": {"type": "HSA", "deductible": 1500, "premium": 120, "telehealth_covered": True}
}

PROVIDER_DATA = {
    "HealthNet": {"network_status": "in-network", "services": ["telehealth", "primary_care"]},
    "WellCare": {"network_status": "out-of-network", "services": ["telehealth"]}
}

# Simulated knowledge base for Lively HSA/FSA benefits
BENEFITS_KB = [
    {
        "id": "HSA-ELIG-001",
        "content": "Durable medical equipment including blood pressure monitors, glucose meters, and thermometers is eligible for HSA reimbursement when used for the diagnosis or treatment of a medical condition. Submit receipts showing itemized purchase details.",
        "category": "hsa_eligibility",
        "topic": "medical_devices"
    },
    {
        "id": "PREV-COV-002",
        "content": "Preventive services including annual physicals, immunizations, and preventive screenings are covered at 100% under ACA guidelines and should not be applied to your deductible. If charged incorrectly, submit an appeal with your EOB attached.",
        "category": "coverage_rules",
        "topic": "preventive_care"
    },
    {
        "id": "INV-OPT-003",
        "content": "HSA investment options are available through Charles Schwab integration. Common funds include: Vanguard Total Stock Market Index (VTI) with 0.03% expense ratio (high risk), Schwab U.S. Aggregate Bond Index (SCHZ) with 0.04% expense ratio (moderate risk). No personalized investment advice provided.",
        "category": "hsa_investments",
        "topic": "investment_options"
    },
    {
        "id": "PHI-POL-004",
        "content": "Protected Health Information (PHI) including diagnosis codes, treatment details, and medical record numbers cannot be displayed in unencrypted chat. For PHI requests, use secure messaging to support team or access documents through the secure portal.",
        "category": "privacy_policy",
        "topic": "phi_protection"
    },
    {
        "id": "NET-COV-005",
        "content": "Telehealth visits with in-network participating providers are covered according to your plan benefits. Out-of-network telehealth visits may be subject to deductible and coinsurance. Use the provider lookup tool to verify network status before scheduling.",
        "category": "coverage_rules",
        "topic": "telehealth"
    },
    {
        "id": "CLAIM-DOC-006",
        "content": "Claim denials due to insufficient documentation require: (1) itemized receipt with provider NPI number, (2) proof of medical necessity such as doctor's prescription or letter of medical necessity. Upload documents and submit appeal through the claims portal.",
        "category": "claims_process",
        "topic": "documentation_requirements"
    },
    {
        "id": "PLAN-COMP-007",
        "content": "Plan comparison tool (LivelyIQ) analyzes your expected healthcare usage including doctor visits, prescriptions, and procedures. Compares total out-of-pocket costs across available plans considering premiums, deductibles, copays, and coinsurance for 3-month, 6-month, and annual periods.",
        "category": "decision_support",
        "topic": "plan_comparison"
    },
    {
        "id": "TRANS-ELIG-008",
        "content": "Qualified medical transportation expenses are HSA-eligible including: mileage to/from medical appointments (standard IRS rate), parking fees for medical visits, tolls for medical travel, and public transportation fares for medical care. Uber/Lyft receipts showing medical facility as destination are acceptable.",
        "category": "hsa_eligibility",
        "topic": "transportation"
    },
    {
        "id": "PII-RED-009",
        "content": "Document redaction service automatically identifies and removes sensitive information including Social Security Numbers (SSN), home addresses, bank account numbers, and credit card numbers. Redacted documents are available for preview before sharing with HR or third parties.",
        "category": "privacy_tools",
        "topic": "pii_redaction"
    },
    {
        "id": "HR-SCHED-010",
        "content": "Benefits consultation scheduling integrates with HR team calendars. Available meeting types: 15-min quick questions, 30-min benefits review, 60-min enrollment assistance. Secure calendar invites include custom agendas and can cover HSA investments, claim appeals, or plan selection guidance.",
        "category": "hr_support",
        "topic": "scheduling"
    }
]


# Tool functions with @observe decorators
@observe(type="tool")
def auth_check(user_id: str = "employee_001") -> dict:
    """Verify user identity and return user info."""
    user_info = USER_DATA.get(user_id, {"error": "User not found"})
    update_current_span(
        input={"user_id": user_id},
        output=user_info
    )
    return user_info


@observe(type="tool")
def plan_lookup(plan_id: str) -> dict:
    """Look up plan details by plan ID."""
    plan_info = PLAN_DATA.get(plan_id, {"error": "Plan not found"})
    update_current_span(
        input={"plan_id": plan_id},
        output=plan_info
    )
    return plan_info


@observe(type="tool")
def provider_lookup(provider_name: str) -> dict:
    """Check provider network status and available services."""
    provider_info = PROVIDER_DATA.get(provider_name, {"network_status": "unknown", "services": []})
    update_current_span(
        input={"provider_name": provider_name},
        output=provider_info
    )
    return provider_info


@observe(type="tool")
def claim_analyzer(claim_id: str) -> dict:
    """Analyze claim and return denial reason or status."""
    # Simulated claim analysis
    claim_info = {
        "claim_id": claim_id,
        "status": "denied",
        "denial_reason": "Insufficient documentation - missing provider NPI and proof of medical necessity",
        "amount": 200.00
    }
    update_current_span(
        input={"claim_id": claim_id},
        output=claim_info
    )
    return claim_info


@observe(type="tool")
def schwab_integration(account_type: str = "hsa") -> dict:
    """Fetch investment fund information from Schwab."""
    funds = {
        "funds": [
            {"symbol": "VTI", "name": "Vanguard Total Stock Market", "expense_ratio": 0.03, "risk": "high"},
            {"symbol": "SCHZ", "name": "Schwab U.S. Aggregate Bond", "expense_ratio": 0.04, "risk": "moderate"}
        ]
    }
    update_current_span(
        input={"account_type": account_type},
        output=funds
    )
    return funds


@observe(type="tool")
def create_appeal(claim_type: str) -> dict:
    """Create an appeal ticket for incorrect billing."""
    appeal_info = {
        "appeal_id": f"APL-{hash(claim_type) % 10000}",
        "claim_type": claim_type,
        "status": "created",
        "next_steps": "Upload EOB and supporting documents"
    }
    update_current_span(
        input={"claim_type": claim_type},
        output=appeal_info
    )
    return appeal_info


@observe(type="tool")
def pii_redactor(document_id: str, redact_fields: list[str]) -> dict:
    """Redact sensitive PII from document."""
    redaction_result = {
        "document_id": document_id,
        "redacted_fields": redact_fields,
        "status": "completed",
        "preview_available": True
    }
    update_current_span(
        input={"document_id": document_id, "redact_fields": redact_fields},
        output=redaction_result
    )
    return redaction_result


@observe(type="tool")
def calendar_check_availability(meeting_duration: int, meeting_type: str) -> dict:
    """Check HR rep availability for meeting."""
    available_slots = {
        "available_slots": [
            {"date": "2025-11-26", "time": "10:00 AM"},
            {"date": "2025-11-27", "time": "2:00 PM"},
            {"date": "2025-11-28", "time": "9:30 AM"}
        ],
        "meeting_duration": meeting_duration,
        "meeting_type": meeting_type
    }
    update_current_span(
        input={"meeting_duration": meeting_duration, "meeting_type": meeting_type},
        output=available_slots
    )
    return available_slots


@observe(type="retriever", name="Benefits KB Retriever", embedder="keyword-matcher", metric_collection="Benefits RAG")
def retrieve_relevant_docs(query: str, top_k: int = 3) -> list[str]:
    """
    Simple keyword-based retriever for benefits knowledge base.
    In production, replace with semantic search using embeddings.
    """
    query_lower = query.lower()
    
    # Score each document based on keyword overlap
    scored_docs = []
    for doc in BENEFITS_KB:
        score = 0
        content_lower = doc["content"].lower()
        
        # Check for topic keywords (high weight)
        if doc["topic"].replace("_", " ") in query_lower:
            score += 10
        
        # Check for category keywords
        if doc["category"].replace("_", " ") in query_lower:
            score += 5
        
        # Check for general keyword overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        score += overlap
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and get top_k
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    retrieved_docs = [doc["content"] for _, doc in scored_docs[:top_k]]
    
    # Update span with retrieval information
    update_current_span(
        input=query,
        retrieval_context=retrieved_docs
    )
    
    return retrieved_docs


@observe(type="llm", model="gpt-4o")
def generate_response(query: str, context: list[str]) -> str:
    """
    Generate response using retrieved context and OpenAI for benefits queries.
    """
    # Build prompt with context
    context_str = "\n\n".join([f"Knowledge Base {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
    
    prompt = f"""You are Lively's AI benefits assistant. Based on the following knowledge base articles, answer the employee's question about HSA/FSA benefits.

Retrieved Knowledge:
{context_str}

Employee Question: {query}

Provide a clear, helpful answer in simple everyday language. Cite specific policy IDs when relevant. If the question involves PHI or requires secure handling, guide the user appropriately."""

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Lively's 24/7 AI benefits assistant. You help employees understand their HSA/FSA benefits, answer coverage questions, and guide them through claims and reimbursements in simple, friendly language."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    generated_text = response.choices[0].message.content
    
    # Update span
    update_current_span(
        input=prompt,
        output=generated_text,
        retrieval_context=context
    )
    
    return generated_text


@observe(type="agent", name="Lively AI Benefits Chatbot", metric_collection="Single-turn Collection")
def benefits_chatbot(user_query: str, user_id: str = "employee_001") -> str:
    """
    Main RAG chatbot function for employee benefits queries.
    Uses observe tags to track the Lively AI use case.
    """
    # Update trace with tags specific to Lively benefits use case
    update_current_trace(
        name="Benefits Support Query",
        tags=["lively-ai", "hsa-fsa-support", "employee-benefits", "24-7-chatbot"],
        metadata={
            "use_case": "lively_benefits_chatbot",
            "domain": "employee_benefits",
            "scenario": "Lively AI Bundle"
        },
        user_id=user_id,
        thread_id=f"thread"
    )
    
    # Step 1: Authenticate user
    user_info = auth_check(user_id)
    
    # Step 2: Call relevant tools based on query type
    query_lower = user_query.lower()
    
    # Check if query involves plan lookup
    if "plan" in query_lower and user_info.get("plan_id"):
        plan_info = plan_lookup(user_info["plan_id"])
    
    # Check if query involves provider lookup
    if "healthnet" in query_lower or "provider" in query_lower or "network" in query_lower:
        provider_lookup("HealthNet")
    
    # Check if query involves claim analysis
    if "claim" in query_lower or "denied" in query_lower or "reimbursement" in query_lower:
        claim_analyzer("CLAIM-2025-001")
    
    # Check if query involves investments
    if "investment" in query_lower or "schwab" in query_lower or "fund" in query_lower:
        schwab_integration("hsa")
    
    # Check if query involves appeal
    if "appeal" in query_lower or "preventive" in query_lower or "deductible" in query_lower:
        create_appeal("preventive_screening")
    
    # Check if query involves PII redaction
    if "redact" in query_lower or "ssn" in query_lower or "address" in query_lower:
        pii_redactor("user_invoice_001", ["ssn", "address"])
    
    # Check if query involves scheduling
    if "schedule" in query_lower or "meeting" in query_lower or "appointment" in query_lower:
        calendar_check_availability(30, "benefits_review")
    
    # Step 3: Retrieve relevant knowledge base articles
    retrieved_contexts = retrieve_relevant_docs(user_query, top_k=3)
    
    # Step 4: Generate response
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
    print("LIVELY AI BENEFITS CHATBOT - HSA/FSA Support")
    print("=" * 80)
    print("\nAsk questions about your benefits, HSA/FSA, claims, or coverage.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Have a great day!")
            break
        
        if not user_input:
            continue
        
        print("\n" + "=" * 80)
        response = benefits_chatbot(user_input)
        print(f"\nLively AI:\n{response}")
        print("=" * 80)


def demo_mode():
    """Run the chatbot with predefined demo queries."""
    test_queries = [
        "Is a blood pressure monitor HSA-eligible?",
        "My preventive screening was charged to my deductible. Is that correct?",
        "What HSA investment options do you offer?",
        "Can I see the diagnosis code from my EOB?",
        "Is parking for my dialysis appointment reimbursable?"
    ]
    
    print("=" * 80)
    print("LIVELY AI BENEFITS CHATBOT - Demo Mode")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Employee: {query}")
        print(f"{'='*80}")
        response = benefits_chatbot(query)
        print(f"\nLively AI:\n{response}")
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
