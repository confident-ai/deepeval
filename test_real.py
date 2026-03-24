"""
Comprehensive end-to-end test of the GoodMem + DeepEval integration.

Uses a GoodMem space pre-loaded with SQuAD 2.0 articles covering:
  - Energy (physics)
  - American Idol
  - FBI history
  - Greek diaspora
  - Universal Studios

Evaluates a full RAG pipeline: retrieve from GoodMem → generate with OpenAI → score
with multiple DeepEval metrics across answerable and unanswerable queries.

Required env vars:
  GOODMEM_BASE_URL, GOODMEM_API_KEY, GOODMEM_SPACE_ID, OPENAI_API_KEY
"""

import os

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.integrations.goodmem import GoodMemConfig, GoodMemChunk, GoodMemRetriever
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
retriever = GoodMemRetriever(
    GoodMemConfig(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        space_id=os.environ["GOODMEM_SPACE_ID"],
        top_k=3,
    )
)
client = OpenAI()
GENERATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "Answer the question accurately based only on the provided context. "
    "If the context doesn't contain enough information, say so."
)

# ---------------------------------------------------------------------------
# 2. Test queries — one per SQuAD article, plus an unanswerable query
# ---------------------------------------------------------------------------
test_queries = [
    # --- Energy (physics) ---
    {
        "query": "What are the main forms of energy in physics?",
        "expected": (
            "Common energy forms include kinetic energy of a moving object, "
            "potential energy stored by position in a force field, elastic energy, "
            "and other forms like chemical, thermal, and electromagnetic energy."
        ),
    },
    # --- American Idol ---
    {
        "query": "Who created American Idol and when did it first air?",
        "expected": (
            "American Idol was created by Simon Fuller, produced by "
            "19 Entertainment, and first aired on Fox on June 11, 2002."
        ),
    },
    # --- FBI ---
    {
        "query": "What was the FBI's role in enforcing civil rights laws?",
        "expected": (
            "The FBI is charged with the responsibility of enforcing compliance "
            "with United States Civil Rights Acts."
        ),
    },
    # --- Greeks ---
    {
        "query": "Where have Greek colonies been historically established?",
        "expected": (
            "Greek colonies and communities have been historically established "
            "on the shores of the Mediterranean Sea and Black Sea, centered "
            "around the Aegean and Ionian seas."
        ),
    },
    # --- Universal Studios ---
    {
        "query": "When did Carl Laemmle open Universal's production facility?",
        "expected": (
            "On March 15, 1915, Carl Laemmle opened the world's largest motion "
            "picture production facility, Universal City Studios."
        ),
    },
    # --- Unanswerable (no relevant content in the space) ---
    {
        "query": "What is the capital of Mongolia?",
        "expected": (
            "The context does not contain information about the capital of Mongolia."
        ),
    },
]

# ---------------------------------------------------------------------------
# 3. Demonstrate retrieve_chunks() — structured retrieval with scores/IDs
# ---------------------------------------------------------------------------
print("\n=== Structured Retrieval Demo (retrieve_chunks) ===")
demo_chunks = retriever.retrieve_chunks(test_queries[0]["query"])
print(f"\nQuery: {test_queries[0]['query']}")
for i, chunk in enumerate(demo_chunks):
    print(f"  Chunk {i + 1}: score={chunk.score:.4f}  chunk_id={chunk.chunk_id[:16]}...  memory_id={chunk.memory_id[:16]}...")
    print(f"           {chunk.content[:80]}...")

# ---------------------------------------------------------------------------
# 4. Build test cases: retrieve → generate → package
# ---------------------------------------------------------------------------
test_cases = []

for item in test_queries:
    query = item["query"]

    # Retrieve context from GoodMem (plain text for LLMTestCase)
    chunks = retriever.retrieve(query)
    print(f"\n--- Query: {query} ---")
    print(f"  Retrieved {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        preview = c[:100].replace("\n", " ")
        print(f"  Chunk {i + 1}: {preview}...")

    # Generate answer grounded in retrieved context
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{chr(10).join(chunks)}\n\nQuestion: {query}",
            },
        ],
    )
    answer = response.choices[0].message.content
    print(f"  Answer: {answer[:200]}...")

    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=item["expected"],
            retrieval_context=chunks,
        )
    )

# ---------------------------------------------------------------------------
# 5. Evaluate with multiple RAG metrics
# ---------------------------------------------------------------------------
print("\n\n=== Running DeepEval Evaluation ===\n")
metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini"),
    ContextualRelevancyMetric(model="gpt-4o-mini"),
    # FaithfulnessMetric processes all retrieval chunks per claim and can
    # exceed OpenAI timeouts on lower-tier keys. Uncomment with a higher-tier key:
    # FaithfulnessMetric(model="gpt-4o-mini"),
]
results = evaluate(
    test_cases,
    metrics,
    async_config=AsyncConfig(max_concurrent=2, throttle_value=1),
)
