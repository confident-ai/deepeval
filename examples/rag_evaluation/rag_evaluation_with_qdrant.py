# To run this example, you need to install the following dependencies:
#
# pip install datasets langchain langchain-text-splitters openai qdrant-client deepeval
#

# Set connection credentials for OpenAI, Confident AI, and Qdrant below

# Then, run the following command:
# python examples/rag_evaluation/rag_evaluation_with_qdrant.py

# You can then find results of the evaluation in the Confident AI dashboard

from tqdm.notebook import tqdm
from datasets import load_dataset
from qdrant_client import QdrantClient
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import deepeval

# Get your key from https://platform.openai.com/api-keys
OPENAI_API_KEY = "<OPENAI_API_KEY>"

# Get your Confident AI API key from https://app.confident-ai.com
CONFIDENT_AI_API_KEY = "<CONFIDENT_AI_API_KEY>"

# Get a FREE forever cluster at https://cloud.qdrant.io/
# More info: https://qdrant.tech/documentation/cloud/create-cluster/
QDRANT_URL = "<QDRANT_URL>"
QDRANT_API_KEY = "<QDRANT_API_KEY>"
COLLECTION_NAME = "qdrant-deepeval"

EVAL_SIZE = 10
RETRIEVAL_SIZE = 3

dataset = load_dataset("atitaarora/qdrant_doc", split="train")

langchain_docs = [
    LangchainDocument(
        page_content=doc["text"], metadata={"source": doc["source"]}
    )
    for doc in tqdm(dataset)
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

docs_contents, docs_metadatas = [], []

for doc in docs_processed:
    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
        docs_contents.append(doc.page_content)
        docs_metadatas.append(doc.metadata)
    else:
        print(
            "Warning: Some documents do not have 'page_content' or 'metadata' attributes."
        )

# Uses FastEmbed - https://qdrant.tech/documentation/fastembed/
# To generate embeddings for the documents
# The default model is `BAAI/bge-small-en-v1.5`
client.add(
    collection_name=COLLECTION_NAME,
    metadata=docs_metadatas,
    documents=docs_contents,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def query_with_context(query, limit):

    search_result = client.query(
        collection_name=COLLECTION_NAME, query_text=query, limit=limit
    )

    contexts = [
        "document: " + r.document + ",source: " + r.metadata["source"]
        for r in search_result
    ]
    prompt_start = """ You're assisting a user who has a question based on the documentation.
        Your goal is to provide a clear and concise response that addresses their query while referencing relevant information
        from the documentation.
        Remember to:
        Understand the user's question thoroughly.
        If the user's query is general (e.g., "hi," "good morning"),
        greet them normally and avoid using the context from the documentation.
        If the user's query is specific and related to the documentation, locate and extract the pertinent information.
        Craft a response that directly addresses the user's query and provides accurate information
        referring the relevant source and page from the 'source' field of fetched context from the documentation to support your answer.
        Use a friendly and professional tone in your response.
        If you cannot find the answer in the provided context, do not pretend to know it.
        Instead, respond with "I don't know".

        Context:\n"""

    prompt_end = f"\n\nQuestion: {query}\nAnswer:"

    prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end

    res = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )

    return (contexts, res.choices[0].text)


qdrant_qna_dataset = load_dataset("atitaarora/qdrant_doc_qna", split="train")


def create_deepeval_dataset(dataset, eval_size, retrieval_window_size):
    test_cases = []
    for i in range(eval_size):
        entry = dataset[i]
        question = entry["question"]
        answer = entry["answer"]
        context, rag_response = query_with_context(
            question, retrieval_window_size
        )
        test_case = deepeval.test_case.LLMTestCase(
            input=question,
            actual_output=rag_response,
            expected_output=answer,
            retrieval_context=context,
        )
        test_cases.append(test_case)
    return test_cases


test_cases = create_deepeval_dataset(
    qdrant_qna_dataset, EVAL_SIZE, RETRIEVAL_SIZE
)

deepeval.login_with_confident_api_key(CONFIDENT_AI_API_KEY)

deepeval.evaluate(
    test_cases=test_cases,
    metrics=[
        deepeval.metrics.AnswerRelevancyMetric(),
        deepeval.metrics.FaithfulnessMetric(),
        deepeval.metrics.ContextualPrecisionMetric(),
        deepeval.metrics.ContextualRecallMetric(),
        deepeval.metrics.ContextualRelevancyMetric(),
    ],
)
