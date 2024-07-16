from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool

#############################################################
#### Step 1: Define the base LLM and the embedding model
#############################################################

# LLM
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True)

# Embedding Model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)

# Set Llamaindex Configs
Settings.llm = llm
Settings.embed_model = embed_model

#############################################################
#### Step 2: We leverage the indexing and retrieval
# functionalities of LlamaIndex to define individual query
# engines for our documents.
#############################################################

# Building Indexes for each of the Documents
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./tracing_data/storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./tracing_data/storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
    print(
        "Index was already created. We just loaded it from the local storage."
    )

except:

    index_loaded = False
    print("Index is not present. We need it to create it again.")

if not index_loaded:

    print("Creating Index..")

    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["./tracing_data/lyft.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./tracing_data/uber.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(
        persist_dir="./tracing_data/storage/lyft"
    )
    uber_index.storage_context.persist(
        persist_dir="./tracing_data/storage/uber"
    )

    index_loaded = True

# Creating Query engines on top of the indexes
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

print("LlamaIndex Query Engines created successfully.")

#############################################################
#### Step 3: We now use the LlamaIndex QueryEngineTool
# abstraction to transform these query engines into Tools,
# which would later be provided to the LLM.
#############################################################

# creating tools for each of our query engines
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

#############################################################
#### Step 4: We convert the LlamaIndex Tools into a format
# compatible with Langchain Agents.
#############################################################

# convert to langchain format
llamaindex_to_langchain_converted_tools = [
    t.to_langchain_tool() for t in query_engine_tools
]

# Another Langchain Tool
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=search.run,
    description="Use for when you need to perform an internet search to find information that another tool can not provide.",
)
langchain_tools = [duckduckgo_tool]

# Combine to create final list of tools
tools = llamaindex_to_langchain_converted_tools + langchain_tools

#############################################################
#### Step 5: We’ll initialize Langchain’s latest Tool
# Calling Agent.
#############################################################

system_context = "You are a stock market expert.\
You will answer questions about Uber and Lyft companies as in the persona of a veteran stock market investor."

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_context,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(
    llm,
    tools,
    prompt,
)

#############################################################
#### Step 6: Next, we’ll put the agent to the test
# with our queries.
#############################################################

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

#############################################################
#### Step 7: Tracking responses
#############################################################
import deepeval
from deepeval.tracing import Tracer, TraceType, QueryAttributes


def ask_llm(question):
    with Tracer(TraceType.QUERY) as tracer:
        response = agent_executor.invoke({"input": question})
        tracer.set_attributes(
            QueryAttributes(input=question, output=response["output"])
        )
        tracer.track(
            event_name="Hybrid Integrations",
            input=question,
            response=response["output"],
            model="gpt-4-1106-preview",
        )


deepeval.trace_langchain()
deepeval.trace_llama_index()

question = "What was Lyft's revenue growth in 2021?"
ask_llm(question)
