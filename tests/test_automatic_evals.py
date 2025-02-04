from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.integrations import trace_llama_index, trace_langchain
from deepeval import evaluate, auto_evaluate

#######################################################
### LLamaIndex ########################################
#######################################################

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Response

trace_llama_index(auto_eval=True)

Settings.llm = OpenAI(model="gpt-4-turbo-preview")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
documents = SimpleDirectoryReader("data").load_data()
node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
index = VectorStoreIndex(nodes)


async def chatbot(input):
    query_engine = index.as_query_engine(similarity_top_k=5)
    res: Response = await query_engine.aquery(input)
    return res.response


#######################################################
### LangChain #########################################
#######################################################

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain import hub
# import bs4


# trace_langchain(auto_eval=True)

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# )
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(
#     documents=splits, embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# async def chatbot(input: str) -> str:
#     output = await rag_chain.ainvoke(input)
#     return output


#######################################################
### AutoEvaluate ######################################
#######################################################

auto_evaluate(chatbot, metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()])
