# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain import hub
# import bs4

# from deepeval.tracing import Observer, TraceType, QueryAttributes
# import deepeval
# import asyncio
# import time

# deepeval.trace_langchain()

# #############################################################
# ### Setup LLM ###############################################
# #############################################################

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# # Web loader with specific parsing criteria
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# # Text splitting
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# )
# splits = text_splitter.split_documents(docs)

# # Vector store creation
# vectorstore = Chroma.from_documents(
#     documents=splits, embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()

# # Prompt loading
# prompt = hub.pull("rlm/rag-prompt")


# # Helper function for formatting docs
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# # RAG chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# #############################################################
# ### Chatbot Function with Tracing and Monitoring ###########
# #############################################################


# async def chatbot(input: str) -> str:
#     start_time = time.time()  # Record the start time
#     with Observer(trace_type=TraceType.QUERY) as query_trace:
#         output = await rag_chain.ainvoke(input)
#         completion_time = time.time() - start_time

#         # Set attributes for the trace
#         query_trace.set_attributes(QueryAttributes(input=input, output=output))

#         # Monitor the event
#         query_trace.monitor(
#             input=input,
#             response=output,
#             model="gpt-3.5-turbo-0125",
#             completion_time=completion_time,
#         )

#         return output


# #############################################################
# ### Test Chatbot Inputs #####################################
# #############################################################

# user_inputs = [
#     "What is Task Decomposition?",
#     "What is Chain of Thought",
#     "What are AI agents",
#     "What is planning",
#     "What is react framework (not the js library)",
# ]


# async def query_and_print(query: str):
#     output = await chatbot(query)
#     # print(f"Query: {query}\nResponse: {output}\n")


# async def main():
#     tasks = [query_and_print(query) for query in user_inputs]
#     await asyncio.gather(*tasks)


# if __name__ == "__main__":
#     asyncio.run(main())
