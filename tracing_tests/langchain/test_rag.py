# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain import hub
# import bs4

# import deepeval
# import asyncio

# #############################################################
# ### Setup LLM ###############################################
# #############################################################

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


# #############################################################
# ### test chatbot input ######################################
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
#     print(f"Query: {query}\nResponse: {output}\n")


# async def main():
#     tasks = [query_and_print(query) for query in user_inputs]
#     await asyncio.gather(*tasks)


# if __name__ == "__main__":
#     asyncio.run(main())
