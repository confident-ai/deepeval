import os
import re
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)

import deepeval


# URL validation using regular expressions
def validate_url(url):
    url_regex = re.compile(
        r"^(?:http|ftp)s?://"  # http://, https://, ftp://, or ftps://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"  # domain...
        r"(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain name
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or IP
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(url_regex, url) is not None


# Function to perform URL summarization
def summarize_url(api_key, url):
    # Validate inputs
    if not api_key or not url:
        print("API key or URL is missing.")
        return
    if not validate_url(url):
        print("Invalid URL provided.")
        return
    # Load URL data
    try:
        if "youtube.com" in url:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        else:
            loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
        data = loader.load()
        # Initialize the ChatOpenAI module, load and run the summarize chain
        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=api_key
        )
        prompt_template = (
            "Write a summary of the following in 250-300 words:\n\n{text}\n"
        )
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["text"]
        )
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain.invoke(data)
        return summary
    except Exception as e:
        print(f"Exception occurred: {e}")


# Call the function with the environment API key and sample URL
deepeval.trace_langchain()
openai_api_key = os.getenv("OPENAI_API_KEY")
url = "https://www.confident-ai.com/"
summarize_url(openai_api_key, url)
