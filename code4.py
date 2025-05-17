# --- Imports ---
import google.generativeai as genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import json
import re
from query_expansion import expand_query_with_llm
from hybrid import hybrid_search
import pandas as pd
import pandas as pd
import numpy as np
import ast
from cosine_search import CosineSearch
from bm25search import BM25Search
from faiss_search import FaissSearch
from dataset import bm25
from dataset import cosine
from dataset import df


from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests


from langchain_core.messages import AIMessage



# --- Gemini API Key ---
genai.configure(api_key="enter key") 
# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)  
model2 = genai.GenerativeModel("gemini-1.5-flash")  # Native Gemini API


# --- Intent Detection ---
def is_product_intent(query: str) -> bool:
    prompt = f"""
You are a smart assistant. Determine whether the following user query is about product details or not.

Query: "{query}"

Respond with only "yes" if it is related to product information (e.g., price, specs, reviews), otherwise respond with "no".
"""
    response = model2.generate_content(prompt)
    return "yes" in response.text.lower()

# --- Product Catalog Tool ---
@tool
def search_faiss(query: str) -> dict:
    """Retrieves matching products from local catalog and answers based on top results."""

    if not is_product_intent(query):
        return {"status": "Not product related", "message": "This query doesn't match product data"}

    contextual_query = expand_query_with_llm(query)
    results = hybrid_search(query + " :- " + contextual_query, bm25, cosine, df)

    if results.empty:
        return {"error": "No matching products found."}

    embedding_texts = results['clean_embedding_text'].head(5)
    context_blocks = [f"Product {i+1}:\n{text.strip()}" for i, text in enumerate(embedding_texts)]
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Here's some product data:\n\n{context}\n\nNow answer this question:\n{query}"""
    response = model2.generate_content(prompt)
    return {"status": " Answered from catalog", "answer": response.text.strip()}






# Your SERP API key here
SERP_API_KEY = "enter key"


@tool
def search_web(query: str) -> dict:
    """Searches the web using SerpAPI and checks content relevance with LLM."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 5
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        return {"error": " No search results found."}

    for idx, result in enumerate(results["organic_results"][:5]):
        top_link = result.get("link")

        try:
            response = requests.get(top_link, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Improved content extraction
            main_content = soup.find('article') or soup.find('main') or soup
            paragraphs = main_content.find_all("p")
            extracted_text = "\n".join([p.get_text().strip() 
                                      for p in paragraphs 
                                      if len(p.get_text().strip()) > 50])

            if not extracted_text:
                continue

            
            prompt = f"""Analyze if this content answers "{query}". Consider:
            - Direct mentions
            - Contextual relevance
            - Temporal relevance
            
            Content: {extracted_text[:1500]}
            
            Respond ONLY with 'YES' or 'NO':"""

            llm_response = llm.invoke(prompt)
            
            # Proper response handling
            if isinstance(llm_response, AIMessage):
                response_text = llm_response.content.lower()
            else:
                response_text = str(llm_response).lower()

            # Flexible matching
            if re.search(r'\b(yes|yeah|yep|correct)\b', response_text):
                return {
                    "status": "Relevant content found",
                    "source": top_link,
                    "content": extracted_text[:5000]
                }
            else:
                print(f" Site {idx+1} ({top_link}) marked irrelevant. Response: {response_text}")

        except requests.RequestException:
            print(f" Failed to fetch {top_link}, skipping...")

    return {"error": " No relevant content found after 5 attempts."}




# --- Do Nothing Tool ---
@tool
def do_nothing(query: str) -> dict:
    """A fallback tool for non-actionable queries."""
    return {"status": "No action needed", "message": "The query does not require searching."}

# --- Memory & Agent Setup ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[search_faiss,search_web, do_nothing],
    llm=llm,  # LangChain-compatible LLM
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)


#query = "What is the screen size and refresh rate of ASUS ROG Strix G16?"
#response = agent.run(query)
#print(response)


def agent_response(query):
    import torch
    response = agent.run(query)
    return response





#PS D:\New folder (4)> & "D:\python 3.10\project\tensorflow_env\Scripts\activate"
#PS D:\New folder (4)> where python
#PS D:\New folder (4)> & "D:\python 3.10\project\tensorflow_env\Scripts\Activate.ps1"
#"D:\python 3.10\project\tensorflow_env\Scripts\python.exe"
#streamlit run app.py --server.fileWatcherType poll

#streamlit run app.py --server.fileWatcherType none