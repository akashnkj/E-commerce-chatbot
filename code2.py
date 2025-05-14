import streamlit as st
import pandas as pd
import os
import io
import json
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
#from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv('.env')
groq_api_key = os.getenv('GROQ_API_KEY')
#cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b",
    api_key=groq_api_key
)

#llm.to_empty()

# import torch

# # Assuming your model is a PyTorch model
# model = torch.nn.Module()  # Dummy example, replace with your actual model

# # Move from "meta" to another device (e.g., CPU or GPU)
# model.to_empty()  # Use .to_empty() instead of .to(device)


# # Initialize Cohere Embeddings
# embedding_model = CohereEmbeddings(
#     model="embed-english-v3.0",
#     cohere_api_key=cohere_api_key
# )

# Load CSV data into a buffer
buffer = io.StringIO()
with open('Flipkart_laptops.csv', 'r') as file:
    buffer.write(file.read())

buffer.seek(0)  # Reset buffer position

# Read laptop data from buffer
laptops = pd.read_csv(buffer)

# Convert DataFrame rows into formatted text for FAISS embedding
laptop_texts = laptops.apply(lambda row: " ".join(map(str, row)), axis=1).tolist()

# Create FAISS vector store

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(laptop_texts, embedding_model)


# Create retriever (limit results for efficiency)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define custom prompt template
template = """You are a laptop expert. Use the following context to answer the question.
Context:{context}, Question: {question}"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define retrieval tool
retrieval_tool = Tool(
    name="LaptopRecommendationTool",
    func=lambda query: retriever.get_relevant_documents(query),
    description="Provides relevant laptops based on specifications."
)

# Initialize AI agent with conversational memory
agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Streamlit UI
st.title("💻 AI Laptop Recommendation System")
st.write("Ask about laptop specifications, battery life, or performance!")

# User query input
user_query = st.text_input("Enter your laptop-related question:", "")

if st.button("Get Recommendation"):
    if user_query:
        response = agent.run(user_query)
        st.write("# AI Recommendation")
        st.write(response)

        chat_history = memory.load_memory_variables({})
        st.write("# Conversation History")
        st.json(chat_history)

        def save_conversation(history, filename="conversation_history.json"):
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []

            existing_data.extend(history)

            with open(filename, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, indent=4)

        conversation_history = chat_history["chat_history"]
        history_list = [
            {"query": message.content, "response": conversation_history[idx + 1].content}
            for idx, message in enumerate(conversation_history[:-1]) if idx % 2 == 0
        ]
        save_conversation(history_list)
        st.write("Conversation saved to conversation_history.json!")
    else:
        st.warning("Please enter a question first!")
