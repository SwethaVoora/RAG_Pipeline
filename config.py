'''
This file contains the configuration settings for the application. Made this file to ensure DRY. 
Because most of these variables, settings and client were used in multiple files:
(indexing_pipeline, retrieval_and_generation_pipeline and app.py)

It includes the following variables:
- OPENAI_API_KEY
- PINECONE_API_KEY
- LANGSMITH_API_KEY
- LANGSMITH_PROJECT
- LANGSMITH_TRACING_V2
- embeddings_model
- pc(pinecone client)
- index
- index_name
- logging configuration
'''

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv(dotenv_path=".env")

# Load API Keys safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")


# Configure logging
logging.basicConfig(level=logging.INFO)


# Set the logging level for httpx to WARNING to suppress INFO-level logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# If Pinecone or other libraries also log under their own logger names,
# set their level similarly. For example, if Pinecone uses "pinecone":
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)

# Defining the Embedding model to be used for embedding a user query
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=1536,
)


# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "testprojectv1"
index = pc.Index(index_name)
