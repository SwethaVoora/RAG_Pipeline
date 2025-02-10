'''
- This file contains the code for the indexing pipeline only(does not include the retrieval and generation pipeline).
- The content of this file was part of the RAG_pipeline notebook file previously.
- All the methods in this file were seperate cells in the RAG_pipeline notebook.
- I just combined the indexing related methods and applicable methods into this file to ensure modularity and maintainability.
'''
import os
from dotenv import load_dotenv
from langsmith import utils
import time
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from pinecone import Pinecone,ServerlessSpec


load_dotenv(dotenv_path=".env")

# Load API Keys Safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")

# Enabling tracing in LangSmith
utils.tracing_is_enabled()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index Name
index_name = "testprojectv1"

# Create index
index = pc.Index(index_name)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)


# In this example, we'll create the index if it doesn't exist and then proceed with the indexing pipeline next
def ensure_index():
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if index_name in existing_indexes:
        logging.info(f"Index '{index_name}' already exists. Skipping creation.")
    else:
        logging.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(5)  # Ensure index is ready


# This method is to load the document and split it into chunks
def load_and_split_documents(filepath):
    logging.info("Loading document...")
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    return {"all_splits": splits, "total_Splits:": len(splits), "message": "Documents loaded and split successfully!"}


# This method is to generate embeddings for the chunks/splits created in the previous method
def embed_documents(inputs):
    splits = inputs["all_splits"]
    logging.info("Generating embeddings...")
    # embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    embeddings = embeddings_model.embed_documents([split.page_content for split in splits])

    # Compute norms (to ensure embeddings aren't garbage)
    norms = [sum(e[:5]) for e in embeddings[:5]]  # Get first 5 embeddings' norm
    
    return {"all_embeddings": embeddings, "norms": norms, "message": "Embeddings generated successfully!"}


# This method is to upsert the embeddings of the chunks/splits from the previous method into Pinecone Using Batch Upsert
def upsert_embeddings(data):  # This function expects a dictionary
    splits = data["splits"]["all_splits"]
    embeddings = data["embeddings"]["all_embeddings"]
    logging.info(f"Upserting {len(embeddings)} documents into Pinecone...")
    index = pc.Index(index_name)

    vectors = [
        {
            "id": f"doc_{split.metadata.get('source')}_{i}_{split.metadata.get('page_label', 'no_label')}",  # Ensure ID is valid and unique(because vectors of same Id get overwritten by the latest vector)
            "values": emb,
            "metadata": {"text": split.page_content}
        }
        for i, (split, emb) in enumerate(zip(splits, embeddings)) if len(emb) > 0
    ]

    logging.info(f"The first vector to be upserted is: {vectors[0]}")

    BATCH_SIZE = 100  # Recommended batch size
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace='ns2')
        logging.info(f"Upserted batch {i // BATCH_SIZE + 1} of {len(vectors) // BATCH_SIZE + 1}")
    
    logging.info(f"Upserted {len(vectors)} vectors into the vector store.")
    return f"Upserted {len(vectors)} vectors into the vector store."


# Turn Functions into Runnables
load_split_runnable = RunnableLambda(load_and_split_documents)
embed_runnable = RunnableLambda(embed_documents)
upsert_runnable = RunnableLambda(upsert_embeddings)


# Using LangChain's | Operator for an Indexing Chain
indexing_chain = (
    load_split_runnable 
    | {
        "splits": RunnablePassthrough(),
        "embeddings": embed_runnable
    }
    | upsert_runnable
)


# This method is to define the indexing pipeline and pre-requisites before upserting the chunks and embeddings
def run_indexing_pipeline(filepath):
    ensure_index()
    indexing_chain.invoke(filepath)
    logging.info("Indexing pipeline completed successfully!")


# Run the indexing pipeline
run_indexing_pipeline("./data/BOFA_safedepositbox_disclosures.pdf")


# Print index stats to verify index creation and upsertion
print(index.describe_index_stats())