import os
from dotenv import load_dotenv
from langsmith import utils
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pinecone import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate

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

# Created a retriever to embed the user query and retrieve relevant document chunks from `ns1`
def retriever(question):
    # always include the question in []. because embed_documents expects a list. 
    # If its not a list, then each character will be treated as a separate document that needs to be embedded seperately.
    embeddedQuestion = embeddings_model.embed_documents([question]) 
    similar_docs = index.query(vector=embeddedQuestion, top_k=3, namespace="ns1", include_metadata=True)
    return similar_docs

# created a function to format the retrieved document chunks into a single string
def formatContext(retrieved_docs):
    return "\n".join(doc.metadata["text"] for doc in retrieved_docs['matches'])


# Wrap retriever and formatContext as runnables
retriever_runnable = RunnableLambda(retriever)  # Wrap retriever
formatContext_runnable = RunnableLambda(formatContext)  # Wrap formatContext


# This is the prompt template, which takes the context and question as inputs and passes them to the model
prompt = ChatPromptTemplate.from_template("""
    Answer the user question based on the following context.
    If you dont know the answer, just say you dont know.
                                          
    Context: {context} 
                                          
    Question: {question}""")

# we are using openai's gpt-3.5-turbo to answer our query
model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model="gpt-3.5-turbo")

# Defining the output parser to parse the output of the LLM and exrtact the content out of the AIMessage object response
def outputParser(response):
    return response.content


# Chain
rag_chain = (
    {"context": retriever_runnable | formatContext_runnable, "question": RunnablePassthrough()}
    | prompt
    | model
    | outputParser
)

# Question
rag_chain.invoke("What is the locking and unlocking system like at BOFA?")