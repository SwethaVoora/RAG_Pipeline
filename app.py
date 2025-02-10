import streamlit as st
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from config import index, embeddings_model
from retrieval_and_generation_pipeline import rag_chain
from helper import get_last_qa_context, store_chat_history, expand_followup_query  # Import the chain you built for retrieval
import uuid, logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Created a method to classify if the user query is a follow-up by comparing its embedding with the combined embedding of the last Q&A pair.
def is_followup_question(user_query, context):
    logging.info("1. Classifying the query as follow-up or not...")
    # If the last Q&A pair is not available, return False
    context = get_last_qa_context(st.session_state.chat_history)
    if context == "":
        return False

    # Embedding-based follow-up detection using cosine similarity
    # Since embeddings_model.embed_documents returns a list, we wrap them with np.array.
    query_embedding = np.array(embeddings_model.embed_documents([user_query]))
    context_embedding = np.array(embeddings_model.embed_documents([context]))

    # cosine_similarity returns a 2D array (1x1 in this case), so we extract the first element.
    similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
    logging.info(f"2. Follow-up Detection | Cosine Similarity: {'True' if similarity >= 0.25 else 'False'} | Similarity: {similarity}")

    # Rule-based follow-up detection
    ambiguous_keywords = {"it", "more", "that", "this"}
    rule_based_followup = any(word in user_query.lower().split() for word in ambiguous_keywords)
    logging.info(f"3. Follow-up Detection | Rule-based: {'True' if rule_based_followup else 'False'}")
    
    # Return True if the similarity is above 0.25 or if the rule-based follow-up detection is True.
    return similarity >= 0.20 or rule_based_followup


# Streamlit UI
st.title("Interactive RAG-powered Q&A")
st.subheader("Please, Delete your chat history before exiting the app.", divider="rainbow")
st.write("Ask a question based on the indexed knowledge base.")

# Session State for chat history display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Generate a unique session ID if not already created
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generates a random UUID - this will be the name of this user's conv_vector namespace

session_namespace = f"conv_vector_{st.session_state.session_id}"
# print(f"Your session namespace: {session_namespace}") # Just for debugging purposes

# Display chat history/chat messages
for message in st.session_state.chat_history:
    st.chat_message(message['role']).markdown(message['content'])

# chat_input field Allows "Enter" to submit
user_query = st.chat_input("Enter your question:")

if user_query is not None:  # Ensures it only runs if the user has actually entered something
    if user_query.strip() == "":  # Show warning only if the input is explicitly empty
        st.warning("Please enter a question.")
    else:

        # Compute the context from the chat history once.
        context = get_last_qa_context(st.session_state.chat_history)


        expanded_user_query = ""
        # Detect if it's a follow-up
        if is_followup_question(user_query, context):
            logging.info("CLASSIFICATION RESULT:This is a follow-up question.")
            # 3. Expand the follow-up question
            expanded_user_query = expand_followup_query(user_query, context)

        # 1. Display the user query as part of the chat history
        st.chat_message("user").markdown(user_query)

        # 2. Append user query to chat history/messages
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # 3. Invoke/Run the RAG pipeline
        # ##############################################################################################################
        if len(st.session_state.chat_history) < 2 or expanded_user_query == "": # If this is the first question or a non follow-up question
            response = rag_chain.invoke({"question": user_query, "user_specific_namespace": "", "expanded_user_query": ""})
        else:
            response = rag_chain.invoke({"question": user_query, "user_specific_namespace": session_namespace, "expanded_user_query": expanded_user_query})
        # ############################################################################################################## 
        
        # 4. Display the RAG output as part of the chat history
        st.chat_message("assistant").markdown(response)

        # 5. Append RAG response to chat history/messages
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Store the latest Q&A in the vector store
        store_chat_history(user_query, response, session_namespace)

# Delete chat history button
if st.button("Delete Chat History"):
    print(f"Deleting the chat history for session namespace: {session_namespace}")
    index.delete(namespace=session_namespace, delete_all=True)
    st.session_state.chat_history = []  # Clear UI history
    st.success("Chat history deleted successfully.")