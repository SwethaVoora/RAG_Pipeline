import logging
from config import index, embeddings_model

# Created a method to retrieve the last Q&A pair from the chat history and format them into a single string.
def get_last_qa_context(chat_history):
    if len(chat_history) < 2:
        return ""
    last_idx = len(chat_history)
    prev_message = chat_history[last_idx - 2]  # Last user query
    last_message = chat_history[last_idx - 1]  # Last assistant response

    if prev_message.get("role") == "user" and last_message.get("role") == "assistant":
        return f"Previous User Query: {prev_message['content']}\nPrevious Assistant Response: {last_message['content']}"
    return ""


# Cretaing a method to expand the user query if it is a follow-up question
def expand_followup_query(user_query, context):
    # Append the follow-up question to the context.
    expanded_query = f"{context}\nFollow-up question: {user_query}"
    # logging.info("Expanded follow-up query: " + expanded_query)
    return expanded_query.strip()

# Store latest Q&A in user specific namespace prefixed with `conv_vector`
def store_chat_history(user_query, llm_response, user_specific_namespace):
    logging.info("Storing latest Q&A into conv_vector...")
    embedding = embeddings_model.embed_documents([user_query + " " + llm_response])
    
    vector = {
        "id": f"chat_{hash(user_query)}",
        "values": embedding[0],
        "metadata": {"text": f"Q: {user_query}\nA: {llm_response}"}
    }
    
    index.upsert(vectors=[vector], namespace=user_specific_namespace)
    logging.info("Chat history stored successfully.")
