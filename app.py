import streamlit as st
from config import index
from orchestrator import execute_user_query
import uuid


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
        # 1. Display the user query as part of the chat history
        st.chat_message("user").markdown(user_query)

        # 2. Route and invoke either unstructured or structured path based on prior turns
        result = execute_user_query(user_query, st.session_state.chat_history, session_namespace)
        response = result["answer"]
        route = result["route"]

        # 3. Append user query to chat history/messages
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # 4. Display the output
        st.chat_message("assistant").markdown(response)
        st.caption(f"Route: {route} | Reason: {result['router_reason']}")

        # 5. Append RAG response to chat history/messages
        st.session_state.chat_history.append({"role": "assistant", "content": response, "route": route})

# Delete chat history button
if st.button("Delete Chat History"):
    print(f"Deleting the chat history for session namespace: {session_namespace}")
    index.delete(namespace=session_namespace, delete_all=True)
    st.session_state.chat_history = []  # Clear UI history
    st.success("Chat history deleted successfully.")
