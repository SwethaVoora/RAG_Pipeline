import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from agent_graph import run_langgraph_agent_query
from config import embeddings_model
from helper import get_last_qa_context, expand_followup_query, store_chat_history
from retrieval_and_generation_pipeline import rag_chain
from router import route_query


def _is_unstructured_followup(user_query: str, chat_history: list[dict]) -> bool:
    logging.info("1. Classifying the query as follow-up or not...")
    context = get_last_qa_context(chat_history)
    if context == "":
        return False

    query_embedding = np.array(embeddings_model.embed_documents([user_query]))
    context_embedding = np.array(embeddings_model.embed_documents([context]))
    similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
    logging.info(
        "2. Follow-up Detection | Cosine Similarity: %s | Similarity: %s",
        "True" if similarity >= 0.25 else "False",
        similarity,
    )

    ambiguous_keywords = {"it", "more", "that", "this", "they", "them"}
    rule_based_followup = any(word in user_query.lower().split() for word in ambiguous_keywords)
    logging.info("3. Follow-up Detection | Rule-based: %s", "True" if rule_based_followup else "False")

    return similarity >= 0.20 or rule_based_followup


def execute_user_query(user_query: str, chat_history: list[dict], session_namespace: str) -> dict:
    decision = route_query(user_query, chat_history)

    if decision.route == "structured":
        result = run_langgraph_agent_query(user_query)
        return {
            "answer": result["answer"],
            "route": "structured",
            "router_reason": decision.reason,
            "router_confidence": decision.confidence,
        }

    context = get_last_qa_context(chat_history)
    expanded_user_query = ""
    if _is_unstructured_followup(user_query, chat_history):
        logging.info("CLASSIFICATION RESULT:This is a follow-up question.")
        expanded_user_query = expand_followup_query(user_query, context)

    if len(chat_history) < 2 or expanded_user_query == "":
        answer = rag_chain.invoke(
            {"question": user_query, "user_specific_namespace": "", "expanded_user_query": ""}
        )
    else:
        answer = rag_chain.invoke(
            {
                "question": user_query,
                "user_specific_namespace": session_namespace,
                "expanded_user_query": expanded_user_query,
            }
        )

    store_chat_history(user_query, answer, session_namespace)
    return {
        "answer": answer,
        "route": "unstructured",
        "router_reason": decision.reason,
        "router_confidence": decision.confidence,
    }
