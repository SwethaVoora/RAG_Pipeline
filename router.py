from dataclasses import dataclass


@dataclass
class RouteDecision:
    route: str
    reason: str
    confidence: float


STRUCTURED_KEYWORDS = {
    "sql",
    "table",
    "database",
    "supabase",
    "postgres",
    "postgresql",
    "schema",
    "users",
    "user id",
    "orders",
    "order_items",
    "products",
    "revenue",
    "sales",
    "count",
    "group by",
    "join",
}

UNSTRUCTURED_KEYWORDS = {
    "pdf",
    "document",
    "policy",
    "bofa",
    "safe deposit",
    "disclosure",
    "vault",
    "knowledge base",
}


def _last_assistant_route(chat_history: list[dict]) -> str | None:
    for message in reversed(chat_history):
        if message.get("role") == "assistant" and message.get("route") in {"structured", "unstructured"}:
            return message["route"]
    return None


def route_query(user_query: str, chat_history: list[dict]) -> RouteDecision:
    q = user_query.lower().strip()

    structured_hits = sum(1 for kw in STRUCTURED_KEYWORDS if kw in q)
    unstructured_hits = sum(1 for kw in UNSTRUCTURED_KEYWORDS if kw in q)

    # Bias follow-up pronouns toward the last assistant route.
    has_followup_pronoun = any(token in q.split() for token in {"it", "that", "those", "them", "more"})
    prev_route = _last_assistant_route(chat_history)
    if has_followup_pronoun and prev_route:
        return RouteDecision(route=prev_route, reason="Follow-up pronoun with prior route context.", confidence=0.72)

    if structured_hits > unstructured_hits and structured_hits > 0:
        return RouteDecision(route="structured", reason="Structured intent keywords detected.", confidence=0.9)

    if unstructured_hits > structured_hits and unstructured_hits > 0:
        return RouteDecision(route="unstructured", reason="Document-oriented keywords detected.", confidence=0.9)

    # Default fallback: keep unstructured as primary user experience for existing app.
    return RouteDecision(route="unstructured", reason="No strong structured signals; defaulting to unstructured.", confidence=0.55)
