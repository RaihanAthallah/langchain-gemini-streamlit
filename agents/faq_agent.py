import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.llm_clients import build_chat_client
from app.semantic_search import search_relevant_chunks

logger = logging.getLogger(__name__)

MIN_RELEVANCE_SCORE = 0.68
MAX_RETRY = 1


class FAQState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    query: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    attempt: int


def _conversation_snippet(messages: list, max_items: int = 6) -> str:
    if not messages:
        return ""
    recent = messages[-max_items:]
    lines: list[str] = []
    for item in recent:
        role = "assistant" if isinstance(item, AIMessage) else "user"
        lines.append(f"{role}: {item.content}")
    return "\n".join(lines)


def retrieve_chunks(state: FAQState) -> FAQState:
    logger.info("Retrieving chunks for query: %s", state["query"][:60])
    chunks = search_relevant_chunks(state["query"], top_k=4)
    logger.info("Retrieved %d chunks", len(chunks))
    return {**state, "retrieved_chunks": chunks}


def rewrite_query(state: FAQState) -> FAQState:
    logger.info("Rewriting query (attempt %d): %s", state["attempt"] + 1, state["question"][:60])
    llm = build_chat_client(temperature=0.0)
    history_text = _conversation_snippet(state.get("messages", []))
    prompt = (
        "Rewrite this FAQ question to improve semantic search retrieval. "
        "Keep meaning the same and return only one rewritten question.\n\n"
        f"Conversation context:\n{history_text}\n\n"
        f"Question: {state['question']}"
    )
    rewritten = llm.invoke(prompt).content.strip()
    logger.info("Rewritten query: %s", rewritten[:60])
    return {**state, "query": rewritten, "attempt": state["attempt"] + 1}


def generate_answer(state: FAQState) -> FAQState:
    logger.info("Generating answer from %d chunks", len(state["retrieved_chunks"]))
    llm = build_chat_client(temperature=0.1)
    history_text = _conversation_snippet(state.get("messages", []))
    context = "\n\n".join(
        [f"[{item['source_file']}#{item['chunk_index']}]\n{item['content']}" for item in state["retrieved_chunks"]]
    )
    prompt = (
        "You are a FAQ assistant for Dexa Medica.\n"
        "Use only the provided context to answer the user's question.\n"
        "If the context does not contain enough information, say you do not know.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Question:\n{state['question']}\n\n"
        f"Context:\n{context}"
    )
    answer = llm.invoke(prompt).content.strip()
    return {**state, "answer": answer, "messages": [AIMessage(content=answer)]}


def no_answer(state: FAQState) -> FAQState:
    logger.info("No relevant chunks found, returning fallback message")
    message = (
        "I could not find a reliable answer in the FAQ documents. "
        "Please rephrase your question or ask about Dexa Medica FAQ details."
    )
    return {
        **state,
        "answer": message,
        "messages": [AIMessage(content=message)],
    }


def route_after_retrieval(state: FAQState) -> Literal["answer", "rewrite", "no_answer"]:
    if not state["retrieved_chunks"]:
        if state["attempt"] < MAX_RETRY:
            return "rewrite"
        return "no_answer"

    best_score = float(state["retrieved_chunks"][0].get("score", 0.0))
    if best_score >= MIN_RELEVANCE_SCORE:
        return "answer"
    if state["attempt"] < MAX_RETRY:
        return "rewrite"
    return "no_answer"


def build_faq_graph():
    graph = StateGraph(FAQState)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("answer", generate_answer)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer": "answer",
            "rewrite": "rewrite",
            "no_answer": "no_answer",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("no_answer", END)

    return graph.compile(checkpointer=MemorySaver())


FAQ_GRAPH = build_faq_graph()


def invoke_faq_agent(question: str, thread_id: str = "default") -> dict[str, Any]:
    logger.info("Invoking FAQ agent | thread_id=%s | question=%s", thread_id, question[:80])
    initial_state: FAQState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "query": question,
        "retrieved_chunks": [],
        "answer": "",
        "attempt": 0,
    }
    result = FAQ_GRAPH.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    logger.info("FAQ agent completed | answer_len=%d | chunks=%d", len(result["answer"]), len(result["retrieved_chunks"]))
    return {
        "answer": result["answer"],
        "chunks": result["retrieved_chunks"],
    }

