from langchain_core.tools import tool

from agents.faq_agent import invoke_faq_agent


@tool
def search_faq_answer(question: str) -> str:
    """Answer FAQ questions using the Dexa Medica PDF knowledge base."""
    result = invoke_faq_agent(question, thread_id="lab8-supervisor")
    return result["answer"]

