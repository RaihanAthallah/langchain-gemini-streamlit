import os
import sys
from pathlib import Path
from typing import Literal

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from agents.faq_agent import invoke_faq_agent

try:
    import agents.DBQNA as DBQNA  # type: ignore
except Exception:
    DBQNA = None

try:
    import agents.RAG as RAG  # type: ignore
except Exception:
    RAG = None


load_dotenv(override=True)
st.title("Lab 8 Supervisor (DBQNA + RAG + FAQ)")


class BestAgent(BaseModel):
    agent_name: Literal["DBQNA", "RAG", "FAQ"]


class SupervisorState(MessagesState):
    user_question: str


router_model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"), temperature=0)
DB_PATH = os.getenv("DB_PATH", "")


def _last_ai_message(output: dict) -> AIMessage:
    messages = output.get("messages", [])
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, AIMessage):
            return last
        return AIMessage(content=str(getattr(last, "content", last)))
    return AIMessage(content="No answer was produced.")


def supervisor(state: SupervisorState) -> Command[Literal["DBQNA", "RAG", "FAQ", END]]:
    last_message = state["messages"][-1]
    instruction = SystemMessage(
        content=(
            "You are a supervisor that routes user questions to the best agent.\n"
            "- DBQNA: questions answerable from SQL/database records.\n"
            "- RAG: company profile or Dexa Medica document questions.\n"
            "- FAQ: frequently asked questions from FAQ documents.\n"
            "Return only one agent name."
        )
    )
    model_with_structure = router_model.with_structured_output(BestAgent)
    response = model_with_structure.invoke([instruction, last_message])
    return Command(update={"user_question": last_message.content}, goto=response.agent_name)


def call_faq(state: SupervisorState) -> Command[Literal[END]]:
    prompt = state["user_question"]
    result = invoke_faq_agent(prompt, thread_id="lab8-supervisor")
    return Command(goto=END, update={"messages": [AIMessage(content=result["answer"])]})


def call_rag(state: SupervisorState) -> Command[Literal[END]]:
    if RAG is None or not hasattr(RAG, "graph"):
        return Command(
            goto=END,
            update={"messages": [AIMessage(content="RAG agent module is missing in this workspace.")]},
        )
    prompt = state["user_question"]
    result = RAG.graph.invoke({"messages": HumanMessage(content=prompt)})
    return Command(goto=END, update={"messages": [_last_ai_message(result)]})


def call_dbqna(state: SupervisorState) -> Command[Literal[END]]:
    if DBQNA is None or not hasattr(DBQNA, "graph"):
        return Command(
            goto=END,
            update={"messages": [AIMessage(content="DBQNA agent module is missing in this workspace.")]},
        )
    prompt = state["user_question"]
    result = DBQNA.graph.invoke(
        {
            "messages": HumanMessage(content=prompt),
            "db_name": DB_PATH,
            "user_question": prompt,
        }
    )
    return Command(goto=END, update={"messages": [_last_ai_message(result)]})


supervisor_agent = (
    StateGraph(SupervisorState)
    .add_node("supervisor", supervisor)
    .add_node("RAG", call_rag)
    .add_node("DBQNA", call_dbqna)
    .add_node("FAQ", call_faq)
    .add_edge(START, "supervisor")
    .compile(name="supervisor")
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Write your question here ...")
if prompt:
    st.session_state.chat_history.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        with st.spinner("Routing and generating answer..."):
            result = supervisor_agent.invoke({"messages": [HumanMessage(content=prompt)]})
            final = _last_ai_message(result).content
        st.markdown(final)
        st.session_state.chat_history.append({"role": "ai", "content": final})

