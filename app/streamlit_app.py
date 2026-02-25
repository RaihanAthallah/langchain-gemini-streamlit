import logging
import sys
import uuid
from pathlib import Path

import httpx

_project_root = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from agents.faq_agent import invoke_faq_agent
from app.ingest import ingest_pdf_folder


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _render_sidebar() -> None:
    st.sidebar.title("Conversation")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    for index, message in enumerate(st.session_state.messages, start=1):
        role = "You" if message["role"] == "user" else "Assistant"
        st.sidebar.write(f"{index}. {role}: {message['content'][:80]}")


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(source)


def _invoke_with_retry(prompt: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(max_attempts):
        try:
            logger.info("Processing chat request (attempt %d/%d): %s", attempt + 1, max_attempts, prompt[:80])
            result = invoke_faq_agent(prompt, thread_id=st.session_state.thread_id)
            logger.info("Chat request completed. Chunks: %d", len(result["chunks"]))
            return result
        except (httpx.RemoteProtocolError, httpx.ConnectError, ConnectionError) as e:
            last_error = e
            logger.warning("Chat request failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_attempts - 1:
                st.toast(f"Connection issue, retrying ({attempt + 1}/{max_attempts})...")
    raise last_error


def _sources_from_chunks(chunks: list[dict]) -> list[str]:
    return [
        f"{item.get('source_file', '-')}"
        f" | chunk {item.get('chunk_index', '-')}"
        f" | score {float(item.get('score', 0.0)):.3f}"
        for item in chunks
    ]


def main() -> None:
    st.set_page_config(page_title="Dexa FAQ Agent", page_icon="💬", layout="wide")
    _init_state()
    _render_sidebar()

    st.title("Dexa Medica FAQ Agent")
    st.caption("Ask questions based on FAQ and profile PDFs.")

    if st.button("Ingest PDFs from base-knowledge"):
        with st.spinner("Ingesting documents..."):
            count = ingest_pdf_folder(Path("base-knowledge"))
        st.success(f"Ingestion complete. Processed {count} chunks.")

    _render_history()

    prompt = st.chat_input("Ask a question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = _invoke_with_retry(prompt)
        answer = result["answer"]
        sources = _sources_from_chunks(result["chunks"])
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()

