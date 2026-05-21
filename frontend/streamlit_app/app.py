from __future__ import annotations

import json
from typing import Any

import streamlit as st

from api_client import MODEL_SERVER_URL, ApiClientError, api_get, api_post


MEMORY_TYPES = ("episodic", "semantic", "procedural")


def main() -> None:
    st.set_page_config(page_title="Maintainer's Copilot", layout="wide")
    _init_session()
    _render_sidebar()

    st.title("Maintainer's Copilot")
    st.caption(f"Backend: {MODEL_SERVER_URL}")

    chat_tab, memory_tab, admin_tab = st.tabs(["Chat", "Memory", "Admin"])
    with chat_tab:
        render_chat_page()
    with memory_tab:
        render_memory_page()
    with admin_tab:
        render_admin_page()


def _init_session() -> None:
    defaults = {
        "access_token": None,
        "email": None,
        "role": None,
        "chat_response": None,
        "memories": None,
        "memory_search_results": None,
        "audit_rows": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_sidebar() -> None:
    st.sidebar.header("Login")
    if is_logged_in():
        st.sidebar.success(f"{st.session_state.email} ({st.session_state.role})")
        if st.sidebar.button("Logout", use_container_width=True):
            for key in ("access_token", "email", "role"):
                st.session_state[key] = None
            st.rerun()
        return

    with st.sidebar.form("login_form"):
        email = st.text_input("Email", value="user@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        try:
            payload = api_post(
                "/auth/login",
                json={"email": email, "password": password},
            )
        except ApiClientError as exc:
            st.sidebar.error(str(exc))
            return

        st.session_state.access_token = payload["access_token"]
        st.session_state.email = payload["email"]
        st.session_state.role = payload["role"]
        st.sidebar.success("Logged in.")
        st.rerun()


def render_chat_page() -> None:
    st.subheader("Chat")
    if not require_login():
        return

    with st.form("chat_form"):
        message = st.text_area("Message", height=120, placeholder="Ask for triage guidance...")
        issue_title = st.text_input("Issue title")
        issue_body = st.text_area("Issue body", height=180)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            use_rag = st.checkbox("Use RAG", value=True)
        with col_b:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Send")

    if submitted:
        if not message.strip():
            st.warning("Message is required.")
        else:
            body = {
                "message": message,
                "issue_title": issue_title or None,
                "issue_body": issue_body or None,
                "use_rag": use_rag,
                "top_k": int(top_k),
            }
            try:
                st.session_state.chat_response = api_post(
                    "/chat",
                    token=st.session_state.access_token,
                    json=body,
                )
            except ApiClientError as exc:
                st.error(str(exc))

    response = st.session_state.chat_response
    if response:
        st.markdown("#### Answer")
        st.write(response.get("answer", ""))

        tool_calls = response.get("tool_calls") or []
        with st.expander("Tool calls", expanded=bool(tool_calls)):
            if tool_calls:
                st.dataframe(tool_calls, use_container_width=True, hide_index=True)
            else:
                st.info("No tools were called.")

        trace = response.get("trace")
        if trace:
            with st.expander("Trace"):
                st.json(trace)


def render_memory_page() -> None:
    st.subheader("Memory")
    if not require_login():
        return

    with st.form("write_memory_form"):
        text = st.text_area("Memory text", height=120)
        memory_type = st.selectbox("Memory type", MEMORY_TYPES)
        metadata_source = st.text_input("Metadata source", placeholder="optional")
        submitted = st.form_submit_button("Write memory")

    if submitted:
        if not text.strip():
            st.warning("Memory text is required.")
        else:
            metadata = {}
            if metadata_source.strip():
                metadata["source"] = metadata_source.strip()
            try:
                memory = api_post(
                    "/memory/write",
                    token=st.session_state.access_token,
                    json={
                        "text": text,
                        "memory_type": memory_type,
                        "metadata": metadata,
                    },
                )
                st.success(f"Wrote memory {memory['memory_id']}")
            except ApiClientError as exc:
                st.error(str(exc))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("List memories", use_container_width=True):
            try:
                st.session_state.memories = api_get(
                    "/memory",
                    token=st.session_state.access_token,
                ).get("memories", [])
            except ApiClientError as exc:
                st.error(str(exc))
    with col_b:
        limit = st.number_input("Memory limit", min_value=1, max_value=100, value=10, step=1)

    query = st.text_input("Search query")
    if st.button("Search memories", use_container_width=True):
        try:
            st.session_state.memory_search_results = api_post(
                "/memory/search",
                token=st.session_state.access_token,
                json={"query": query or None, "limit": int(limit)},
            ).get("memories", [])
        except ApiClientError as exc:
            st.error(str(exc))

    if st.session_state.memories is not None:
        st.markdown("#### Memories")
        render_records(st.session_state.memories)

    if st.session_state.memory_search_results is not None:
        st.markdown("#### Search results")
        render_records(st.session_state.memory_search_results)


def render_admin_page() -> None:
    st.subheader("Admin")
    if not require_login():
        return
    if st.session_state.role != "admin":
        st.warning("Admin only.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Admin ping", use_container_width=True):
            try:
                st.success(api_get("/admin/ping", token=st.session_state.access_token))
            except ApiClientError as exc:
                st.error(str(exc))
    with col_b:
        if st.button("Load memory audit", use_container_width=True):
            try:
                st.session_state.audit_rows = api_get(
                    "/memory/audit",
                    token=st.session_state.access_token,
                )
            except ApiClientError as exc:
                st.error(str(exc))

    if st.session_state.audit_rows is not None:
        st.markdown("#### Memory audit")
        render_records(st.session_state.audit_rows)


def render_records(records: list[dict[str, Any]]) -> None:
    if not records:
        st.info("No records found.")
        return
    st.dataframe(records, use_container_width=True, hide_index=True)
    with st.expander("Raw JSON"):
        st.code(json.dumps(records, indent=2), language="json")


def require_login() -> bool:
    if is_logged_in():
        return True
    st.info("Please log in from the sidebar.")
    return False


def is_logged_in() -> bool:
    return bool(st.session_state.access_token)


if __name__ == "__main__":
    main()
