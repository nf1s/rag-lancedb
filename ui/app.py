"""Streamlit chat, ingestion and administration UI."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")
TIMEOUT = httpx.Timeout(300.0, connect=10.0)
ADMIN_ROLES = {"admin", "owner"}

st.set_page_config(page_title="RAG on LanceDB", page_icon="📚", layout="centered")


# --------------------------------------------------------------------------- http


def _headers() -> dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Call the API; clear the session when the token is no longer valid."""
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        response = client.request(method, path, headers=_headers(), **kwargs)
    if response.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear()
        st.rerun()
    return response


def error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else payload
    except Exception:  # noqa: BLE001
        detail = response.text
    return f"{response.status_code}: {detail}"


def iter_sse(path: str, payload: dict[str, Any]) -> Iterator[tuple[str | None, dict[str, Any]]]:
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        with client.stream("POST", path, json=payload, headers=_headers()) as response:
            if response.status_code >= 400:
                response.read()
                raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
            event: str | None = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    raw = line.split(":", 1)[1].strip()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"text": raw}
                    yield event, data


# --------------------------------------------------------------------------- auth


def login_page() -> None:
    st.title("📚 RAG on LanceDB")
    st.caption(f"API: {API_URL}")
    with st.form("login"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if not submitted:
        return

    try:
        response = httpx.post(
            f"{API_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        st.error(f"API unreachable: {exc}")
        return

    if response.status_code != 200:
        st.error(f"Sign-in failed: {error_text(response)}")
        return

    data = response.json()
    st.session_state["token"] = data["token"]
    st.session_state["user"] = data["user"]
    st.session_state["workspaces"] = data["workspaces"]
    st.session_state["workspace_id"] = data["workspaces"][0]["id"] if data["workspaces"] else None
    st.session_state["messages"] = []
    st.rerun()


def refresh_me() -> None:
    response = api("GET", "/auth/me")
    if response.status_code != 200:
        return
    data = response.json()
    st.session_state["user"] = data["user"]
    st.session_state["workspaces"] = data["workspaces"]
    workspace_ids = {workspace["id"] for workspace in data["workspaces"]}
    if st.session_state.get("workspace_id") not in workspace_ids:
        st.session_state["workspace_id"] = (
            data["workspaces"][0]["id"] if data["workspaces"] else None
        )
        st.session_state["messages"] = []


# --------------------------------------------------------------------------- ui


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for position, source in enumerate(sources, start=1):
            heading = (source.get("metadata") or {}).get("heading")
            label = source.get("source", "unknown")
            if heading:
                label = f"{label} > {heading}"
            score = float(source.get("score") or 0.0)
            st.markdown(f"**[{position}] {label}** · score `{score:.3f}`")
            preview = source.get("text", "")
            if len(preview) > 400:
                preview = preview[:400] + " …"
            st.caption(preview)


def sidebar() -> None:
    user = st.session_state["user"]
    workspaces = st.session_state["workspaces"]
    with st.sidebar:
        st.subheader("Session")
        suffix = " · global admin" if user.get("global_role") == "admin" else ""
        st.caption(f"Signed in as **{user['username']}**{suffix}")

        if workspaces:
            labels = {f"{w['name']} ({w['role']})": w["id"] for w in workspaces}
            options = list(labels)
            current = st.session_state.get("workspace_id")
            index = next(
                (position for position, label in enumerate(options) if labels[label] == current),
                0,
            )
            choice = st.selectbox("Workspace", options, index=index)
            if labels[choice] != current:
                st.session_state["workspace_id"] = labels[choice]
                st.session_state["messages"] = []
                st.rerun()
        else:
            st.warning("No workspaces are visible to this account.")

        workspace_id = st.session_state.get("workspace_id")
        if workspace_id:
            stats = api("GET", f"/ws/{workspace_id}/stats")
            if stats.status_code == 200:
                body = stats.json()
                st.metric("Indexed chunks", body.get("rows", 0))
                st.caption(
                    f"{body.get('sources', 0)} source(s) · "
                    f"FTS {'ready' if body.get('fts_ready') else 'not built'}"
                )

        try:
            health = httpx.get(f"{API_URL}/healthz", timeout=5.0).json()
            store_ok = health.get("store", {}).get("status") == "ok"
            llm_ok = health.get("llm", {}).get("reachable")
            st.caption(
                "Store " + ("🟢" if store_ok else "🔴") + " · LLM " + ("🟢" if llm_ok else "🔴")
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Health unavailable: {exc}")

        if st.button("Refresh"):
            refresh_me()
            st.rerun()
        if st.button("Log out"):
            try:
                api("POST", "/auth/logout")
            finally:
                st.session_state.clear()
                st.rerun()


def ingest_page(workspace: dict[str, Any] | None) -> None:
    if workspace is None:
        st.info("Select a workspace in the sidebar.")
        return
    st.subheader(f"Ingest into {workspace['name']}")
    uploads = st.file_uploader(
        "Add Markdown or text documents",
        type=["md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploads, type="primary"):
        files = [("files", (upload.name, upload.getvalue(), "text/plain")) for upload in uploads]
        with st.spinner("Chunking and embedding…"):
            response = api("POST", f"/ws/{workspace['id']}/ingest", files=files)
        if response.status_code == 200:
            for result in response.json().get("results", []):
                if result.get("warning"):
                    st.warning(f"{result['source']}: {result['warning']}")
                else:
                    st.success(f"{result['source']}: {result['chunks']} chunks indexed")
        else:
            st.error(f"Ingest failed: {error_text(response)}")


def chat_page(workspace: dict[str, Any] | None) -> None:
    if workspace is None:
        st.info("Select a workspace in the sidebar.")
        return
    st.caption(f"Answering from **{workspace['name']}**")

    st.session_state.setdefault("messages", [])
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask about your documents")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources_holder: dict[str, Any] = {}

        def token_stream() -> Iterator[str]:
            try:
                for event, data in iter_sse(f"/ws/{workspace['id']}/query", {"question": prompt}):
                    if event == "sources":
                        sources_holder["sources"] = data.get("sources", [])
                    elif event == "token":
                        yield data.get("text", "")
                    elif event == "error":
                        yield f"\n\n**Error:** {data.get('message')}"
            except Exception as exc:  # noqa: BLE001
                yield f"\n\n**Request failed:** {exc}"

        answer = st.write_stream(token_stream())
        sources = sources_holder.get("sources", [])
        render_sources(sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def admin_page(user: dict[str, Any], workspaces: list[dict[str, Any]]) -> None:
    st.subheader("Administration")
    is_global_admin = user.get("global_role") == "admin"

    if is_global_admin:
        with st.form("create_workspace"):
            slug = st.text_input("Workspace slug", placeholder="team-docs")
            name = st.text_input("Display name", placeholder="Team docs")
            if st.form_submit_button("Create workspace"):
                response = api("POST", "/workspaces", json={"slug": slug, "name": name})
                if response.status_code == 201:
                    st.success(f"Workspace '{response.json()['slug']}' created")
                    refresh_me()
                    st.rerun()
                else:
                    st.error(error_text(response))

        with st.form("create_user"):
            username = st.text_input("New username")
            password = st.text_input("Temporary password", type="password")
            global_role = st.selectbox("Global role", ["user", "admin"])
            if st.form_submit_button("Create user"):
                response = api(
                    "POST",
                    "/users",
                    json={
                        "username": username,
                        "password": password,
                        "global_role": global_role,
                    },
                )
                if response.status_code == 201:
                    st.success(f"User '{response.json()['username']}' created")
                else:
                    st.error(error_text(response))

    admin_workspaces = [
        workspace
        for workspace in workspaces
        if is_global_admin or workspace.get("role") in ADMIN_ROLES
    ]
    if not admin_workspaces:
        st.info("You are not an admin of any workspace.")
        return

    labels = {
        f"{workspace['name']} ({workspace['role']})": workspace["id"]
        for workspace in admin_workspaces
    }
    choice = st.selectbox("Manage workspace", list(labels))
    workspace_id = labels[choice]
    workspace = next(w for w in admin_workspaces if w["id"] == workspace_id)

    members_response = api("GET", f"/workspaces/{workspace_id}/members")
    members = members_response.json() if members_response.status_code == 200 else []
    if members_response.status_code != 200:
        st.error(error_text(members_response))
    elif members:
        st.dataframe(members, width="stretch")
    else:
        st.caption("No members yet.")

    with st.form("add_member"):
        member_name = st.text_input("Username to add")
        role = st.selectbox("Role", ["viewer", "editor", "admin", "owner"])
        if st.form_submit_button("Add or update member"):
            response = api(
                "POST",
                f"/workspaces/{workspace_id}/members",
                json={"username": member_name, "role": role},
            )
            if response.status_code == 201:
                st.success(f"Granted '{role}' to '{member_name}'")
                st.rerun()
            else:
                st.error(error_text(response))

    if members:
        removable = {
            f"{member['username']} ({member['role']})": member["user_id"] for member in members
        }
        picked = st.selectbox("Remove member", list(removable))
        if st.button("Remove member"):
            response = api(
                "DELETE",
                f"/workspaces/{workspace_id}/members/{removable[picked]}",
            )
            if response.status_code == 200:
                st.success("Membership removed")
                st.rerun()
            else:
                st.error(error_text(response))

    if st.button("Rebuild search index"):
        response = api("POST", f"/ws/{workspace_id}/reindex")
        if response.status_code == 200:
            st.success("Search index rebuilt")
        else:
            st.error(error_text(response))

    if is_global_admin or workspace.get("role") == "owner":
        st.divider()
        st.warning("Deleting a workspace permanently removes all of its documents.")
        confirm = st.text_input(f"Type '{workspace['slug']}' to confirm")
        if st.button("Delete workspace", disabled=confirm != workspace["slug"]):
            response = api("DELETE", f"/workspaces/{workspace_id}")
            if response.status_code == 200:
                st.success("Workspace deleted")
                refresh_me()
                st.rerun()
            else:
                st.error(error_text(response))


def main() -> None:
    if not st.session_state.get("token"):
        login_page()
        return

    if "user" not in st.session_state:
        refresh_me()
    if "user" not in st.session_state:
        st.error("Could not load the session. Try signing in again.")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
        return

    sidebar()

    st.title("📚 RAG on LanceDB")
    user = st.session_state["user"]
    workspaces = st.session_state["workspaces"]
    workspace_id = st.session_state.get("workspace_id")
    workspace = next(
        (candidate for candidate in workspaces if candidate["id"] == workspace_id),
        None,
    )

    labels = ["Ingest", "Chat"]
    can_admin = user.get("global_role") == "admin" or any(
        candidate.get("role") in ADMIN_ROLES for candidate in workspaces
    )
    if can_admin:
        labels.append("Admin")

    tabs = st.tabs(labels)
    with tabs[0]:
        ingest_page(workspace)
    with tabs[1]:
        chat_page(workspace)
    if can_admin:
        with tabs[2]:
            admin_page(user, workspaces)


main()
