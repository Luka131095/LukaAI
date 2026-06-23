from __future__ import annotations

import base64
import os
import time

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.elements.widgets.chat import ChatInputValue
from supabase import create_client

st.set_page_config(page_title="AI Assistant", layout="wide")

APP_VERSION = "general-purpose-v2"

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 5

SYSTEM_PROMPT = ""


def _max_output_tokens(model: str) -> int:
    if "opus" in model.lower():
        return 8192
    return 8192


def type_writer(text):
    for char in text:
        yield char
        time.sleep(0.005)


def _image_media_type(uploaded_file) -> str:
    media_type = uploaded_file.type
    if media_type in ALLOWED_IMAGE_TYPES:
        return media_type
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, media_type or "image/jpeg")


def build_user_content(text: str, files) -> str | list:
    blocks = []
    text = (text or "").strip()

    if text:
        blocks.append({"type": "text", "text": text})

    image_count = 0
    for uploaded_file in files or []:
        if not hasattr(uploaded_file, "getvalue"):
            continue

        image_count += 1
        if image_count > MAX_IMAGES_PER_MESSAGE:
            raise ValueError(f"Maximum {MAX_IMAGES_PER_MESSAGE} images per message.")

        data = uploaded_file.getvalue()
        if len(data) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"{uploaded_file.name} is too large (max 5 MB).")

        media_type = _image_media_type(uploaded_file)
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"{uploaded_file.name}: unsupported type. Use PNG, JPEG, GIF, or WebP."
            )

        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("utf-8"),
            },
        })

    if not blocks:
        raise ValueError("Enter a message or attach at least one image.")

    if blocks[0]["type"] != "text":
        blocks.insert(0, {"type": "text", "text": "Please analyze the attached image(s)."})

    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


def render_message_content(content: str | list) -> None:
    if isinstance(content, str):
        st.markdown(content)
        return
    for block in content:
        if block["type"] == "text":
            st.markdown(block["text"])
        elif block["type"] == "image":
            st.image(base64.b64decode(block["source"]["data"]), use_container_width=True)


def _anthropic_messages():
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg["role"] in ("user", "assistant")
    ]


# --- Supabase helpers ---

def fetch_conversations():
    if not supabase:
        return []
    try:
        data = supabase.table("conversations").select("*").order("created_at", desc=True).execute()
        return data.data or []
    except Exception:
        return []


def create_conversation(title: str) -> str | None:
    if not supabase:
        return None
    try:
        data = supabase.table("conversations").insert({"title": title}).select().execute()
        return data.data[0]["id"]
    except Exception:
        return None


def load_messages(conversation_id: str) -> list:
    if not supabase:
        return []
    try:
        data = (
            supabase.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        return [{"role": row["message_role"], "content": row["message_content"]} for row in data.data]
    except Exception:
        return []


def save_message(conversation_id: str, role: str, content: str | list):
    if not supabase or not conversation_id:
        return
    try:
        if isinstance(content, list):
            content = str(content)
        supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "message_role": role,
            "message_content": content,
        }).execute()
    except Exception:
        pass


def delete_conversation(conversation_id: str):
    if not supabase:
        return
    try:
        supabase.table("conversations").delete().eq("id", conversation_id).execute()
    except Exception:
        pass


def rename_conversation(conversation_id: str, new_title: str):
    if not supabase:
        return
    try:
        supabase.table("conversations").update({"title": new_title}).eq("id", conversation_id).execute()
    except Exception:
        pass


def get_title_from_message(text: str) -> str:
    if isinstance(text, list):
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block["text"]
                break
        else:
            return "New Conversation"
    return text[:50].strip() + ("…" if len(text) > 50 else "")


# --- Claude ---

def get_response():
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY is not set."
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        kwargs = dict(
            model=ANTHROPIC_MODEL,
            max_tokens=_max_output_tokens(ANTHROPIC_MODEL),
            messages=_anthropic_messages(),
            temperature=1,
        )
        if SYSTEM_PROMPT:
            kwargs["system"] = SYSTEM_PROMPT
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except anthropic.APIError as e:
        return f"Error: {e.message}"
    except Exception as e:
        return f"Error: {str(e)}"


# --- Session state init ---

if st.session_state.get("app_version") != APP_VERSION:
    st.session_state.clear()
    st.session_state.app_version = APP_VERSION

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "conversations" not in st.session_state:
    st.session_state.conversations = fetch_conversations()
if "editing_conversation_id" not in st.session_state:
    st.session_state.editing_conversation_id = None


# --- Sidebar ---

with st.sidebar:
    st.title("Conversations")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.active_conversation_id = None
        st.session_state.messages = []
        st.session_state.editing_conversation_id = None
        st.rerun()

    st.divider()

    for convo in st.session_state.conversations:
        cid = convo["id"]
        label = convo.get("title") or "New Conversation"
        is_active = cid == st.session_state.active_conversation_id
        is_editing = cid == st.session_state.editing_conversation_id

        if is_editing:
            new_title = st.text_input(
                "Rename", value=label, key=f"rename_input_{cid}", label_visibility="collapsed"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"save_{cid}", use_container_width=True):
                    rename_conversation(cid, new_title.strip() or label)
                    st.session_state.editing_conversation_id = None
                    st.session_state.conversations = fetch_conversations()
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"cancel_{cid}", use_container_width=True):
                    st.session_state.editing_conversation_id = None
                    st.rerun()
        else:
            col_title, col_edit, col_del = st.columns([7, 1, 1])
            with col_title:
                if st.button(
                    label, key=f"conv_{cid}", use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    if not is_active:
                        st.session_state.active_conversation_id = cid
                        st.session_state.messages = load_messages(cid)
                        st.session_state.editing_conversation_id = None
                        st.rerun()
            with col_edit:
                if st.button("✏️", key=f"edit_{cid}"):
                    st.session_state.editing_conversation_id = cid
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{cid}"):
                    delete_conversation(cid)
                    if is_active:
                        st.session_state.active_conversation_id = None
                        st.session_state.messages = []
                    st.session_state.conversations = fetch_conversations()
                    st.rerun()


# --- Main chat area ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message_content(msg["content"])

if not st.session_state.messages and not st.session_state.active_conversation_id:
    st.markdown("### Hello! How can I help you today?")
    st.caption("Start typing below to begin a new conversation.")

chat_submission = st.chat_input(
    "Message AI Assistant…",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "gif", "webp"],
)

if chat_submission is not None:
    if isinstance(chat_submission, ChatInputValue):
        user_text = chat_submission.text or ""
        uploaded_files = chat_submission.files or []
    else:
        user_text = chat_submission
        uploaded_files = []

    try:
        user_content = build_user_content(user_text, uploaded_files)
    except ValueError as err:
        st.error(str(err))
    else:
        # Create a new conversation on first message
        if not st.session_state.active_conversation_id:
            title = get_title_from_message(user_content)
            convo_id = create_conversation(title)
            st.session_state.active_conversation_id = convo_id
            st.session_state.conversations = fetch_conversations()

        convo_id = st.session_state.active_conversation_id

        st.session_state.messages.append({"role": "user", "content": user_content})
        save_message(convo_id, "user", user_content)

        with st.chat_message("user"):
            render_message_content(user_content)

        with st.chat_message("assistant"):
            response = get_response()
            "".join(char for char in st.write_stream(type_writer(response)))
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_message(convo_id, "assistant", response)
