from __future__ import annotations

import base64
import os
import time

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.elements.widgets.chat import ChatInputValue
from supabase import create_client

st.set_page_config(page_title="AI Assistant", layout="centered")

APP_VERSION = "general-purpose-v1"

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


def _max_output_tokens(model: str) -> int:
    """Anthropic requires max_tokens; use each model's maximum output cap."""
    model_lower = model.lower()
    if "opus" in model_lower:
        return 8192
    return 8192  # Sonnet, Haiku, and other current models

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 5

SYSTEM_PROMPT = ""


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
    """Build Anthropic message content from chat text and uploaded images."""
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
                f"{uploaded_file.name}: unsupported type. "
                "Use PNG, JPEG, GIF, or WebP."
            )

        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(data).decode("utf-8"),
                },
            }
        )

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
            st.image(
                base64.b64decode(block["source"]["data"]),
                use_container_width=True,
            )


def _anthropic_messages():
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg["role"] in ("user", "assistant")
    ]


def load_conversation_history():
    if not supabase:
        return
    try:
        data = supabase.table("conversations").select("*").order("created_at", desc=False).execute()
        for row in data.data:
            st.session_state.messages.append({
                "role": row["message_role"],
                "content": row["message_content"]
            })
    except Exception as e:
        st.warning(f"Could not load conversation history: {str(e)}")


def save_message_to_db(role: str, content: str):
    if not supabase:
        return
    try:
        if isinstance(content, list):
            content = str(content)
        supabase.table("conversations").insert({
            "message_role": role,
            "message_content": content
        }).execute()
    except Exception as e:
        st.warning(f"Could not save message: {str(e)}")


def get_response():
    if not ANTHROPIC_API_KEY:
        return (
            "Error: ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file in the project folder."
        )
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


if st.session_state.get("app_version") != APP_VERSION:
    st.session_state.clear()
    st.session_state.app_version = APP_VERSION

if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    load_conversation_history()
    st.session_state.initialized = True

with st.container():
    if not st.session_state.messages:
        first_message = """Hello! I'm an AI assistant. How can I help you today?

You can ask me anything, or attach an image and I'll analyze it for you.
"""
        st.session_state.messages.append({"role": "assistant", "content": first_message})
        with st.chat_message("assistant"):
            "".join(char for char in st.write_stream(type_writer(first_message)))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message_content(msg["content"])

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
        st.session_state.messages.append({"role": "user", "content": user_content})
        save_message_to_db("user", user_content)

        with st.chat_message("user"):
            render_message_content(user_content)

        with st.chat_message("assistant"):
            response = get_response()
            "".join(char for char in st.write_stream(type_writer(response)))
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            save_message_to_db("assistant", response)
