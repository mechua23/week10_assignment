from datetime import datetime
import json
from pathlib import Path
import time
import uuid

import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")


HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REQUEST_TIMEOUT_SECONDS = 30
CHAT_HISTORY_HEIGHT = 520
SIDEBAR_CHAT_LIST_HEIGHT = 540
CHATS_DIR = Path("chats")
MEMORY_FILE = Path("memory.json")
MEMORY_EXTRACTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, friendly AI assistant."
MEMORY_EXTRACTION_PROMPT = (
    "Given the user's latest message, extract durable personal facts or preferences as a JSON object. "
    "Prefer stable traits such as name, language, interests, favorite topics, communication style, "
    "or other useful long-term preferences. Do not include temporary requests. If there are no useful "
    "personal facts, return {}. Return valid JSON only."
)


def load_hf_token():
    """Safely load the Hugging Face token from Streamlit secrets."""
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None, (
            "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
            "and restart the app."
        )

    if not str(token).strip():
        return None, (
            "Your Hugging Face token is empty. Set `HF_TOKEN` in "
            "`.streamlit/secrets.toml` and restart the app."
        )

    return str(token).strip(), None


def post_chat_completion(token, payload, stream=False):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        return None, "The request to Hugging Face timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return (
            None,
            "Unable to reach Hugging Face right now. Check your network connection and try again.",
        )
    except requests.exceptions.RequestException:
        return None, "Something went wrong while contacting Hugging Face. Please try again."

    if response.status_code in (401, 403):
        response.close()
        return None, "The Hugging Face token is invalid or unauthorized."
    if response.status_code == 429:
        response.close()
        return None, "The Hugging Face API rate limit was reached. Please wait and try again."
    if not response.ok:
        try:
            error_payload = response.json()
            error_message = error_payload.get("error", {}).get("message")
        except ValueError:
            error_message = None

        if error_message:
            response.close()
            return None, f"Hugging Face returned an error: {error_message}"

        response.close()
        return (
            None,
            f"Hugging Face returned an error ({response.status_code}). Please try again later.",
        )

    return response, None


def start_chat_stream(token, messages):
    """Start a streamed chat-completions request to the Hugging Face Inference Router."""
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": True,
    }
    return post_chat_completion(token, payload, stream=True)


def extract_stream_text(event_data):
    choices = event_data.get("choices")
    if not choices:
        return ""

    delta = choices[0].get("delta", {})
    content = delta.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "".join(text_parts)

    return ""


def stream_response_chunks(response):
    saw_text = False

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue

            data = raw_line[5:].strip()
            if data == "[DONE]":
                break

            try:
                event_data = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("The API returned malformed streaming data.") from exc

            chunk = extract_stream_text(event_data)
            if not chunk:
                continue

            saw_text = True
            yield chunk
            time.sleep(0.02)
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The streamed response timed out. Please try again.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "The streamed response was interrupted. Check your network connection and try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Something went wrong while reading the streamed response.") from exc
    finally:
        response.close()

    if not saw_text:
        raise ValueError("The API returned an unusable streamed response.")


def current_timestamp():
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(timestamp):
    return datetime.fromisoformat(timestamp).strftime("%b %d, %I:%M %p")


def ensure_chats_dir():
    CHATS_DIR.mkdir(exist_ok=True)


def load_memory_from_disk():
    try:
        content = MEMORY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    if not content:
        return {}

    try:
        memory = json.loads(content)
    except json.JSONDecodeError:
        return {}

    return memory if isinstance(memory, dict) else {}


def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def merge_memory(existing, new):
    merged = dict(existing)

    for key, value in new.items():
        current_value = merged.get(key)

        if isinstance(current_value, dict) and isinstance(value, dict):
            merged[key] = merge_memory(current_value, value)
        elif isinstance(current_value, list) and isinstance(value, list):
            merged_list = list(current_value)
            for item in value:
                if item not in merged_list:
                    merged_list.append(item)
            merged[key] = merged_list
        else:
            merged[key] = value

    return merged


def build_memory_system_prompt(memory):
    if not memory:
        return DEFAULT_SYSTEM_PROMPT

    memory_json = json.dumps(memory, indent=2)
    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        "Use the following saved user memory to personalize responses when it is relevant. "
        "Do not mention the memory unless it fits naturally, and do not invent details.\n"
        f"User memory:\n{memory_json}"
    )


def build_conversation_messages(chat_messages, memory):
    return [{"role": "system", "content": build_memory_system_prompt(memory)}] + chat_messages


def parse_assistant_content(data):
    choices = data.get("choices")
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        combined_text = "".join(text_parts).strip()
        if combined_text:
            return combined_text

    return None


def parse_json_object(text):
    candidate = text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    start_index = candidate.find("{")
    end_index = candidate.rfind("}")
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None

    candidate = candidate[start_index : end_index + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def extract_memory_from_message(token, user_message):
    payload = {
        "model": MEMORY_EXTRACTION_MODEL,
        "messages": [
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }

    response, error = post_chat_completion(token, payload, stream=False)
    if error:
        return None

    try:
        data = response.json()
    except ValueError:
        response.close()
        return None
    finally:
        response.close()

    content = parse_assistant_content(data)
    if not content:
        return None

    return parse_json_object(content)


def chat_file_path(chat_id):
    return CHATS_DIR / f"{chat_id}.json"


def sort_chats(chats):
    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    return chats


def is_valid_message(message):
    return (
        isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    )


def normalize_chat(chat):
    if not isinstance(chat, dict):
        return None

    chat_id = chat.get("id")
    title = chat.get("title")
    created_at = chat.get("created_at")
    updated_at = chat.get("updated_at")
    messages = chat.get("messages")

    if not all(isinstance(value, str) and value.strip() for value in [chat_id, title, created_at, updated_at]):
        return None
    if not isinstance(messages, list) or not all(is_valid_message(message) for message in messages):
        return None

    try:
        datetime.fromisoformat(created_at)
        datetime.fromisoformat(updated_at)
    except ValueError:
        return None

    return {
        "id": chat_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
    }


def load_chats_from_disk():
    ensure_chats_dir()
    chats = []

    for file_path in CHATS_DIR.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                chat = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        normalized_chat = normalize_chat(chat)
        if normalized_chat is not None:
            chats.append(normalized_chat)

    return sort_chats(chats)


def save_chat(chat):
    ensure_chats_dir()
    file_path = chat_file_path(chat["id"])
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(chat, file, indent=2)


def delete_chat_file(chat_id):
    file_path = chat_file_path(chat_id)
    try:
        file_path.unlink()
    except FileNotFoundError:
        pass


def build_chat_title(messages):
    for message in messages:
        if message["role"] == "user" and message["content"].strip():
            return message["content"].strip()[:30]
    return "New Chat"


def create_chat(initial_messages=None):
    messages = initial_messages or []
    timestamp = current_timestamp()
    return {
        "id": str(uuid.uuid4()),
        "title": build_chat_title(messages),
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": messages,
    }


def get_active_chat():
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state["chats"]:
        if chat["id"] == active_chat_id:
            return chat
    return None


def set_active_chat(chat_id):
    st.session_state["active_chat_id"] = chat_id


def add_new_chat():
    chat = create_chat()
    st.session_state["chats"].insert(0, chat)
    save_chat(chat)
    set_active_chat(chat["id"])


def delete_chat(chat_id):
    chats = st.session_state["chats"]
    active_chat_id = st.session_state.get("active_chat_id")
    remaining_chats = [chat for chat in chats if chat["id"] != chat_id]
    st.session_state["chats"] = remaining_chats
    delete_chat_file(chat_id)

    if active_chat_id == chat_id:
        if remaining_chats:
            set_active_chat(remaining_chats[0]["id"])
        else:
            new_chat = create_chat()
            st.session_state["chats"] = [new_chat]
            save_chat(new_chat)
            set_active_chat(new_chat["id"])


def initialize_session_state():
    if "chats" not in st.session_state:
        existing_messages = st.session_state.pop("messages", [])
        saved_chats = load_chats_from_disk()

        if saved_chats:
            st.session_state["chats"] = saved_chats
            st.session_state["active_chat_id"] = saved_chats[0]["id"]
        else:
            initial_chat = create_chat(existing_messages)
            st.session_state["chats"] = [initial_chat]
            st.session_state["active_chat_id"] = initial_chat["id"]
            save_chat(initial_chat)
    elif "active_chat_id" not in st.session_state:
        first_chat = st.session_state["chats"][0] if st.session_state["chats"] else None
        st.session_state["active_chat_id"] = first_chat["id"] if first_chat else None

    if "memory" not in st.session_state:
        st.session_state["memory"] = load_memory_from_disk()


def render_chat_history():
    history_container = st.container(height=CHAT_HISTORY_HEIGHT)
    return history_container


def render_sidebar():
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", use_container_width=True):
            add_new_chat()
            st.rerun()

        chat_list_container = st.container(height=SIDEBAR_CHAT_LIST_HEIGHT)
        with chat_list_container:
            if not st.session_state["chats"]:
                st.caption("No chats yet. Create one to get started.")
                return

            for chat in st.session_state["chats"]:
                is_active = chat["id"] == st.session_state.get("active_chat_id")
                title_col, delete_col = st.columns([5, 1])
                button_type = "primary" if is_active else "secondary"

                with title_col:
                    if st.button(
                        chat["title"],
                        key=f"chat_select_{chat['id']}",
                        use_container_width=True,
                        type=button_type,
                    ):
                        set_active_chat(chat["id"])
                with delete_col:
                    if st.button("✕", key=f"chat_delete_{chat['id']}", use_container_width=True):
                        delete_chat(chat["id"])
                        st.rerun()

                timestamp_label = format_timestamp(chat["updated_at"])
                if is_active:
                    st.caption(f"Active • {timestamp_label}")
                else:
                    st.caption(timestamp_label)

        with st.expander("User Memory", expanded=True):
            if st.button("Clear Memory", use_container_width=True):
                st.session_state["memory"] = {}
                save_memory(st.session_state["memory"])
                st.rerun()
            st.json(st.session_state["memory"])


def main():
    initialize_session_state()
    st.session_state["chats"] = sort_chats(st.session_state["chats"])
    render_sidebar()

    st.title("My AI Chat")

    token, token_error = load_hf_token()
    if token_error:
        st.error(token_error)
        return

    history_container = render_chat_history()
    active_chat = get_active_chat()

    if active_chat is None:
        st.info("No active chat. Create a new chat from the sidebar to get started.")
        st.chat_input("Type a message and press Enter", disabled=True)
        return

    prompt = st.chat_input("Type a message and press Enter")
    response = None
    api_error = None

    if prompt:
        active_chat["messages"].append({"role": "user", "content": prompt})
        active_chat["title"] = build_chat_title(active_chat["messages"])
        active_chat["updated_at"] = current_timestamp()
        save_chat(active_chat)

        response, api_error = start_chat_stream(
            token,
            build_conversation_messages(active_chat["messages"], st.session_state["memory"]),
        )

        if api_error:
            st.error(api_error)

    with history_container:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt and not api_error:
            with st.chat_message("assistant"):
                try:
                    reply = st.write_stream(stream_response_chunks(response))
                except (RuntimeError, ValueError) as stream_error:
                    st.error(str(stream_error))
                else:
                    if isinstance(reply, str) and reply.strip():
                        active_chat["messages"].append({"role": "assistant", "content": reply})
                        active_chat["updated_at"] = current_timestamp()
                        save_chat(active_chat)

                        extracted_memory = extract_memory_from_message(token, prompt)
                        if extracted_memory:
                            st.session_state["memory"] = merge_memory(
                                st.session_state["memory"], extracted_memory
                            )
                            save_memory(st.session_state["memory"])
                    else:
                        st.error("The API returned an unusable streamed response.")


if __name__ == "__main__":
    main()
