import streamlit as st
from evaluater import test
from huggingface_hub import hf_hub_download
import zipfile
from openai import OpenAI
import json
import os
import shutil
import re
from templates.base_template import BASE_TEMPLATE,BASE_SIMILAR_TEMPLATE,BASE_BOTH_TEMPLATE,BASE_GUIDELINE_TEMPLATE

def highlight_terms(text: str, dominant_terms: list[dict]) -> str:

    if not text or not dominant_terms:
        return text

    lines = text.split("\n")
    new_lines = []
    in_list = False

    for line in lines:
        if re.match(r"^\s*\d+\.\s+", line):
            if not in_list:
                new_lines.append("<ol>")
                in_list = True
            content = re.sub(r"^\s*\d+\.\s+", "", line)
            new_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                new_lines.append("</ol>")
                in_list = False
            new_lines.append(line)

    if in_list:
        new_lines.append("</ol>")

    text = "\n".join(new_lines)

    term_dict = {}
    for t in dominant_terms:
        term = (t.get("original_text") or t.get("term") or "").strip()
        raw_score = float(t.get("contribution", 0.0))

        if not term or len(term) < 3:
            continue

        key = term.lower()
        if key not in term_dict or raw_score > term_dict[key]["raw_score"]:
            term_dict[key] = {
                "term": term,
                "raw_score": raw_score
            }

    if not term_dict:
        return text

    sorted_terms = sorted(term_dict.values(), key=lambda x: len(x["term"]), reverse=True)
    highlighted = text

    for item in sorted_terms:
        term  = item["term"]
        raw   = item["raw_score"]

        if raw >= 0.9:
            color = "#8b0000"
            bg    = "#ffe6e6"
            hover_bg = "#ffd6d6"
        elif raw >= 0.85:
            color = "#ff8c00"
            bg    = "#fff2e0"
            hover_bg = "#ffdfb3"
        else:
            continue

        pattern = re.compile(re.escape(term), re.IGNORECASE)

        def _wrap(m):
            original = m.group(0)
            return (
                f"<span title='Contribution: {round(raw, 3)}' "
                f"style='color:{color};"
                f"background-color:{bg};"
                f"padding:2px 6px;"
                f"border-radius:6px;"
                f"font-weight:600;"
                f"cursor:pointer;"
                f"transition: background-color 0.2s ease, box-shadow 0.2s ease;' "
                f"onmouseover=\"this.style.backgroundColor='{hover_bg}'; this.style.boxShadow='0 0 6px rgba(0,0,0,0.25)'\" "
                f"onmouseout=\"this.style.backgroundColor='{bg}'; this.style.boxShadow='none'\">"
                f"{original}</span>"
            )

        highlighted = pattern.sub(_wrap, highlighted)

    return highlighted


def extract_guidelines():
    repo_id = "Cryo3978/medguidelines"
    zip_filename = "guidelines.zip"
    other_filenames = [
        "embeddings.npy",
        "embeddings_choices.npy",
        "index.faiss",
        "index_choices.faiss"
    ]

    target_dir = os.getcwd()

    for file in other_filenames:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        target_path = os.path.join(target_dir, file)
        if not os.path.exists(target_path):
            shutil.copy(cached_path, target_path)

    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_filename,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )

    extract_dir = os.path.join(target_dir, "guidelines")

    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    return extract_dir


def clear_chat():
    st.session_state.messages = []


def initialize_provider_settings(provider_choice):
    provider_configs = {
        "OpenAI": {
            "api_key_source": os.getenv("OPENAI_API_KEY"),
            "base_url_source": "https://api.openai.com/v1",
            "fallback_model": "gpt-4o-mini"
        },
        "Deepseek": {
            "api_key_source": os.getenv("DEEPSEEK_API_KEY"),
            "base_url_source": "https://api.deepseek.com/v1",
            "fallback_model": "deepseek-chat"
        }
    }
    return provider_configs.get(provider_choice, {})


if __name__ == "__main__":

    if "initialized" not in st.session_state:
        extract_guidelines()
        st.session_state.initialized = True

    if "locked_case" not in st.session_state:
        st.session_state.locked_case = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "📋 Patient Case"
    if "enable_similar_case" not in st.session_state:
        st.session_state.enable_similar_case = False
    if "enable_guidelines" not in st.session_state:
        st.session_state.enable_guidelines = False
    if "initial_done" not in st.session_state:
        st.session_state.initial_done = False

    st.title("🩺MED-COPILOT")

    
    with st.sidebar:
        st.markdown("## 🤖 AI Provider")

        available_providers = ["OpenAI", "Deepseek"]

        if "current_provider_choice" not in st.session_state:
            st.session_state.current_provider_choice = available_providers[0]

        provider_selection = st.selectbox(
            "Choose AI Provider:",
            available_providers,
            key="current_provider_choice"
        )

        provider_settings = initialize_provider_settings(provider_selection)

        if not provider_settings.get("api_key_source"):
            st.error(f"Missing API key for {provider_selection}.")
            st.stop()

        api_client = OpenAI(
            api_key=provider_settings["api_key_source"],
            base_url=provider_settings["base_url_source"]
        )

        model_list = sorted([m.id for m in api_client.models.list()])

        session_key = f"model_for_{provider_selection}"
        default_model = provider_settings.get("fallback_model", model_list[0])

        if session_key not in st.session_state:
            st.session_state[session_key] = (
                default_model if default_model in model_list else model_list[0]
            )

        chosen_model = st.selectbox("Model:", model_list, key=session_key)

        test_engine = test(
            OPENAI_API_KEY=provider_settings["api_key_source"],
            model=chosen_model
        )

        st.button("🔄 Reset Conversation", on_click=clear_chat)

    tab_case, tab_chat = st.tabs(["📋 Patient Case", "💬 Chat / Q&A"])

    with tab_case:
        st.session_state.active_tab = "📋 Patient Case"
        st.subheader("Step 1 — Enter and lock the patient case")

        case_input = st.text_area(
            "Patient case / HPI / key findings:",
            height=220,
            value=st.session_state.locked_case or "",
            placeholder="Please enter the patient's case..."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔒 Lock / Update Patient Case"):
                if case_input.strip():

                    st.session_state.locked_case = case_input.strip()
                    st.session_state.initial_done = False
                    st.session_state.enable_similar_case = False
                    st.session_state.enable_guidelines = False

                    full_prompt = f"Patient Case:\n{st.session_state.locked_case}"

                    display_text, response_key_info, dominant_terms = \
                        test_engine.evaluate_SOA_P_with_patientbase_graphrag(full_prompt)

                    st.session_state.latest_patientcase = highlight_terms(
                        display_text,
                        dominant_terms
                    )

                    if isinstance(response_key_info, tuple):
                        st.session_state.latest_guidelines = response_key_info[0].replace("\\n", "\n")
                    else:
                        st.session_state.latest_guidelines = response_key_info.replace("\\n", "\n")

                    st.session_state.initial_done = True
                    st.success("Patient case locked and initial evaluation completed!")

        with col2:
            if st.button("🗑️ Clear Locked Case"):
                st.session_state.locked_case = None
                st.session_state.initial_done = False
                st.info("Locked case cleared.")

    with tab_chat:
        st.session_state.active_tab = "💬 Chat / Q&A"

        st.subheader("Step 2 — Review similar case and guidelines")

        if not st.session_state.locked_case or not st.session_state.initial_done:
            st.warning("Please lock a patient case first.")
            st.stop()

        with st.expander("🧠 Most Similar Patient Case"):
            st.markdown(st.session_state.latest_patientcase, unsafe_allow_html=True)

        with st.expander("📚 Key Medical Guidelines (GraphRAG)"):
            st.markdown(st.session_state.latest_guidelines)

        colA, colB = st.columns(2)

        with colA:
            st.session_state.enable_similar_case = st.checkbox(
                "Use similar case",
                value=st.session_state.enable_similar_case
            )

        with colB:
            st.session_state.enable_guidelines = st.checkbox(
                "Use guidelines",
                value=st.session_state.enable_guidelines
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    if st.session_state.active_tab == "💬 Chat / Q&A":
        user_input = st.chat_input("Ask any follow-up question about this patient...")
    else:
        user_input = None

    if user_input and st.session_state.active_tab == "💬 Chat / Q&A":

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        context_data = {
            "Patient_Case": st.session_state.locked_case,
            "Similar_Patients": st.session_state.latest_patientcase,
            "Key_Info": st.session_state.latest_guidelines,
            "User_Input": user_input,
        }

        if not st.session_state.enable_similar_case and not st.session_state.enable_guidelines:
            template = BASE_TEMPLATE
        elif st.session_state.enable_similar_case and not st.session_state.enable_guidelines:
            template = BASE_SIMILAR_TEMPLATE
        elif not st.session_state.enable_similar_case and st.session_state.enable_guidelines:
            template = BASE_GUIDELINE_TEMPLATE
        else:
            template = BASE_BOTH_TEMPLATE

        required_keys = set(re.findall(r"\{(.*?)\}", template))
        filtered_context = {key: context_data[key] for key in required_keys if key in context_data}
        full_prompt = template.format(**filtered_context)

        gpt_response = api_client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You are a medical assistant who answers based on the provided context."},
                {"role": "user", "content": full_prompt}
            ]
        )

        reply = gpt_response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
