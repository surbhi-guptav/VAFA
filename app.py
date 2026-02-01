"""Streamlit application for Voice Activated Form Assistant workflow.

Flow:
 1. User speaks -> speech captured -> transcript processed
 2. Text is cleaned & classified -> entity extracted
 3. Show predicted field + extracted value for confirmation
 4. User confirms (stores in form) or retries (speak again)
 5. After all target fields filled -> allow edit & download/print

Assumptions:
 - External modules provide: ModelLoader, TextProcessor, TextClassifier, ExtractFields, speech_to_text
 - Classification labels correspond to one of: Name, Phone Number, Amount, Account Number
"""

from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging
import sys
import time
import threading
from typing import Dict, Optional

from assisstants.loader.model_loader import ModelLoader
from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.extractor.fields_extractor import ExtractFields
from assisstants.voice.voice import speech_to_text

import streamlit as st
import re
from dataclasses import dataclass


# --------------------------- Configuration & Constants --------------------------- #
TARGET_FIELDS_ORDER = ["Name", "Phone Number", "Amount", "Account Number"]
FIELD_KEY_MAP = {
    "Name": "name",
    "Phone Number": "phone_number",
    "Amount": "amount",
    "Account Number": "account_number",
}

# Minimum classifier confidence required to accept prediction
CONFIDENCE_THRESHOLD = 0.60


# Caching Heavy Resources 
@st.cache_resource(show_spinner=False)
def load_models():
    model, tokenizer = ModelLoader.get_model(), ModelLoader.get_tokenizer()
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def get_classifier():
    return TextClassifier()

@st.cache_resource(show_spinner=False)
def get_extractor():
    return ExtractFields()

@st.cache_resource(show_spinner=False)
def get_text_processor():
    return TextProcessor()


# Utility Functions
def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.form_data: Dict[str, Optional[str]] = {FIELD_KEY_MAP[f]: "" for f in TARGET_FIELDS_ORDER}
        st.session_state.captured_text = ""
        st.session_state.predicted_label = None
        st.session_state.extracted_entity = None
        st.session_state.history = []  # list of dict entries
        st.session_state.recording = False
        st.session_state.last_error = None
        st.session_state.progress_count = 0
        st.session_state.pending_retry = False
        st.session_state.capture_duration = 5  # seconds default
        st.session_state.active_field_focus = None  # field currently being updated
        logging.info("Session state initialized")


def remaining_fields():
    filled = {k: v for k, v in st.session_state.form_data.items() if v}
    return [f for f in TARGET_FIELDS_ORDER if not st.session_state.form_data[FIELD_KEY_MAP[f]]]


def all_fields_filled():
    return all(st.session_state.form_data[FIELD_KEY_MAP[f]] for f in TARGET_FIELDS_ORDER)


def capture_speech_blocking(duration: int = 5):
    """Capture speech synchronously using provided speech_to_text helper."""
    try:
        stt = speech_to_text()
        stt.start_listening()
        time.sleep(duration)
        stt.stop_listening()
        transcripts = stt.get_transcripts()
        stt.clear_transcripts()
        if not transcripts:
            return ""
        return transcripts[0]
    except Exception as e:
        logging.error(f"Speech capture error: {e}")
        st.session_state.last_error = str(e)
        return "[API Error]"


def process_and_extract(text: str):
    processor = get_text_processor()
    classifier = get_classifier()
    extractor = get_extractor()

    processed = processor.process_text(text)
    # ---- Classification (robust) ---- #
    def canonicalize_label(lbl: str | None):
        if not lbl:
            return None
        l = lbl.lower().strip().replace('_', ' ')
        if 'phone' in l or 'mobile' in l or 'contact' in l:
            return 'Phone Number'
        if 'amount' in l or 'money' in l or 'rupee' in l or 'rs' in l or 'price' in l:
            return 'Amount'
        if 'account' in l or 'acct' in l:
            return 'Account Number'
        if 'name' in l or 'person' in l:
            return 'Name'
        return None

    raw_label = None
    raw_confidence = None
    try:
        raw_label, raw_confidence = classifier.classify(processed, return_prob=True)
    except TypeError:
        try:
            model, tokenizer = load_models()
            raw_label, raw_confidence = classifier.classify(processed, model=model, tokenizer=tokenizer, return_prob=True)
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            raw_label = None
            raw_confidence = None
    except Exception as e:
        logging.error(f"Classification error: {e}")
        raw_label = None
        raw_confidence = None

    label = canonicalize_label(str(raw_label) if raw_label is not None else None)

    # Defensive rule: many casual questions ("where are you going", "how are you")
    # are being misclassified by the model as 'Name'. If the raw text looks like
    # a question (starts with a wh-word or ends with '?'), prefer to treat it as
    # unknown so the UI can ask for clarification instead of extracting a name.
    try:
        tstrip = text.strip() if isinstance(text, str) else ""
        if label == 'Name':
            if re.search(r"^(where|what|who|why|how|when)\b", tstrip, re.I) or tstrip.endswith('?') or re.search(r"\b(where are you|how are you)\b", tstrip, re.I):
                logging.info("Input looks like a question; overriding label to None")
                label = None
    except Exception:
        pass

    # if classifier is not confident enough, treat as unknown
    if raw_confidence is not None and raw_confidence < CONFIDENCE_THRESHOLD:
        logging.info(f"Low confidence ({raw_confidence:.3f}) - treating as unknown")
        label = None

    # Heuristic classification fallback if still None
    if label is None:
        t = processed.lower()
        if re.search(r"\b(account|ac number|a/c)\b", t) or re.search(r"\b\d{10,18}\b", t):
            label = 'Account Number'
        elif re.search(r"\b(rupees|rs|inr|dollar|usd|amount|pay)\b", t):
            label = 'Amount'
        elif re.search(r"\b(phone|mobile|contact|call)\b", t) or re.search(r"\b\d{10}\b", t):
            label = 'Phone Number'
        # Do not default arbitrary alphabetic text to 'Name'.
        # Leave label as None when no clear field evidence is found so the UI can ask for clarification.

    # ---- Extraction (robust) ---- #
    entity = None
    extractor_result = None
    if label:
        try:
            # pass both raw text and processed text to extractor for best results
            extractor_result = extractor.extract(label, text, processed_text=processed)
        except TypeError:
            try:
                model, tokenizer = load_models()
                extractor_result = extractor.extract(label, processed, model=model, tokenizer=tokenizer)
            except Exception as e:
                logging.error(f"Extractor signature mismatch: {e}")
        except Exception as e:
            logging.error(f"Extractor error: {e}")

    # Normalize extractor output (could be str, tuple, dict, list)
    if extractor_result:
        if isinstance(extractor_result, str):
            entity = extractor_result.strip()
        elif isinstance(extractor_result, (list, tuple)):
            # take first non-empty string-like item
            for item in extractor_result:
                if isinstance(item, str) and item.strip():
                    entity = item.strip(); break
                if isinstance(item, dict):
                    for k in ('entity', 'value', 'text'):
                        if k in item and item[k]:
                            entity = str(item[k]).strip(); break
                    if entity: break
        elif isinstance(extractor_result, dict):
            for k in ('entity', 'value', 'text', label.lower().replace(' ', '_')):
                if k in extractor_result and extractor_result[k]:
                    entity = str(extractor_result[k]).strip(); break

    # ---- Fallback extraction rules ---- #
    heuristic_used = False
    fallback_reason = None
    text_for_rules = processed
    if label and (not entity or entity == ""):
        heuristic_used = True
        if label == 'Phone Number':
            m = re.search(r"(\+?\d[\d\s-]{8,}\d)", text_for_rules)
            if m:
                entity = re.sub(r"[\s-]", "", m.group(1))
                fallback_reason = 'regex phone pattern'
        if label == 'Amount' and not entity:
            m = re.search(r"(?:rs\.?|inr|usd|dollars|rupees)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)", text_for_rules, re.IGNORECASE)
            if m:
                entity = m.group(1).replace(',', '')
                fallback_reason = 'regex amount pattern'
        if label == 'Account Number' and not entity:
            m = re.search(r"\b\d{6,18}\b", text_for_rules)
            if m:
                entity = m.group(0)
                fallback_reason = 'regex account pattern'
        if label == 'Name' and not entity:
            # 1. Pattern: my name is / i am / this is / myself <name words>
            name_pattern = re.search(r"(?:my name is|i am|this is|myself)\s+([a-zA-Z']+(?:\s+[a-zA-Z']+){0,2})", text, re.IGNORECASE)
            if name_pattern:
                entity = name_pattern.group(1).strip().title()
                fallback_reason = 'intro phrase pattern'
            # 2. Capitalized tokens from original raw text (keep first 2-3)
            if not entity:
                raw_tokens = [t for t in text.split() if re.match(r"[A-Za-z]", t)]
                caps = [w for w in raw_tokens if w[0].isupper()]
                if len(caps) >= 1:
                    entity = ' '.join(caps[:2]).strip()
                    fallback_reason = 'capitalized raw tokens'
            # 3. Fallback: last two alphabetic tokens (user likely ended with name)
            if not entity and raw_tokens:
                entity = ' '.join(raw_tokens[-2:]).title()
                fallback_reason = 'last tokens heuristic'

    if st.session_state.get('debug_mode'):
        with st.expander('🔍 Debug Output', expanded=True):
            st.write({
                'raw_input': text,
                'processed': processed,
                'raw_label': raw_label,
                'normalized_label': label,
                'extractor_result_type': type(extractor_result).__name__,
                'extractor_result': extractor_result,
                'final_entity': entity,
                'heuristic_used': heuristic_used,
                'fallback_reason': fallback_reason,
            })

    return processed, label, entity


def confirm_entity():
    label = st.session_state.predicted_label
    entity = st.session_state.extracted_entity
    if not (label and entity):
        return
    key = FIELD_KEY_MAP.get(label)
    if key:
        st.session_state.form_data[key] = entity
        st.session_state.history.append({"label": label, "entity": entity})
        st.session_state.captured_text = ""
        st.session_state.predicted_label = None
        st.session_state.extracted_entity = None
        st.session_state.pending_retry = False
        st.session_state.progress_count = sum(1 for f in TARGET_FIELDS_ORDER if st.session_state.form_data[FIELD_KEY_MAP[f]])


def reset_current_capture():
    st.session_state.captured_text = ""
    st.session_state.predicted_label = None
    st.session_state.extracted_entity = None
    st.session_state.pending_retry = False


def reset_field(field_label: str):
    key = FIELD_KEY_MAP[field_label]
    st.session_state.form_data[key] = ""
    st.session_state.progress_count = sum(1 for f in TARGET_FIELDS_ORDER if st.session_state.form_data[FIELD_KEY_MAP[f]])


def download_form_txt():
    lines = [f"{f}: {st.session_state.form_data[FIELD_KEY_MAP[f]]}" for f in TARGET_FIELDS_ORDER]
    return ("Form Summary\n" + "\n".join(lines)).encode("utf-8")


# --------------------------- UI Components --------------------------- #
def inject_custom_css():
    st.markdown(
        """
        <style>
        .main-title {text-align:center; font-size:2.4rem; font-weight:700; background: linear-gradient(90deg,#1d976c,#2f80ed); -webkit-background-clip:text; color:transparent; margin-bottom:0.5rem;}
        .subtitle {text-align:center; font-size:1rem; color:#555; margin-bottom:1.5rem;}
        .entity-box {padding:1rem; border-radius:12px; background:#f5f9ff; border:1px solid #d9ecff;}
        .good-pill {display:inline-block; padding:4px 12px; background:#1d976c; color:#fff; border-radius:20px; font-size:0.8rem; margin-right:6px;}
        .warn-pill {display:inline-block; padding:4px 12px; background:#d9534f; color:#fff; border-radius:20px; font-size:0.8rem; margin-right:6px;}
        .progress-wrapper {margin-top:0.5rem;}
        .field-done {background:#e8fff3; border-left:4px solid #28a745; padding:6px 10px; border-radius:6px;}
        .field-pending {background:#fff7e6; border-left:4px solid #ff9800; padding:6px 10px; border-radius:6px;}
        .footer-note {text-align:center; font-size:0.75rem; color:#888; margin-top:2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown("<div class='main-title'>Voice Activated Form Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Speak each piece of information. Confirm or retry until the form is complete.</div>",
        unsafe_allow_html=True,
    )


def render_progress():
    total = len(TARGET_FIELDS_ORDER)
    completed = sum(1 for f in TARGET_FIELDS_ORDER if st.session_state.form_data[FIELD_KEY_MAP[f]])
    st.progress(completed / total)
    cols = st.columns(total)
    for i, f in enumerate(TARGET_FIELDS_ORDER):
        filled = bool(st.session_state.form_data[FIELD_KEY_MAP[f]])
        with cols[i]:
            st.metric(f, "✔" if filled else "…")


def render_field_status_panel():
    with st.expander("Field Status", expanded=False):
        for f in TARGET_FIELDS_ORDER:
            val = st.session_state.form_data[FIELD_KEY_MAP[f]]
            css = "field-done" if val else "field-pending"
            st.markdown(f"<div class='{css}'><strong>{f}:</strong> {val if val else 'Pending'}</div>", unsafe_allow_html=True)


def render_capture_section():
    st.subheader("1. Capture Speech")
    st.checkbox("Debug mode", key='debug_mode', help="Show internal processing details")
    st.slider("Recording duration (seconds)", min_value=2, max_value=12, key="capture_duration")
    capture_button = st.button("🎤 Capture Speech", type="primary")
    if capture_button:
        with st.spinner("Listening..."):
            text = capture_speech_blocking(st.session_state.capture_duration)
        st.session_state.captured_text = text
        if text in ("[Unrecognized Speech]", "[API Error]", ""):
            st.session_state.pending_retry = True
            st.warning("Speech not recognized. Please try again.")
        else:
            with st.spinner("Processing & extracting..."):
                processed, label, entity = process_and_extract(text)
            st.session_state.predicted_label = label
            st.session_state.extracted_entity = entity
            st.session_state.pending_retry = False
            logging.info(f"Captured: label={label} entity={entity}")

    if st.session_state.captured_text:
        st.markdown("**Transcript:**")
        st.info(st.session_state.captured_text)


def render_confirmation_section():
    if st.session_state.pending_retry:
        if st.button("🔁 Try Again"):
            reset_current_capture()
        return

    label = st.session_state.predicted_label
    entity = st.session_state.extracted_entity
    if not label:
        st.info("No label predicted yet. Try capturing speech again with clearer wording (e.g., 'My name is ...', 'Phone number is ...').")
        return
    manual_entry = None
    if not entity:
        st.warning("Label predicted but no entity extracted. You can enter it manually or retry.")
        manual_entry = st.text_input("Enter value manually", key="manual_entry")
    st.subheader("2. Confirm Extraction")
    st.markdown(
        f"<div class='entity-box'><span class='good-pill'>Predicted Field</span> <strong>{label}</strong><br><span class='good-pill'>Extracted Value</span> <code>{entity}</code></div>",
        unsafe_allow_html=True,
    )
    if not entity:
        st.caption("Tip: Say phrases like 'my phone number is nine eight seven ...' or 'amount is five thousand rupees'.")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("✅ Confirm", key="confirm_entity"):
            if not st.session_state.extracted_entity and manual_entry:
                st.session_state.extracted_entity = manual_entry
            confirm_entity()
            st.success("Saved to form.")
    with c2:
        if st.button("❌ Not Correct", key="retry_entity"):
            reset_current_capture()
            st.info("Please capture again.")
    with c3:
        placeholder = FIELD_KEY_MAP.get(label)
        if placeholder:
            existing = st.session_state.form_data.get(placeholder, "")
            if existing and existing != entity:
                st.caption("Note: This will overwrite the previous value if confirmed.")


def render_form_editor():
    st.subheader("3. Review & Edit Form")
    edited = False
    for f in TARGET_FIELDS_ORDER:
        key = FIELD_KEY_MAP[f]
        current = st.session_state.form_data[key]
        new_val = st.text_input(f, value=current, key=f"edit_{key}")
        if new_val != current:
            st.session_state.form_data[key] = new_val
            edited = True
    if edited:
        st.success("Form updated.")
    st.download_button(
        label="📝 Download TXT",
        data=download_form_txt(),
        file_name="form_summary.txt",
        mime="text/plain",
    )
    if st.button("🖨 Print (Browser Dialog)"):
        st.info("Use your browser's print dialog (Ctrl+P / Cmd+P).")
        st.markdown("<script>window.print()</script>", unsafe_allow_html=True)


def render_reset_options():
    with st.expander("Reset / Advanced"):
        if st.button("Reset Entire Form"):
            for f in TARGET_FIELDS_ORDER:
                st.session_state.form_data[FIELD_KEY_MAP[f]] = ""
            reset_current_capture()
            st.session_state.progress_count = 0
            st.success("Form reset.")
        for f in TARGET_FIELDS_ORDER:
            if st.session_state.form_data[FIELD_KEY_MAP[f]]:
                if st.button(f"Clear {f}", key=f"clear_{f}"):
                    reset_field(f)
                    st.info(f"{f} cleared.")


def main():
    try:
        init_session_state()
        inject_custom_css()
        render_header()
        render_progress()
        render_field_status_panel()

        if not all_fields_filled():
            render_capture_section()
            render_confirmation_section()
        else:
            st.success("All fields captured! You can edit or download below.")
            render_form_editor()

        render_reset_options()
        st.markdown("<div class='footer-note'>Voice Activated Form Assistant • Session based • Experimental UI</div>", unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise AssisstantException(e, sys)


if __name__ == "__main__":
    logging.info("Starting Streamlit Voice Activated Form Assistant UI")
    main()