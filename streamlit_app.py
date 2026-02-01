import re
import io
from datetime import datetime
import streamlit as st
import pandas as pd
from assisstants.extractor.fields_extractor import ExtractFields
from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.loader.model_loader import ModelLoader
from assisstants.voice.voice import speech_to_text
from assisstants.logging.logger import logging
from assisstants.exception.exception import AssisstantException
import speech_recognition as sr
import sys
import time


# ========================= VOICE MODEL CONSTANTS & CACHING ========================= #
TARGET_FIELDS_ORDER = ["Name", "Phone Number", "Amount", "Account Number"]
FIELD_KEY_MAP = {
    "Name": "name",
    "Phone Number": "phone_number",
    "Amount": "amount",
    "Account Number": "account_number",
}
CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource(show_spinner=False)
def load_models():
    return ModelLoader.get_model(), ModelLoader.get_tokenizer()

@st.cache_resource(show_spinner=False)
def get_classifier():
    return TextClassifier()

@st.cache_resource(show_spinner=False)
def get_extractor():
    return ExtractFields()

@st.cache_resource(show_spinner=False)
def get_text_processor():
    return TextProcessor()


# ========================= VOICE MODEL HELPER FUNCTIONS ========================= #
def capture_speech_blocking(duration: int = 5):
    """Capture speech synchronously using speech_to_text."""
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
        st.session_state.voice_last_error = str(e)
        return "[API Error]"


def canonicalize_label(lbl: str | None):
    """Normalize classifier output to standard field names."""
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


def process_and_extract_voice(text: str):
    """Process text, classify field type, and extract entity value."""
    processor = get_text_processor()
    classifier = get_classifier()
    extractor = get_extractor()

    processed = processor.process_text(text)
    
    # Classification
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
    except Exception as e:
        logging.error(f"Classification error: {e}")

    label = canonicalize_label(str(raw_label) if raw_label is not None else None)

    # Defensive rule: filter question-like inputs
    try:
        tstrip = text.strip() if isinstance(text, str) else ""
        if label == 'Name':
            if re.search(r"^(where|what|who|why|how|when)\b", tstrip, re.I) or tstrip.endswith('?'):
                logging.info("Input looks like a question; overriding label to None")
                label = None
    except Exception:
        pass

    # Confidence threshold check
    if raw_confidence is not None and raw_confidence < CONFIDENCE_THRESHOLD:
        logging.info(f"Low confidence ({raw_confidence:.3f}) - treating as unknown")
        label = None

    # Heuristic fallback
    if label is None:
        t = processed.lower()
        if re.search(r"\b(account|ac number|a/c)\b", t) or re.search(r"\b\d{10,18}\b", t):
            label = 'Account Number'
        elif re.search(r"\b(rupees|rs|inr|dollar|usd|amount|pay)\b", t):
            label = 'Amount'
        elif re.search(r"\b(phone|mobile|contact|call)\b", t) or re.search(r"\b\d{10}\b", t):
            label = 'Phone Number'

    # Extraction
    entity = None
    extractor_result = None
    if label:
        try:
            extractor_result = extractor.extract(label, text, processed_text=processed)
        except TypeError:
            try:
                model, tokenizer = load_models()
                extractor_result = extractor.extract(label, processed, model=model, tokenizer=tokenizer)
            except Exception as e:
                logging.error(f"Extractor error: {e}")
        except Exception as e:
            logging.error(f"Extractor error: {e}")

    # Normalize extraction result
    if extractor_result:
        if isinstance(extractor_result, str):
            entity = extractor_result.strip()
        elif isinstance(extractor_result, (list, tuple)):
            for item in extractor_result:
                if isinstance(item, str) and item.strip():
                    entity = item.strip()
                    break
        elif isinstance(extractor_result, dict):
            for k in ('entity', 'value', 'text'):
                if k in extractor_result and extractor_result[k]:
                    entity = str(extractor_result[k]).strip()
                    break

    # Fallback extraction rules
    if label and (not entity or entity == ""):
        if label == 'Phone Number':
            m = re.search(r"(\+?\d[\d\s-]{8,}\d)", text)
            if m:
                entity = re.sub(r"[\s-]", "", m.group(1))
        elif label == 'Amount':
            m = re.search(r"(?:rs\.?|inr|usd|dollars|rupees)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)", text, re.IGNORECASE)
            if m:
                entity = m.group(1).replace(',', '')
        elif label == 'Account Number':
            m = re.search(r"\b\d{6,18}\b", text)
            if m:
                entity = m.group(0)
        elif label == 'Name':
            name_pattern = re.search(r"(?:my name is|i am|this is|myself)\s+([a-zA-Z']+(?:\s+[a-zA-Z']+){0,2})", text, re.IGNORECASE)
            if name_pattern:
                entity = name_pattern.group(1).strip().title()
            if not entity:
                raw_tokens = [t for t in text.split() if re.match(r"[A-Za-z]", t)]
                caps = [w for w in raw_tokens if w and w[0].isupper()]
                if len(caps) >= 1:
                    entity = ' '.join(caps[:2]).strip()

    return processed, label, entity


def get_field_errors():
    errors = {}
    required_fields = [
        'First Name',
        'Last Name',
        'Phone Number',
        'Amount',
        'Account Number'
    ]
    for field in required_fields:
        if not st.session_state.fields.get(field, '').strip():
            errors[field] = f"{field} is required"
    return errors



def generate_receipt_text():
    """Generate receipt text for download."""
    receipt_lines = [
        "=" * 50,
        "VAFA BANKING TRANSACTION RECEIPT",
        "=" * 50,
        f"Transaction Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        "-" * 50,
    ]
    
    required_fields = ['First Name', 'Last Name', 'Phone Number', 'Account Number', 'Amount', 'Bank Branch', 'Transaction Type']
    for field in required_fields:
        value = st.session_state.fields.get(field, 'N/A')
        receipt_lines.append(f"{field:.<35} {value}")
    
    receipt_lines.extend([
        "-" * 50,
        "Thank you for using VAFA!",
        "=" * 50,
    ])
    return '\n'.join(receipt_lines)


def init_state():

    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'transaction_type' not in st.session_state:
        st.session_state.transaction_type = 'Deposit'
    if 'fields' not in st.session_state:
        st.session_state.fields = {
            'First Name': '',
            'Last Name': '',
            'Account Number': '',
            'Phone Number': '',
            'Amount': '',
            'Bank Branch': 'SBI Sehore',
            'Transaction Type': 'Deposit'
        }
    if 'previous_page' not in st.session_state:
        st.session_state.previous_page = 'Home'
    if 'nav_visible' not in st.session_state:
        st.session_state.nav_visible = False
    
    # Voice model state
    if 'voice_captured_text' not in st.session_state:
        st.session_state.voice_captured_text = ""
    if 'voice_predicted_label' not in st.session_state:
        st.session_state.voice_predicted_label = None
    if 'voice_extracted_entity' not in st.session_state:
        st.session_state.voice_extracted_entity = None
    if 'voice_pending_retry' not in st.session_state:
        st.session_state.voice_pending_retry = False
    if 'voice_last_error' not in st.session_state:
        st.session_state.voice_last_error = None
    if 'voice_capture_duration' not in st.session_state:
        st.session_state.voice_capture_duration = 5
    if 'voice_debug_mode' not in st.session_state:
        st.session_state.voice_debug_mode = False
    if 'show_errors' not in st.session_state:
        st.session_state.show_errors = False


def set_page(p, remember_prev=True):
    if remember_prev:
        st.session_state.previous_page = st.session_state.page
    st.session_state.page = p


def go_back():
    set_page(st.session_state.previous_page, remember_prev=False)


def add_styles():
    css = """
    <style>
    /* Main background and text colors */
        /* ===== CENTER & LIMIT APP WIDTH ===== */
    section.main > div {
        max-width: 1100px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    body, .main {background: #f0f4f9 !important;}
    
    /* Titles and headings */
    .big-title {font-size:48px; font-weight:800; color:#0052CC; text-align:center; font-family: 'Segoe UI', Roboto, Arial; margin-bottom:10px;}
    .lead {font-size:20px; color:#000000; text-align:center; margin-bottom:30px; font-weight:500;}
    .main-container {max-width:960px; margin:18px auto;}
    .accent {color:#0066FF; font-weight:700}
    .center {text-align:center}
    
    /* Cards with blue gradient */
    .card {background:linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding:18px; border-radius:12px; box-shadow:0 6px 16px rgba(0, 82, 204, 0.1); border-left: 4px solid #0052CC;}
    
    /* Language selection text */
    .lang-instruction {font-size:24px; font-weight:700; color:#000000; text-align:center; margin: 30px 0 20px 0;}
    
    /* Navigation styling (text-only items) */
    .nav-header {background-color: #0052CC; color: white; padding: 16px; text-align: center; font-weight: 800;}
    .nav-link {display:block; padding: 14px 18px; color: #000000 !important; text-decoration: none; background: transparent; border-bottom: 1px solid rgba(0,0,0,0.06);}
    .nav-link:hover {background: rgba(0,82,204,0.06); color: #000000 !important}
    .nav-link.active {background: rgba(0,82,204,0.12); font-weight:800;}

    /* Inputs and textarea */
    input[type="text"], textarea, .stTextInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #90CAF9 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    input[type="text"]:focus, textarea:focus, .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #0052CC !important;
        box-shadow: 0 0 8px rgba(0, 82, 204, 0.3) !important;
    }

    /* Placeholder text style */
    input::placeholder, textarea::placeholder,
    .stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder,
    ::-webkit-input-placeholder, ::-moz-placeholder, :-ms-input-placeholder, ::-ms-input-placeholder {
        color: #64B5F6 !important;
        opacity: 1 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* Buttons - Primary Blue */
    button, .stButton>button {
        background: linear-gradient(90deg, #0052CC 0%, #0066FF 100%) !important;
        color: #ffffff !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        min-height: 50px !important;
    }
    
    button:hover, .stButton>button:hover {
        background: linear-gradient(90deg, #003DA5 0%, #0052CC 100%) !important;
        box-shadow: 0 4px 12px rgba(0, 82, 204, 0.3) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {background-color: #E3F2FD;}
    .stTabs [data-baseweb="tab"] {color: #000000 !important; font-weight: 700;}
    .stTabs [aria-selected="true"] {background-color: #0052CC; color: white;}

    /* Radio and labels */
    .stRadio label, .stRadio .main label {font-size:16px !important; color: #000000 !important; font-weight: 600;}
    
    /* Subheaders */
    .stSubheader {color: #0052CC !important; font-weight: 800 !important;}

    /* Markdown text and general text */
    .stMarkdown {color: #000000 !important;}
    .stWrite {color: #000000 !important;}
    .stText {color: #000000 !important;}
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {color: #000000 !important;}

    /* ================== FORCE SELECTBOX DROPDOWN TO LIGHT THEME ================== */

    div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 2px solid #90CAF9 !important;
}
    /* Entire dropdown portal */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] * {
        background-color: #ffffff !important;
    }

    /* Dropdown list container */
    div[role="listbox"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 16px rgba(0, 82, 204, 0.25) !important;
    }

    /* Each dropdown option */
    div[role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Hovered option */
    div[role="option"]:hover {
        background-color: #E3F2FD !important;
        color: #000000 !important;
    }

    /* Selected option */
    div[role="option"][aria-selected="true"] {
        background-color: #BBDEFB !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Remove dark gradient overlays */
    div[data-baseweb="menu"] {
        background-image: none !important;
    }

    /* Fix scrollbar background inside dropdown */
    div[role="listbox"]::-webkit-scrollbar {
        background-color: #ffffff !important;
    }

    /* ===== ERROR BORDER FOR INVALID INPUTS ===== */
    .input-error input {
        border: 2px solid #d32f2f !important;
        box-shadow: 0 0 6px rgba(211, 47, 47, 0.35) !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def page_header():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA — Streamlined Form UI (prototype)</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead">A colorful multi-page prototype with manual and voice-driven entry.</div>', unsafe_allow_html=True)


def placeholder_extractor(text: str):
    out = {'Name': '', 'Phone': '', 'Account': '', 'Amount': ''}
    if not text:
        return out
    phone_match = re.search(r"\b[6-9]\d{9}\b", text)
    if phone_match:
        out['Phone'] = phone_match.group(0)
    acc_match = re.search(r"\b\d{11,18}\b", text)
    if acc_match:
        out['Account'] = acc_match.group(0)
    amt_match = re.search(r"(?i)(?:rs\.?\s?|rupees\s?)?([\d,]{1,20}(?:\.\d+)?)|([0-9]+\s*(?:lakh|crore|thousand|k)?)", text)
    if amt_match:
        amt = amt_match.group(1) or amt_match.group(2)
        out['Amount'] = amt.strip()
    name_match = re.search(r"(?i)(?:my name is|i am|this is)\s+([A-Z]?[a-z]{2,}(?:\s+[A-Z]?[a-z]{2,})?)", text)
    if name_match:
        out['Name'] = name_match.group(1).strip()
    
    return out


def page_home():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead">Smart Banking Form Management System</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="lang-instruction">Select Language / भाषा चुनें</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('English', key='home_lang_en', use_container_width=True):
            st.session_state.language = 'en'
            set_page('TransactionType')
    with col2:
        if st.button('हिन्दी', key='home_lang_hi', use_container_width=True):
            st.session_state.language = 'hi'
            set_page('TransactionType')
    
    st.markdown('</div>', unsafe_allow_html=True)


def page_transaction_type():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA</div>', unsafe_allow_html=True)
    
    if st.session_state.language == 'en':
        st.markdown('<div class="lead">Select Transaction Type</div>', unsafe_allow_html=True)
        deposit_label = 'Deposit'
        withdraw_label = 'Withdraw'
        back_label = '← Back'
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(deposit_label, key='trans_deposit', use_container_width=True):
                st.session_state.transaction_type = 'Deposit'
                set_page('Form')
        with col2:
            if st.button(withdraw_label, key='trans_withdraw', use_container_width=True):
                st.session_state.transaction_type = 'Withdraw'
                set_page('Form')
    else:  # Hindi
        st.markdown('<div class="lead">लेनदेन का प्रकार चुनें</div>', unsafe_allow_html=True)
        back_label = '← वापस'
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('जमा करें', key='trans_deposit_hi', use_container_width=True):
                st.session_state.transaction_type = 'Deposit'
                set_page('Form')
        with col2:
            if st.button('निकालें', key='trans_withdraw_hi', use_container_width=True):
                st.session_state.transaction_type = 'Withdraw'
                set_page('Form')
    
    st.write('')
    if st.button(back_label, use_container_width=True):
        go_back()
    
    st.markdown('</div>', unsafe_allow_html=True)


def page_form():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA</div>', unsafe_allow_html=True)
    
    # Set language-specific labels
    if st.session_state.language == 'en':
        title = f"{st.session_state.transaction_type} Form"
        manual_label = 'Manual Form'
        voice_label = 'Voice Recording'
        first_name_label = 'First Name'
        last_name_label = 'Last Name'
        account_label = 'Account Number'
        phone_label = 'Phone Number'
        amount_label = f"{st.session_state.transaction_type} Amount (₹)"
        desc_label = 'Description (Optional)'
        manual_desc = f'Manually enter your {st.session_state.transaction_type.lower()} details'
        voice_desc = 'Use voice recording to fill the form'
        proceed_btn = 'Proceed to Confirmation'
        clear_btn = 'Clear Form'
    else:  # Hindi
        title = f"{st.session_state.transaction_type} फॉर्म"
        manual_label = 'मैनुअल फॉर्म'
        voice_label = 'वॉयस रिकॉर्डिंग'
        first_name_label = 'पहला नाम'
        last_name_label = 'अंतिम नाम'
        account_label = 'खाता संख्या'
        phone_label = 'फोन नंबर'
        amount_label = f"{st.session_state.transaction_type} राशि (₹)"
        desc_label = 'विवरण (वैकल्पिक)'
        manual_desc = f'अपने {st.session_state.transaction_type.lower()} विवरण को मैनुअल रूप से दर्ज करें'
        voice_desc = 'फॉर्म भरने के लिए वॉयस रिकॉर्डिंग का उपयोग करें'
        proceed_btn = 'पुष्टि के लिए आगे बढ़ें'
        clear_btn = 'फॉर्म साफ़ करें'
    
    st.markdown(f'<div class="lead">{title}</div>', unsafe_allow_html=True)
    
    # Field progress counter
    filled_fields = sum(1 for field in ['First Name', 'Last Name', 'Phone Number', 'Account Number', 'Amount'] if st.session_state.fields.get(field, '').strip())
    total_required = 5
    progress_text = f"Form Progress: {filled_fields}/{total_required} fields filled"
    st.progress(filled_fields / total_required, text=progress_text)
    
    tabs = st.tabs([manual_label, voice_label])
    
    # Manual Form Tab
    with tabs[0]:
        st.markdown(f'<div class="card">{manual_desc}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.fields['First Name'] = st.text_input(first_name_label, value=st.session_state.fields.get('First Name', ''), key='manual_fname')
        with col2:
            st.session_state.fields['Last Name'] = st.text_input(last_name_label, value=st.session_state.fields.get('Last Name', ''), key='manual_lname')
        
        st.session_state.fields['Account Number'] = st.text_input(account_label, value=st.session_state.fields.get('Account Number', ''), key='manual_acc')
        st.session_state.fields['Phone Number'] = st.text_input(phone_label, value=st.session_state.fields.get('Phone Number', ''), key='manual_phone')
        st.session_state.fields['Amount'] = st.text_input(amount_label, value=st.session_state.fields.get('Amount', ''), key='manual_amt')
         
         # Bank Branch field
        if st.session_state.language == 'en':
            branch_label = 'Bank Branch'
        else:
            branch_label = 'बैंक शाखा'
        
        st.session_state.fields['Bank Branch'] = st.selectbox(
            branch_label,
            options=['SBI Sehore', 'SBI Bhopal', 'SBI Indore'],
            index=0 if st.session_state.fields.get('Bank Branch', 'SBI Sehore') == 'SBI Sehore' else 1,
            key='manual_branch'
        )
    
    # Voice Recording Tab - Intelligent Voice-Activated Form Assistant
    with tabs[1]:
        st.markdown(f'<div class="card">{voice_desc}</div>', unsafe_allow_html=True)
        st.info('🎤 Voice-Activated Form: Speak your banking details. System will classify and extract fields automatically.')
        
        # Debug mode toggle
        st.checkbox("Debug Mode (Show Processing Details)", key='voice_debug_mode')
        st.slider("Recording duration (seconds)", min_value=2, max_value=12, key="voice_capture_duration")
        
        # ===== CAPTURE SECTION =====
        st.subheader("1. Capture Speech")
        if st.button("🎤 Capture Speech", type="primary", use_container_width=True):
            with st.spinner("Listening..."):
                text = capture_speech_blocking(st.session_state.voice_capture_duration)
            st.session_state.voice_captured_text = text
            
            if text in ("[Unrecognized Speech]", "[API Error]", ""):
                st.session_state.voice_pending_retry = True
                st.warning("Speech not recognized. Please try again.")
            else:
                with st.spinner("Processing & extracting..."):
                    processed, label, entity = process_and_extract_voice(text)
                st.session_state.voice_predicted_label = label
                st.session_state.voice_extracted_entity = entity
                st.session_state.voice_pending_retry = False
                logging.info(f"Captured: label={label} entity={entity}")
        
        if st.session_state.voice_captured_text:
            st.markdown("**Transcript:**")
            st.info(st.session_state.voice_captured_text)
        
        # ===== RETRY SECTION =====
        if st.session_state.voice_pending_retry:
            if st.button("🔁 Try Again"):
                st.session_state.voice_captured_text = ""
                st.session_state.voice_predicted_label = None
                st.session_state.voice_extracted_entity = None
                st.session_state.voice_pending_retry = False
                st.rerun()
        
        # ===== EXTRACTION TABLE & CONFIRMATION =====
        if not st.session_state.voice_pending_retry and st.session_state.voice_predicted_label:
            label = st.session_state.voice_predicted_label
            entity = st.session_state.voice_extracted_entity
            
            st.subheader("2. Confirm & Add to Form")
            
            # Display extraction in table format
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.markdown("**Field Name**")
                st.text(label)
            with col2:
                st.markdown("**Extracted Value**")
                st.text(entity if entity else "[No value extracted]")
            with col3:
                st.markdown("**Status**")
                st.success("✓ Extracted" if entity else "⚠ Pending")
            
            st.divider()
            
            # Manual entry if no entity extracted
            manual_entry = ""
            if not entity:
                st.warning("⚠️ No value could be extracted automatically. Please enter manually:")
                manual_entry = st.text_input("Enter value manually", key="voice_manual_entry", placeholder=f"e.g., Enter {label}...")
            
            # Confirmation buttons
            col_confirm, col_reject, col_tip = st.columns([1, 1, 2])
            
            with col_confirm:
                if st.button("✅ Confirm & Add to Form", key="voice_confirm_entity", use_container_width=True):
                    final_entity = manual_entry if (not entity and manual_entry) else entity
                    if final_entity:
                        # Map label to form field
                        if label == 'Name':
                            parts = final_entity.split()
                            st.session_state.fields['First Name'] = parts[0] if parts else final_entity
                            if len(parts) > 1:
                                st.session_state.fields['Last Name'] = ' '.join(parts[1:])
                        elif label == 'Phone Number':
                            st.session_state.fields['Phone Number'] = final_entity
                        elif label == 'Amount':
                            st.session_state.fields['Amount'] = final_entity
                        elif label == 'Account Number':
                            st.session_state.fields['Account Number'] = final_entity
                        
                        st.session_state.voice_captured_text = ""
                        st.session_state.voice_predicted_label = None
                        st.session_state.voice_extracted_entity = None
                        st.success("✓ Field added to form!")
                        time.sleep(1)
                        st.rerun()
            
            with col_reject:
                if st.button("❌ Reject", key="voice_retry_entity", use_container_width=True):
                    st.session_state.voice_captured_text = ""
                    st.session_state.voice_predicted_label = None
                    st.session_state.voice_extracted_entity = None
                    st.session_state.voice_pending_retry = False
                    st.info("Recording cleared. Try capturing again.")
                    time.sleep(1)
                    st.rerun()
            
            with col_tip:
                st.caption("💡 For better results, speak: 'My name is...', 'Phone is...', 'Amount...'")
        
        # ===== DEBUG OUTPUT =====
        if st.session_state.voice_debug_mode and st.session_state.voice_captured_text:
            with st.expander('🔍 Debug Output', expanded=False):
                if st.session_state.voice_predicted_label:
                    st.write({
                        'raw_input': st.session_state.voice_captured_text,
                        'predicted_label': st.session_state.voice_predicted_label,
                        'extracted_entity': st.session_state.voice_extracted_entity,
                    })
        
        st.divider()
        
        # ===== FORM REVIEW SECTION =====
        if st.session_state.language == 'en':
            st.subheader('3. Review & Edit Form Fields')
        else:
            st.subheader('3. फॉर्म फील्ड की समीक्षा करें')
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.fields['First Name'] = st.text_input(first_name_label, value=st.session_state.fields.get('First Name', ''), key='voice_fname')
        with col2:
            st.session_state.fields['Last Name'] = st.text_input(last_name_label, value=st.session_state.fields.get('Last Name', ''), key='voice_lname')
        
        st.session_state.fields['Account Number'] = st.text_input(account_label, value=st.session_state.fields.get('Account Number', ''), key='voice_acc')
        st.session_state.fields['Phone Number'] = st.text_input(phone_label, value=st.session_state.fields.get('Phone Number', ''), key='voice_phone')
        st.session_state.fields['Amount'] = st.text_input(amount_label, value=st.session_state.fields.get('Amount', ''), key='voice_amt')
        
        # Bank Branch field
        if st.session_state.language == 'en':
            branch_label = 'Bank Branch'
        else:
            branch_label = 'बैंक शाखा'
        
        st.session_state.fields['Bank Branch'] = st.selectbox(
            branch_label,
            options=['SBI Sehore', 'SBI Bhopal', 'SBI Indore'],
            index=0 if st.session_state.fields.get('Bank Branch', 'SBI Sehore') == 'SBI Sehore' else 1,
            key='voice_branch'
        )
    
    # Set transaction type in fields
    st.session_state.fields['Transaction Type'] = st.session_state.transaction_type
    
    st.write('')
    col1, col2 = st.columns(2)
    with col1:
        if st.button(clear_btn, use_container_width=True):
            st.session_state.fields = {
                'First Name': '',
                'Last Name': '',
                'Account Number': '',
                'Phone Number': '',
                'Amount': '',
                'Bank Branch': 'SBI Sehore',
                'Transaction Type': '',
                'Description': ''
            }
            st.rerun()
    with col2:
        if st.button(proceed_btn, use_container_width=True):
            set_page('Confirm')
    
    st.write('')
    if st.session_state.language == 'en':
        back_label = '← Back'
    else:
        back_label = '← वापस'
    
    if st.button(back_label, use_container_width=True):
        go_back()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
def render_required_input(label, field_key, widget_key, errors):
    has_error = field_key in errors and st.session_state.show_errors

    if has_error:
        st.markdown("<div class='input-error'>", unsafe_allow_html=True)

    st.session_state.fields[field_key] = st.text_input(
        label,
        value=st.session_state.fields.get(field_key, ''),
        key=widget_key
    )

    if has_error:
        # st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:#d32f2f;font-size:14px;margin-top:-6px;'>❌ {field_key} is required</div>",
            unsafe_allow_html=True
        )

def page_confirm():
   

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA</div>', unsafe_allow_html=True)
    field_errors = get_field_errors()
    if st.session_state.language == 'en':
        st.markdown('<div class="lead">Confirmation / Edit</div>', unsafe_allow_html=True)
        confirm_desc = 'Review and edit transaction details below. All fields marked with * are required.'
        back_btn = 'Back to Form'
        confirm_btn = 'Confirm & Generate Receipt'
        back_label = '←'
    else:  # Hindi
        st.markdown('<div class="lead">पुष्टि / संपादित करें</div>', unsafe_allow_html=True)
        confirm_desc = 'नीचे लेनदेन विवरण की समीक्षा करें और संपादित करें। * से चिह्नित सभी फील्ड आवश्यक हैं।'
        back_btn = 'फॉर्म पर वापस जाएं'
        confirm_btn = 'पुष्टि करें और पावती उत्पन्न करें'
        back_label = '← वापस'
    
    

    
    st.markdown(f'<div class="card">{confirm_desc}</div>', unsafe_allow_html=True)
    
    # Edit fields
    required_fields = {
        'First Name': 'First Name *',
        'Last Name': 'Last Name *',
        'Phone Number': 'Phone Number *',
        'Account Number': 'Account Number *',
        'Amount': 'Amount *',
    }
    
    optional_fields = {
        'Bank Branch': 'Bank Branch',
        'Transaction Type': 'Transaction Type',
    }
    
    # Required fields
    if st.session_state.language == 'en':
        st.subheader("Required Fields")
    else:
        st.subheader("आवश्यक फील्ड")
    
    

    col1, col2 = st.columns(2)

    with col1:
        render_required_input('First Name *', 'First Name', 'confirm_fname', field_errors)
        render_required_input('Phone Number *', 'Phone Number', 'confirm_phone', field_errors)
        render_required_input('Amount *', 'Amount', 'confirm_amount', field_errors)

    with col2:
        render_required_input('Last Name *', 'Last Name', 'confirm_lname', field_errors)
        render_required_input('Account Number *', 'Account Number', 'confirm_account', field_errors)

    st.divider()
    
    # Optional fields
    if st.session_state.language == 'en':
        st.subheader("Optional Fields")
    else:
        st.subheader("वैकल्पिक फील्ड")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.fields['Bank Branch'] = st.selectbox(
            'Bank Branch',
            options=['SBI Sehore', 'SBI Bhopal', 'SBI Indore'],
            index=0 if st.session_state.fields.get('Bank Branch', 'SBI Sehore') == 'SBI Sehore' else (1 if st.session_state.fields.get('Bank Branch') == 'SBI Bhopal' else 2),
            key='confirm_branch'
        )
        st.session_state.fields['Transaction Type'] = st.selectbox(
            'Transaction Type',
            options=['Deposit', 'Withdrawal'],
            index=0 if st.session_state.fields.get('Transaction Type', 'Deposit') == 'Deposit' else 1,
            key='confirm_type'
        )
    
    with col2:
        pass
    
    
    st.divider()
    
    # Show errors if any
    if st.button(confirm_btn, use_container_width=True):
        st.session_state.show_errors = True
        if not field_errors:
            st.session_state.confirmed_at = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            set_page('Receipt')

    
    st.write('')
    if st.button(back_label, use_container_width=True, key='confirm_back_label_btn'):
        go_back()
    
    st.markdown('</div>', unsafe_allow_html=True)


def page_receipt():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">VAFA</div>', unsafe_allow_html=True)
    
    if st.session_state.language == 'en':
        st.markdown('<div class="lead">Transaction Receipt</div>', unsafe_allow_html=True)
        receipt_title = 'VAFA BANKING TRANSACTION RECEIPT'
        success_msg = '✅ Transaction Confirmed Successfully!'
        download_txt = 'Download Receipt (TXT)'
        print_btn = '🖨 Print Receipt'
        new_transaction = 'New Transaction'
        back_label = '← Back'
    else:  # Hindi
        st.markdown('<div class="lead">लेनदेन पावती</div>', unsafe_allow_html=True)
        receipt_title = 'VAFA बैंकिंग लेनदेन पावती'
        success_msg = '✅ लेनदेन सफलतापूर्वक पुष्टि किया गया!'
        download_txt = 'पावती डाउनलोड करें (TXT)'
        print_btn = '🖨 पावती प्रिंट करें'
        new_transaction = 'नया लेनदेन'
        back_label = '← वापस'
    
    st.success(success_msg)
    
    st.divider()
    
    # Display receipt in formatted table with proper styling
    st.subheader("Receipt Details")
    
    receipt_data = {
        'Field': [],
        'Value': []
    }
    
    required_fields_order = ['First Name', 'Last Name', 'Phone Number', 'Account Number', 'Amount', 'Bank Branch', 'Transaction Type']
    
    for field in required_fields_order:
        value = st.session_state.fields.get(field, 'N/A')
        receipt_data['Field'].append(field)
        receipt_data['Value'].append(value)
    
    # Add confirmation timestamp
    receipt_data['Field'].append('Confirmed At')
    receipt_data['Value'].append(st.session_state.get('confirmed_at', datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    
    # Display as formatted table with CSS styling
    df_receipt = pd.DataFrame(receipt_data)
    
    # Custom styling for the table
    st.markdown(
        """
        <style>
        .receipt-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #ffffff;
            color: #000000;
        }
        .receipt-table th {
            background-color: #0052CC;
            color: #ffffff;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .receipt-table td {
            background-color: #ffffff;
            color: #000000;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .receipt-table tr:hover {
            background-color: #f5f9ff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display table using HTML for better control
    html_table = df_receipt.to_html(classes='receipt-table', index=False, escape=False)
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.divider()
    
    # Generate text receipt
    receipt_text = generate_receipt_text()
    
    # Download buttons
    st.subheader("Download & Print Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label=download_txt,
            data=receipt_text.encode('utf-8'),
            file_name=f"VAFA_Receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime='text/plain',
            use_container_width=True
        )
    
    with col2:
        # HTML version for better printing
        html_receipt = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; font-weight: bold; font-size: 18px; margin-bottom: 20px; }}
                .receipt-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .receipt-table td {{ border: 1px solid #ddd; padding: 10px; }}
                .receipt-table tr:nth-child(odd) {{ background-color: #f9f9f9; }}
                .label {{ font-weight: bold; width: 40%; }}
                .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">{receipt_title}</div>
            <table class="receipt-table">
        """
        
        for field, value in zip(receipt_data['Field'], receipt_data['Value']):
            html_receipt += f"<tr><td class='label'>{field}:</td><td>{value}</td></tr>"
        
        html_receipt += """
            </table>
            <div class="footer">
                <p>Thank you for using VAFA - Voice Activated Form Assistant</p>
                <p>This is an automated receipt. Please keep it for your records.</p>
            </div>
        </body>
        </html>
        """
        
        st.download_button(
            label="⬇️ Download Receipt (HTML)",
            data=html_receipt.encode('utf-8'),
            file_name=f"VAFA_Receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime='text/html',
            use_container_width=True
        )
    
    with col3:
        if st.button(print_btn, use_container_width=True, key='receipt_print_btn'):
            st.info("📋 Use your browser's print dialog (Ctrl+P / Cmd+P) to print this receipt.")
            st.markdown(
                """
                <script>
                    window.print();
                </script>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(new_transaction, use_container_width=True, type="primary", key='receipt_new_transaction'):
            st.session_state.fields = {
                'First Name': '',
                'Last Name': '',
                'Account Number': '',
                'Phone Number': '',
                'Amount': '',
                'Bank Branch': 'SBI Sehore',
                'Transaction Type': 'Deposit',
                'Description': ''
            }
            # Reset voice state
            st.session_state.voice_captured_text = ""
            st.session_state.voice_predicted_label = None
            st.session_state.voice_extracted_entity = None
            st.session_state.voice_pending_retry = False
            set_page('Home')
    with col2:
        if st.button(back_label, use_container_width=True, key='receipt_back_btn'):
            go_back()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_nav():
    # Navigation links (text-only, no header)
    nav_pages = ['Home', 'Form', 'Receipt']

    for page_name in nav_pages:
        is_active = page_name == st.session_state.page
        active_class = 'active' if is_active else ''
        st.markdown(
            f"""
            <a href="?page={page_name}" class="nav-link {active_class}" style="display:block; padding:14px 18px; color: #000000; text-decoration: none; font-weight: {'800' if is_active else '600'}; background-color: transparent; border-bottom:1px solid rgba(0,0,0,0.06);">
                {page_name}
            </a>
            """,
            unsafe_allow_html=True
        )

    # Language display
    st.markdown(
        f"""
        <div style="padding: 12px 18px; margin-top: 12px; color: #000000; text-align: left; font-weight: 600; font-size: 13px;">
            Language: <strong>{st.session_state.language.upper()}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title='VAFA UI Prototype', layout='wide')
    init_state()
    add_styles()

    # If page is provided via query params (from nav anchor links), update page
    params = st.experimental_get_query_params()
    if 'page' in params and isinstance(params.get('page'), list) and params.get('page'):
        requested = params.get('page')[0]
        valid = ['Home', 'TransactionType', 'Form', 'Confirm', 'Receipt']
        if requested in valid and requested != st.session_state.page:
            # don't remember previous when jumping via link
            set_page(requested, remember_prev=False)

    # Top control: menu toggle and back button side by side
    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button('☰', key='menu_toggle', help='Toggle Navigation'):
            st.session_state.nav_visible = not st.session_state.nav_visible
    with cols[1]:
        if st.session_state.page != 'Home':
            if st.button('←', key='back_btn', help='Go Back'):
                go_back()
    with cols[2]:
        pass

    # Layout: show left nav column when visible
    if st.session_state.nav_visible:
        left, main_col = st.columns([1, 4])
        with left:
            render_nav()
        with main_col:
            if st.session_state.page == 'Home':
                page_home()
            elif st.session_state.page == 'TransactionType':
                page_transaction_type()
            elif st.session_state.page == 'Form':
                page_form()
            elif st.session_state.page == 'Receipt':
                page_receipt()
    else:
        # Full-width main content when nav is hidden
        if st.session_state.page == 'Home':
            page_home()
        elif st.session_state.page == 'TransactionType':
            page_transaction_type()
        elif st.session_state.page == 'Form':
            page_form()
        elif st.session_state.page == 'Confirm':
            page_confirm()
        elif st.session_state.page == 'Receipt':
            page_receipt()


if __name__ == '__main__':
    main()
