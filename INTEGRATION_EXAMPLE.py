"""
Example: How to integrate multi-language support into VAFA app.py

This file shows the modifications needed to support Hindi and other languages.
Copy these modifications into your existing app.py
"""

# ============= ADD THESE IMPORTS AT TOP =============
from assisstants.translator.huggingface_translator import HuggingFaceTranslator

# ============= ADD THIS LANGUAGE CONFIGURATION =============
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
}

# ============= MODIFY: Cache translator resource =============
@st.cache_resource(show_spinner=False)
def load_translator():
    """Load translator model once and cache it"""
    return HuggingFaceTranslator()

# ============= ADD THIS FUNCTION: Multi-language processing =============
def process_and_extract_multilingual(raw_text: str, user_language: str):
    """
    Process user input in their language and extract fields.
    
    Pipeline:
    1. If user speaks Hindi/Tamil/etc, translate to English
    2. Process & classify (using English-trained model)
    3. Extract field values
    4. Optionally translate results back to user language
    5. Return all intermediate steps
    
    Args:
        raw_text: User's input in their selected language
        user_language: Language name (e.g., 'Hindi', 'Tamil')
    
    Returns:
        dict with original_text, english_text, label, confidence, extracted_value, etc.
    """
    try:
        result = {
            "original_text": raw_text,
            "user_language": user_language,
            "english_text": raw_text,
            "label": None,
            "confidence": 0.0,
            "extracted_value": None,
            "translated_back": None,
        }
        
        # Step 1: Translate to English if needed
        if user_language != "English":
            translator = load_translator()
            lang_code = SUPPORTED_LANGUAGES[user_language]
            logging.info(f"Translating from {user_language} ({lang_code}) to English")
            
            result["english_text"] = translator.translate(
                raw_text, 
                source_lang=lang_code, 
                target_lang="en"
            )
            st.info(f"🔄 Translated to English: {result['english_text']}")
        
        # Step 2: Process & Classify (English)
        processor = TextProcessor()
        classifier = TextClassifier()
        
        processed_text = processor.process_text(result["english_text"])
        label, confidence = classifier.classify(processed_text, result["english_text"])
        
        result["label"] = label
        result["confidence"] = confidence
        logging.info(f"Classified as: {label} (confidence: {confidence:.2%})")
        
        # Step 3: Extract field value
        if confidence >= CONFIDENCE_THRESHOLD:
            extractor = ExtractFields()
            extracted = extractor.extract(label, result["english_text"], processed_text)
            result["extracted_value"] = extracted
            logging.info(f"Extracted value: {extracted}")
            
            # Step 4: Translate extracted value back to user language
            if user_language != "English" and extracted:
                translator = load_translator()
                result["translated_back"] = translator.translate(
                    str(extracted),
                    source_lang="en",
                    target_lang=SUPPORTED_LANGUAGES[user_language]
                )
        
        return result
    
    except Exception as e:
        logging.error(f"Error in multilingual processing: {str(e)}")
        st.error(f"Processing error: {str(e)}")
        return None

# ============= MODIFY: Streamlit UI to add language selector =============
def main():
    """Modified main function with language support"""
    st.set_page_config(page_title="VAFA - Multi-Language", layout="wide")
    st.title("🌍 Voice Activated Form Assistant (Multi-Language)")
    
    # Load models
    model, tokenizer = load_models()
    
    # ========== NEW: Language Selection ==========
    col1, col2 = st.columns([1, 1])
    
    with col1:
        user_language = st.selectbox(
            "Select Your Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=0  # Default to English
        )
    
    with col2:
        st.info(f"Selected: {user_language}")
    
    # ========== MODIFIED: Input section ==========
    st.subheader("📝 Input Method")
    
    input_method = st.radio("Choose input method:", ["Text Input", "Voice Input"])
    
    user_input = None
    
    if input_method == "Text Input":
        user_input = st.text_area(
            f"Enter text in {user_language}:",
            placeholder=f"e.g., 'My name is Raj' or 'मेरा नाम राज है'"
        )
    else:
        # Voice input (existing code)
        if st.button("🎤 Start Recording"):
            with st.spinner("Recording..."):
                user_input = speech_to_text()
                if user_input:
                    st.success(f"Recorded: {user_input}")
    
    # ========== MODIFIED: Processing ==========
    if user_input and st.button("▶ Process"):
        with st.spinner("Processing your input..."):
            result = process_and_extract_multilingual(user_input, user_language)
            
            if result:
                st.success("✅ Processing completed!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Original", "Processing", "Results"])
                
                with tab1:
                    st.write(f"**Language**: {result['user_language']}")
                    st.write(f"**Original Input**: {result['original_text']}")
                
                with tab2:
                    st.write(f"**English Translation**: {result['english_text']}")
                    st.write(f"**Classified As**: {result['label']}")
                    st.write(f"**Confidence**: {result['confidence']:.2%}")
                
                with tab3:
                    if result['extracted_value']:
                        st.write(f"**Extracted Value (English)**: {result['extracted_value']}")
                        if result['translated_back']:
                            st.write(f"**Extracted Value ({user_language})**: {result['translated_back']}")
                    else:
                        st.warning("No field value extracted")

# ============= FEATURE: Show supported languages =============
def show_supported_languages():
    """Display supported language information"""
    st.sidebar.title("ℹ️ Supported Languages")
    
    st.sidebar.markdown("""
    ### Direct Support (Indian Languages)
    - **Hindi** (हिंदी) - hi
    - **Tamil** (தமிழ்) - ta
    - **Telugu** (తెలుగు) - te
    - **Kannada** (ಕನ್ನಡ) - kn
    - **Marathi** (मराठी) - mr
    - **Bengali** (বাংলা) - bn
    - **Gujarati** (ગુજરાતી) - gu
    
    ### Other Supported
    - English, Spanish, French, German, etc.
    """)

# ============= UPDATED: Form storage with language tracking =============
def store_field_data(field_name: str, value: str, language: str):
    """
    Store field data with language information.
    Allows multi-language forms.
    """
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    st.session_state.form_data[field_name] = {
        "value": value,
        "language": language,
        "timestamp": datetime.now().isoformat()
    }

# ============= RUN APP =============
if __name__ == "__main__":
    main()
    show_supported_languages()
