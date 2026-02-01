"""
Hugging Face Transformers-based translator.
Uses Helsinki-NLP Opus MT models for efficient translation.
Supports 90+ language pairs including Indian languages.
"""

import sys
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class HuggingFaceTranslator:
    """
    Translation using Hugging Face Helsinki-NLP Opus MT models.
    
    Supported Language Codes:
    - en: English
    - hi: Hindi
    - ta: Tamil
    - te: Telugu
    - kn: Kannada
    - mr: Marathi
    - bn: Bengali
    - gu: Gujarati
    - pa: Punjabi
    - ur: Urdu
    """
    
    # Language pair to model name mapping
    MODEL_MAPPING = {
        ("en", "hi"): "Helsinki-NLP/Opus-MT-en-hi",
        ("hi", "en"): "Helsinki-NLP/Opus-MT-hi-en",
        ("en", "ta"): "Helsinki-NLP/Opus-MT-en-ta",
        ("ta", "en"): "Helsinki-NLP/Opus-MT-ta-en",
        ("en", "te"): "Helsinki-NLP/Opus-MT-en-te",
        ("te", "en"): "Helsinki-NLP/Opus-MT-te-en",
        ("en", "kn"): "Helsinki-NLP/Opus-MT-en-kn",
        ("kn", "en"): "Helsinki-NLP/Opus-MT-kn-en",
        ("en", "mr"): "Helsinki-NLP/Opus-MT-en-mr",
        ("mr", "en"): "Helsinki-NLP/Opus-MT-mr-en",
        ("en", "bn"): "Helsinki-NLP/Opus-MT-en-bn",
        ("bn", "en"): "Helsinki-NLP/Opus-MT-bn-en",
        ("en", "gu"): "Helsinki-NLP/Opus-MT-en-gu",
        ("gu", "en"): "Helsinki-NLP/Opus-MT-gu-en",
    }
    
    def __init__(self):
        """Initialize translator with cache for models"""
        self.models_cache = {}  # Store loaded models to avoid reloading
        logging.info("HuggingFaceTranslator initialized")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'hi')
            target_lang: Target language code (e.g., 'hi', 'en')
        
        Returns:
            Translated text
        
        Example:
            translator = HuggingFaceTranslator()
            result = translator.translate("My name is Raj", "en", "hi")
            # Returns: "मेरा नाम राज है"
        """
        try:
            # Skip translation if source and target are same
            if source_lang == target_lang:
                logging.info(f"Source and target languages identical, returning original text")
                return text
            
            # Check if language pair is supported
            if not self.supports_language_pair(source_lang, target_lang):
                raise ValueError(f"Language pair {source_lang}->{target_lang} not supported")
            
            # Get model name for this language pair
            model_name = self.MODEL_MAPPING[(source_lang, target_lang)]
            logging.info(f"Translating from {source_lang} to {target_lang} using {model_name}")
            
            # Load model from cache or download
            if model_name not in self.models_cache:
                logging.info(f"Loading model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.models_cache[model_name] = (tokenizer, model)
            
            tokenizer, model = self.models_cache[model_name]
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                inputs['input_ids'],
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode translation
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Translation successful: '{text}' -> '{translation}'")
            
            return translation
        
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            raise AssisstantException(f"Translation failed: {str(e)}", sys)
    
    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported"""
        return (source_lang, target_lang) in self.MODEL_MAPPING
    
    def get_supported_languages(self) -> list:
        """Get list of unique supported language codes"""
        langs = set()
        for source, target in self.MODEL_MAPPING.keys():
            langs.add(source)
            langs.add(target)
        return sorted(list(langs))
    
    def get_supported_pairs(self) -> dict:
        """Get all supported language pairs"""
        return list(self.MODEL_MAPPING.keys())
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.models_cache.clear()
        logging.info("Model cache cleared")


# Example usage
if __name__ == "__main__":
    translator = HuggingFaceTranslator()
    
    # Test English to Hindi
    english_text = "My name is Raj. I want to transfer 5000 rupees."
    hindi_translation = translator.translate(english_text, "en", "hi")
    print(f"English: {english_text}")
    print(f"Hindi: {hindi_translation}")
    
    # Test Hindi to English
    hindi_text = "मेरा नाम राज है। मैं 5000 रुपये स्थानांतरित करना चाहता हूं।"
    english_back = translator.translate(hindi_text, "hi", "en")
    print(f"\nHindi: {hindi_text}")
    print(f"English: {english_back}")
