"""
Base abstract class for translator implementations.
All translator providers should inherit from this.
"""

from abc import ABC, abstractmethod

class BaseTranslator(ABC):
    """Abstract base class for translation providers"""
    
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            source_lang: Source language code (e.g., 'en', 'hi', 'ta')
            target_lang: Target language code (e.g., 'en', 'hi', 'ta')
        
        Returns:
            Translated text
        
        Raises:
            Exception: If translation fails
        """
        pass
    
    @abstractmethod
    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if translator supports the language pair"""
        pass
    
    def get_supported_languages(self) -> list:
        """Return list of supported language codes"""
        pass
