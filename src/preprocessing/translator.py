from googletrans import Translator
import time
import random
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextTranslator:
    """Translates text between English and Hindi using Google Translate."""
    
    def __init__(self):
        self.translator = Translator()
        self.max_retries = 3
        self.base_delay = 1
        
    def translate_to_english(self, text: str, source_lang: str = 'hi') -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated text in English
        """
        if not text or not text.strip():
            return text
            
        # If already English, return as is
        if source_lang == 'en':
            return text
            
        return self._translate_with_retry(text, source_lang, 'en')
    
    def translate_to_hindi(self, text: str, source_lang: str = 'en') -> str:
        """
        Translate text to Hindi.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated text in Hindi
        """
        if not text or not text.strip():
            return text
            
        # If already Hindi, return as is
        if source_lang == 'hi':
            return text
            
        return self._translate_with_retry(text, source_lang, 'hi')
    
    def _translate_with_retry(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text with retry logic for robustness.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or original text if translation fails
        """
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                
                result = self.translator.translate(
                    text, 
                    src=source_lang, 
                    dest=target_lang
                )
                
                if result and result.text:
                    logger.info(f"Successfully translated from {source_lang} to {target_lang}")
                    return result.text
                else:
                    logger.warning(f"Empty translation result for attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Translation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    logger.error("All translation attempts failed, returning original text")
                    return text
                    
        return text
    
    def batch_translate(self, texts: list, source_lang: str, target_lang: str) -> list:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        translated_texts = []
        
        for i, text in enumerate(texts):
            try:
                translated = self._translate_with_retry(text, source_lang, target_lang)
                translated_texts.append(translated)
                
                # Add delay between translations to avoid rate limiting
                if i < len(texts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to translate text {i}: {str(e)}")
                translated_texts.append(text)  # Use original text if translation fails
                
        return translated_texts
    
    def is_translation_needed(self, text: str, source_lang: str, target_lang: str) -> bool:
        """
        Check if translation is needed.
        
        Args:
            text: Input text
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            True if translation is needed, False otherwise
        """
        return source_lang != target_lang and text and text.strip()