from langdetect import detect, DetectorFactory
import re
from src.utils.logger import get_logger

# Set seed for consistent results
DetectorFactory.seed = 0

logger = get_logger(__name__)

class LanguageDetector:
    """Detects language of input text with focus on English and Hindi."""
    
    def __init__(self):
        self.supported_languages = {'en': 'english', 'hi': 'hindi'}
        
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code ('en' for English, 'hi' for Hindi, 'unknown' for others)
        """
        if not text or len(text.strip()) < 3:
            logger.warning("Text too short for reliable language detection")
            return 'en'  # Default to English
            
        try:
            # Clean text for better detection
            cleaned_text = self._clean_for_detection(text)
            
            if not cleaned_text:
                return 'en'
                
            detected_lang = detect(cleaned_text)
            
            # Map detected language to supported languages
            if detected_lang in self.supported_languages:
                logger.info(f"Detected language: {self.supported_languages[detected_lang]}")
                return detected_lang
            else:
                logger.info(f"Unsupported language detected: {detected_lang}, defaulting to English")
                return 'en'
                
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}, defaulting to English")
            return 'en'
    
    def _clean_for_detection(self, text: str) -> str:
        """
        Clean text for better language detection.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text suitable for language detection
        """
        # Remove URLs, emails, and special characters that might confuse detection
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)  # Keep alphanumeric, spaces, and Devanagari
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_hindi(self, text: str) -> bool:
        """Check if text is in Hindi."""
        return self.detect_language(text) == 'hi'
    
    def is_english(self, text: str) -> bool:
        """Check if text is in English."""
        return self.detect_language(text) == 'en'
    
    def get_language_confidence(self, text: str) -> dict:
        """
        Get language detection confidence scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language codes and confidence scores
        """
        try:
            from langdetect import detect_langs
            
            cleaned_text = self._clean_for_detection(text)
            if not cleaned_text:
                return {'en': 1.0}
                
            lang_probs = detect_langs(cleaned_text)
            
            result = {}
            for lang_prob in lang_probs:
                if lang_prob.lang in self.supported_languages:
                    result[lang_prob.lang] = lang_prob.prob
                    
            return result if result else {'en': 1.0}
            
        except Exception as e:
            logger.error(f"Language confidence detection failed: {str(e)}")
            return {'en': 1.0}