import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextCleaner:
    """Comprehensive text cleaning for IT ticket preprocessing."""
    
    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self._download_nltk_data()
        
        # Load stopwords for both languages
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.english_stopwords = set()
            
        # Common Hindi stopwords (basic set)
        self.hindi_stopwords = {
            'और', 'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'है', 'हैं', 'था', 'थे', 'थी',
            'होगा', 'होगी', 'होंगे', 'कि', 'जो', 'यह', 'वह', 'इस', 'उस', 'एक', 'दो', 'तीन',
            'कुछ', 'सब', 'कोई', 'कहा', 'कहे', 'कहते', 'बहुत', 'अधिक', 'कम', 'ज्यादा'
        }
        
        # IT-specific terms to preserve
        self.preserve_terms = {
            'server', 'database', 'network', 'email', 'password', 'login', 'error', 'bug',
            'crash', 'slow', 'down', 'offline', 'online', 'backup', 'restore', 'update',
            'install', 'uninstall', 'virus', 'malware', 'firewall', 'vpn', 'wifi', 'internet',
            'browser', 'application', 'software', 'hardware', 'printer', 'scanner', 'monitor'
        }
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str, language: str = 'en') -> str:
        """
        Comprehensive text cleaning pipeline.
        
        Args:
            text: Raw input text
            language: Language code ('en' or 'hi')
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Step 1: Basic cleaning
        text = self._basic_clean(text)
        
        # Step 2: Remove URLs and emails
        text = self._remove_urls_emails(text)
        
        # Step 3: Handle special IT terms
        text = self._normalize_it_terms(text)
        
        # Step 4: Remove extra whitespace
        text = self._normalize_whitespace(text)
        
        # Step 5: Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        # Step 6: Remove stopwords if specified
        if self.remove_stopwords:
            text = self._remove_stopwords(text, language)
            
        # Step 7: Final cleanup
        text = self._final_cleanup(text)
        
        return text.strip()
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text
    
    def _remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        return text
    
    def _normalize_it_terms(self, text: str) -> str:
        """Normalize common IT terms and abbreviations."""
        # Common IT term normalizations
        replacements = {
            r'\bpc\b': 'computer',
            r'\blaptop\b': 'computer',
            r'\bdesktop\b': 'computer',
            r'\bwifi\b': 'wireless',
            r'\binternet\b': 'network',
            r'\bapp\b': 'application',
            r'\bpwd\b': 'password',
            r'\blogin\b': 'login',
            r'\blogon\b': 'login',
            r'\bsign in\b': 'login',
            r'\bsign on\b': 'login',
            r'\berr\b': 'error',
            r'\bbug\b': 'error',
            r'\bissue\b': 'problem',
            r'\bprob\b': 'problem'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_stopwords(self, text: str, language: str) -> str:
        """Remove stopwords based on language."""
        try:
            tokens = word_tokenize(text)
            
            if language == 'hi':
                stopwords_set = self.hindi_stopwords
            else:
                stopwords_set = self.english_stopwords
                
            # Keep IT-specific terms even if they're in stopwords
            filtered_tokens = [
                token for token in tokens 
                if token.lower() not in stopwords_set or token.lower() in self.preserve_terms
            ]
            
            return ' '.join(filtered_tokens)
            
        except Exception as e:
            logger.warning(f"Stopword removal failed: {str(e)}, returning original text")
            return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup operations."""
        # Remove standalone punctuation
        text = re.sub(r'\s+[^\w\s]\s+', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (less than 2 characters) except important ones
        important_short = {'it', 'is', 'in', 'on', 'at', 'to', 'of', 'or', 'no'}
        tokens = text.split()
        tokens = [token for token in tokens if len(token) >= 2 or token.lower() in important_short]
        
        return ' '.join(tokens)
    
    def batch_clean(self, texts: List[str], language: str = 'en') -> List[str]:
        """
        Clean multiple texts in batch.
        
        Args:
            texts: List of texts to clean
            language: Language code
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text, language) for text in texts]
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0
            }
            
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }