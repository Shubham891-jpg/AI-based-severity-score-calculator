import numpy as np
import joblib
import os
import sys
import warnings
from typing import List

# Suppress scikit-learn model unpickling version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.language_detector import LanguageDetector
from src.features.embeddings import MultilingualEmbeddings
from src.scoring.severity_scaler import SeverityScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SeverityPredictor:
    """Predict severity scores for IT tickets."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor with trained model components.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.embeddings = None
        self.text_cleaner = None
        self.language_detector = None
        self.embeddings_model_name = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all trained model components."""
        try:
            logger.info("Loading model components...")
            
            # Load main model
            model_path = os.path.join(self.model_dir, "severity_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "severity_scaler.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.scaler = SeverityScaler()
            self.scaler.load_scaler(scaler_path)
            
            # Load embeddings info
            embeddings_info_path = os.path.join(self.model_dir, "embeddings_info.pkl")
            if os.path.exists(embeddings_info_path):
                embeddings_info = joblib.load(embeddings_info_path)
                self.embeddings_model_name = embeddings_info['model_name']
            else:
                self.embeddings_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
                
            # Initialize preprocessing components
            self.text_cleaner = TextCleaner(remove_stopwords=True, lowercase=True)
            self.language_detector = LanguageDetector()
            
            logger.info("All model components loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            raise
    
    def _ensure_embeddings_loaded(self):
        """Lazy load embeddings model and warm up weights."""
        if self.embeddings is None:
            logger.info("Initializing embeddings model...")
            self.embeddings = MultilingualEmbeddings(model_name=self.embeddings_model_name)
            self.embeddings._load_model()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        try:
            # Detect language
            language = self.language_detector.detect_language(text)
            
            # Clean text
            cleaned_text = self.text_cleaner.clean_text(text, language)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {str(e)}, using original text")
            return text
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from preprocessed text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Feature vector
        """
        try:
            # Ensure embeddings are loaded
            self._ensure_embeddings_loaded()
            
            # Get embeddings
            features = self.embeddings.encode_single_text(text)
            
            # Reshape for model input (model expects 2D array)
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def predict_single(self, ticket_text: str) -> dict:
        """
        Predict severity score for a single ticket.
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Input validation
            if not ticket_text or not ticket_text.strip():
                return {
                    'severity_score': 10.0,  # Minimum score for empty text
                    'severity_category': 'Low',
                    'confidence': 0.0,
                    'error': 'Empty or invalid input text'
                }
            
            # Preprocess text
            processed_text = self.preprocess_text(ticket_text)
            
            # Extract features
            features = self.extract_features(processed_text)
            
            # Make prediction
            raw_prediction = self.model.predict(features)[0]
            
            # Scale to severity score
            severity_score = self.scaler.transform(raw_prediction)
            
            # Ensure score is within valid range
            severity_score = self.scaler.clip_scores(severity_score)
            
            # Get severity category
            severity_category = self.scaler.get_severity_category(severity_score)
            
            # Calculate confidence (simplified approach using prediction variance)
            confidence = self._calculate_confidence(features)
            
            result = {
                'severity_score': float(severity_score),
                'severity_category': severity_category,
                'confidence': confidence,
                'processed_text': processed_text,
                'detected_language': self.language_detector.detect_language(ticket_text)
            }
            
            logger.info(f"Prediction completed: {severity_score:.2f} ({severity_category})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'severity_score': 50.0,  # Default middle score
                'severity_category': 'Medium',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, ticket_texts: List[str]) -> List[dict]:
        """
        Predict severity scores for multiple tickets.
        
        Args:
            ticket_texts: List of raw ticket texts
            
        Returns:
            List of prediction dictionaries
        """
        try:
            logger.info(f"Starting batch prediction for {len(ticket_texts)} tickets")
            
            results = []
            for i, text in enumerate(ticket_texts):
                result = self.predict_single(text)
                result['ticket_index'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(ticket_texts)} tickets")
            
            logger.info("Batch prediction completed")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """
        Calculate prediction confidence (simplified approach).
        
        Args:
            features: Feature vector
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate confidence from tree predictions
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [e.predict(features)[0] for e in self.model.estimators_]
                variance = float(np.var(tree_predictions))
                max_variance = 400.0
                confidence = max(0.1, min(0.99, 1.0 - (variance / max_variance)))
                return float(confidence)
            return 0.85
        except Exception:
            return 0.85
    
    def get_memory_usage(self) -> dict:
        """Get process memory usage for debugging."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return {
                'rss_mb': process.memory_info().rss / (1024 * 1024),
                'vms_mb': process.memory_info().vms / (1024 * 1024)
            }
        except Exception:
            return {'error': 'psutil not installed or inaccessible'}
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            info = {
                'model_type': type(self.model).__name__,
                'scaler_info': self.scaler.get_scaler_info(),
                'embedding_model': self.embeddings_model_name,
                'embedding_dim': self.embeddings.embedding_dim if self.embeddings else "not initialized",
                'model_dir': self.model_dir
            }
            
            # Add model-specific info
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info['max_depth'] = self.model.max_depth
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {}
    
    def validate_prediction(self, prediction: dict) -> bool:
        """
        Validate prediction results.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            True if prediction is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['severity_score', 'severity_category']
            for field in required_fields:
                if field not in prediction:
                    return False
            
            # Check score range
            score = prediction['severity_score']
            if not (10 <= score <= 100):
                return False
            
            # Check category
            valid_categories = ['High', 'Medium', 'Low']
            if prediction['severity_category'] not in valid_categories:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {str(e)}")
            return False

def main():
    """Main function for testing predictions."""
    try:
        # Initialize predictor
        predictor = SeverityPredictor()
        
        # Test predictions
        test_tickets = [
            "Server is completely down and no one can access email",
            "Printer not working in office",
            "सर्वर डाउन है और कोई भी काम नहीं कर सकता",
            "Need password reset for my account",
            "Critical database corruption detected"
        ]
        
        print("\n" + "="*60)
        print("TESTING SEVERITY PREDICTIONS")
        print("="*60)
        
        for i, ticket in enumerate(test_tickets, 1):
            result = predictor.predict_single(ticket)
            
            print(f"\nTicket {i}: {ticket}")
            print(f"Severity Score: {result['severity_score']:.2f}")
            print(f"Category: {result['severity_category']}")
            print(f"Language: {result.get('detected_language', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()