import pytest
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.predict import SeverityPredictor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.language_detector import LanguageDetector
from src.scoring.severity_scaler import SeverityScaler

class TestSeverityPredictor:
    """Test cases for severity prediction functionality."""
    
    @pytest.fixture
    def sample_tickets(self):
        """Sample tickets for testing."""
        return [
            "Server is completely down and no one can access email",
            "Printer not working in office",
            "सर्वर डाउन है और कोई भी काम नहीं कर सकता",
            "Need password reset for my account",
            "Critical database corruption detected",
            "",  # Empty text
            "   ",  # Whitespace only
            "a",  # Very short text
        ]
    
    @pytest.fixture
    def expected_score_ranges(self):
        """Expected score ranges for sample tickets."""
        return [
            (80, 100),  # Critical server down
            (20, 50),   # Printer issue
            (80, 100),  # Hindi critical issue
            (10, 30),   # Password reset
            (90, 100),  # Database corruption
            (10, 10),   # Empty text
            (10, 10),   # Whitespace only
            (10, 30),   # Very short text
        ]
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        # This test will only pass if model is trained
        try:
            predictor = SeverityPredictor()
            assert predictor.model is not None
            assert predictor.scaler is not None
            assert predictor.embeddings is not None
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_single_prediction_structure(self, sample_tickets):
        """Test structure of single prediction results."""
        try:
            predictor = SeverityPredictor()
            
            for ticket in sample_tickets[:3]:  # Test first 3 tickets
                result = predictor.predict_single(ticket)
                
                # Check required fields
                assert 'severity_score' in result
                assert 'severity_category' in result
                assert 'confidence' in result
                
                # Check data types
                assert isinstance(result['severity_score'], float)
                assert isinstance(result['severity_category'], str)
                assert isinstance(result['confidence'], float)
                
                # Check value ranges
                assert 10 <= result['severity_score'] <= 100
                assert 0 <= result['confidence'] <= 1
                assert result['severity_category'] in ['High', 'Medium', 'Low']
                
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_prediction_score_ranges(self, sample_tickets, expected_score_ranges):
        """Test that predictions fall within expected ranges."""
        try:
            predictor = SeverityPredictor()
            
            for ticket, (min_score, max_score) in zip(sample_tickets, expected_score_ranges):
                result = predictor.predict_single(ticket)
                score = result['severity_score']
                
                # Allow some tolerance for model predictions
                tolerance = 20
                assert (min_score - tolerance) <= score <= (max_score + tolerance), \
                    f"Score {score} not in expected range [{min_score}, {max_score}] for ticket: {ticket[:50]}"
                
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_batch_prediction(self, sample_tickets):
        """Test batch prediction functionality."""
        try:
            predictor = SeverityPredictor()
            
            results = predictor.predict_batch(sample_tickets[:5])
            
            assert len(results) == 5
            
            for result in results:
                assert 'severity_score' in result
                assert 'severity_category' in result
                assert 'ticket_index' in result
                
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs."""
        try:
            predictor = SeverityPredictor()
            
            # Test empty string
            result = predictor.predict_single("")
            assert result['severity_score'] == 10.0  # Minimum score
            
            # Test whitespace only
            result = predictor.predict_single("   ")
            assert result['severity_score'] == 10.0
            
            # Test None (should be handled gracefully)
            result = predictor.predict_single(None)
            assert 'error' in result or result['severity_score'] == 10.0
            
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_language_detection(self):
        """Test language detection in predictions."""
        try:
            predictor = SeverityPredictor()
            
            # English text
            result = predictor.predict_single("Server is down")
            assert result.get('detected_language') == 'en'
            
            # Hindi text
            result = predictor.predict_single("सर्वर डाउन है")
            assert result.get('detected_language') == 'hi'
            
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")
    
    def test_prediction_consistency(self):
        """Test that same input produces consistent results."""
        try:
            predictor = SeverityPredictor()
            
            ticket = "Database server experiencing performance issues"
            
            # Make multiple predictions
            results = [predictor.predict_single(ticket) for _ in range(3)]
            
            # Check consistency (should be identical for deterministic model)
            scores = [r['severity_score'] for r in results]
            categories = [r['severity_category'] for r in results]
            
            assert len(set(scores)) == 1, "Predictions should be consistent"
            assert len(set(categories)) == 1, "Categories should be consistent"
            
        except FileNotFoundError:
            pytest.skip("Model files not found - train model first")

class TestTextCleaner:
    """Test cases for text cleaning functionality."""
    
    @pytest.fixture
    def text_cleaner(self):
        """Text cleaner instance."""
        return TextCleaner()
    
    def test_basic_cleaning(self, text_cleaner):
        """Test basic text cleaning operations."""
        # Test HTML removal
        text = "<p>Server is down</p>"
        cleaned = text_cleaner.clean_text(text)
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        
        # Test URL removal
        text = "Check http://example.com for details"
        cleaned = text_cleaner.clean_text(text)
        assert "http://example.com" not in cleaned
        
        # Test email removal
        text = "Contact admin@company.com for help"
        cleaned = text_cleaner.clean_text(text)
        assert "admin@company.com" not in cleaned
    
    def test_whitespace_normalization(self, text_cleaner):
        """Test whitespace normalization."""
        text = "Server    is     down"
        cleaned = text_cleaner.clean_text(text)
        assert "    " not in cleaned
        assert cleaned == "server down"  # Assuming lowercase=True
    
    def test_empty_text_handling(self, text_cleaner):
        """Test handling of empty or None text."""
        assert text_cleaner.clean_text("") == ""
        assert text_cleaner.clean_text(None) == ""
        assert text_cleaner.clean_text("   ") == ""

class TestLanguageDetector:
    """Test cases for language detection."""
    
    @pytest.fixture
    def language_detector(self):
        """Language detector instance."""
        return LanguageDetector()
    
    def test_english_detection(self, language_detector):
        """Test English language detection."""
        text = "The server is down and users cannot access email"
        lang = language_detector.detect_language(text)
        assert lang == 'en'
    
    def test_hindi_detection(self, language_detector):
        """Test Hindi language detection."""
        text = "सर्वर डाउन है और कोई भी ईमेल एक्सेस नहीं कर सकता"
        lang = language_detector.detect_language(text)
        assert lang == 'hi'
    
    def test_short_text_handling(self, language_detector):
        """Test handling of very short text."""
        # Should default to English for very short text
        lang = language_detector.detect_language("hi")
        assert lang == 'en'  # Default fallback
    
    def test_empty_text_handling(self, language_detector):
        """Test handling of empty text."""
        lang = language_detector.detect_language("")
        assert lang == 'en'  # Default fallback

class TestSeverityScaler:
    """Test cases for severity scaling."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample model predictions for testing."""
        return np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5])
    
    def test_scaler_fitting(self, sample_predictions):
        """Test scaler fitting process."""
        scaler = SeverityScaler()
        scaler.fit(sample_predictions)
        
        assert scaler.is_fitted
        assert scaler.original_min is not None
        assert scaler.original_max is not None
    
    def test_score_transformation(self, sample_predictions):
        """Test score transformation to 10-100 range."""
        scaler = SeverityScaler()
        scaler.fit(sample_predictions)
        
        scaled_scores = scaler.transform(sample_predictions)
        
        # Check range
        assert np.all(scaled_scores >= 10)
        assert np.all(scaled_scores <= 100)
        
        # Check that min/max predictions map to min/max scores
        assert scaled_scores[0] == 10.0  # Minimum
        assert scaled_scores[-1] == 100.0  # Maximum
    
    def test_single_score_transformation(self, sample_predictions):
        """Test transformation of single scores."""
        scaler = SeverityScaler()
        scaler.fit(sample_predictions)
        
        # Test single value transformation
        single_score = scaler.transform(0.5)
        assert isinstance(single_score, float)
        assert 10 <= single_score <= 100
    
    def test_severity_categories(self):
        """Test severity category assignment."""
        scaler = SeverityScaler()
        
        assert scaler.get_severity_category(95) == "High"
        assert scaler.get_severity_category(85) == "Medium"
        assert scaler.get_severity_category(50) == "Low"
        assert scaler.get_severity_category(15) == "Low"
    
    def test_score_clipping(self):
        """Test score clipping to valid range."""
        scaler = SeverityScaler()
        
        # Test clipping values outside range
        assert scaler.clip_scores(5) == 10.0
        assert scaler.clip_scores(105) == 100.0
        assert scaler.clip_scores(50) == 50.0
        
        # Test clipping arrays
        scores = np.array([5, 50, 105])
        clipped = scaler.clip_scores(scores)
        expected = np.array([10, 50, 100])
        np.testing.assert_array_equal(clipped, expected)

def test_model_integration():
    """Integration test for the complete prediction pipeline."""
    try:
        predictor = SeverityPredictor()
        
        # Test various ticket types
        test_cases = [
            ("Critical system failure", 70, 100),
            ("Minor printer issue", 10, 40),
            ("सिस्टम फेल हो गया है", 70, 100),
            ("पासवर्ड रीसेट चाहिए", 10, 40)
        ]
        
        for ticket, min_expected, max_expected in test_cases:
            result = predictor.predict_single(ticket)
            score = result['severity_score']
            
            # Allow some tolerance
            tolerance = 30
            assert (min_expected - tolerance) <= score <= (max_expected + tolerance), \
                f"Score {score} not in expected range for: {ticket}"
            
            # Ensure all required fields are present
            assert all(key in result for key in [
                'severity_score', 'severity_category', 'confidence'
            ])
            
    except FileNotFoundError:
        pytest.skip("Model files not found - train model first")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])