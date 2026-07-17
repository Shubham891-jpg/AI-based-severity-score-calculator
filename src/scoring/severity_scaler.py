import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Union, List
import pickle
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SeverityScaler:
    """Scale model predictions to severity scores between 10-100."""
    
    def __init__(self, min_score: float = 10.0, max_score: float = 100.0):
        """
        Initialize severity scaler.
        
        Args:
            min_score: Minimum severity score
            max_score: Maximum severity score
        """
        self.min_score = min_score
        self.max_score = max_score
        self.scaler = MinMaxScaler(feature_range=(min_score, max_score))
        self.is_fitted = False
        
        # Store original prediction range for reference
        self.original_min = None
        self.original_max = None
        
    def fit(self, predictions: Union[List[float], np.ndarray]) -> 'SeverityScaler':
        """
        Fit the scaler on training predictions.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Self for method chaining
        """
        if len(predictions) == 0:
            raise ValueError("Cannot fit on empty predictions")
            
        predictions = np.array(predictions).reshape(-1, 1)
        
        try:
            logger.info(f"Fitting severity scaler on {len(predictions)} predictions")
            
            # Store original range
            self.original_min = float(np.min(predictions))
            self.original_max = float(np.max(predictions))
            
            # Fit the scaler
            self.scaler.fit(predictions)
            self.is_fitted = True
            
            logger.info(f"Scaler fitted. Original range: [{self.original_min:.4f}, {self.original_max:.4f}]")
            logger.info(f"Target range: [{self.min_score}, {self.max_score}]")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit severity scaler: {str(e)}")
            raise
    
    def transform(self, predictions: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Transform predictions to severity scores.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Scaled severity scores
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        # Handle single prediction
        if isinstance(predictions, (int, float)):
            predictions = np.array([predictions]).reshape(-1, 1)
            scaled = self.scaler.transform(predictions)
            base_score = float(scaled[0, 0])
            # Remap to new ranges: compress lower scores, expand upper scores
            remapped_score = self._remap_score(base_score)
            return remapped_score
        
        # Handle multiple predictions
        predictions = np.array(predictions).reshape(-1, 1)
        
        try:
            scaled_predictions = self.scaler.transform(predictions)
            base_scores = scaled_predictions.flatten()
            # Remap all scores
            remapped_scores = np.array([self._remap_score(score) for score in base_scores])
            
            logger.info(f"Transformed {len(predictions)} predictions to severity scores")
            return remapped_scores
            
        except Exception as e:
            logger.error(f"Failed to transform predictions: {str(e)}")
            raise
    
    def _remap_score(self, score: float) -> float:
        """
        Remap scores to new severity ranges.
        Original: 10-100 with natural clustering around 30-70
        Target: High (90-100), Medium (80-89), Low (10-79)
        
        Args:
            score: Original scaled score (10-100)
            
        Returns:
            Remapped score fitting new severity ranges
        """
        # Define remapping thresholds based on original score distribution
        # Original high severity: 60-100 -> New high: 90-100
        # Original medium: 40-59 -> New medium: 80-89
        # Original low: 10-39 -> New low: 10-79
        
        if score >= 60:
            # High severity: map 60-100 to 90-100
            return 90 + ((score - 60) / 40) * 10
        elif score >= 40:
            # Medium severity: map 40-59 to 80-89
            return 80 + ((score - 40) / 20) * 9
        else:
            # Low severity: map 10-39 to 10-79
            return 10 + ((score - 10) / 30) * 69
    
    
    def fit_transform(self, predictions: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Fit scaler and transform predictions in one step.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Scaled severity scores
        """
        return self.fit(predictions).transform(predictions)
    
    def inverse_transform(self, severity_scores: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Transform severity scores back to original prediction scale.
        
        Args:
            severity_scores: Scaled severity scores
            
        Returns:
            Original scale predictions
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
            
        # Handle single score
        if isinstance(severity_scores, (int, float)):
            scores = np.array([severity_scores]).reshape(-1, 1)
            original = self.scaler.inverse_transform(scores)
            return float(original[0, 0])
        
        # Handle multiple scores
        scores = np.array(severity_scores).reshape(-1, 1)
        
        try:
            original_predictions = self.scaler.inverse_transform(scores)
            return original_predictions.flatten()
            
        except Exception as e:
            logger.error(f"Failed to inverse transform scores: {str(e)}")
            raise
    
    def get_severity_category(self, score: float) -> str:
        """
        Get severity category based on score.
        
        Args:
            score: Severity score
            
        Returns:
            Severity category string
        """
        if score >= 90:
            return "High"
        elif score >= 80:
            return "Medium"
        elif score >= 10:
            return "Low"
        else:
            return "Low"  # Fallback for scores below 10
    
    def get_score_distribution(self, scores: List[float]) -> dict:
        """
        Get distribution of severity scores by category.
        
        Args:
            scores: List of severity scores
            
        Returns:
            Dictionary with category counts and percentages
        """
        if not scores:
            return {}
            
        categories = {
            "High": 0,
            "Medium": 0,
            "Low": 0
        }
        
        for score in scores:
            category = self.get_severity_category(score)
            categories[category] += 1
        
        total = len(scores)
        distribution = {}
        
        for category, count in categories.items():
            distribution[category] = {
                'count': count,
                'percentage': (count / total) * 100 if total > 0 else 0
            }
        
        return distribution
    
    def validate_score_range(self, scores: Union[float, List[float], np.ndarray]) -> bool:
        """
        Validate that scores are within expected range.
        
        Args:
            scores: Severity scores to validate
            
        Returns:
            True if all scores are within range, False otherwise
        """
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        scores = np.array(scores)
        
        return bool(np.all((scores >= self.min_score) & (scores <= self.max_score)))
    
    def clip_scores(self, scores: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Clip scores to valid range.
        
        Args:
            scores: Scores to clip
            
        Returns:
            Clipped scores within valid range
        """
        if isinstance(scores, (int, float)):
            return float(np.clip(scores, self.min_score, self.max_score))
        
        scores = np.array(scores)
        return np.clip(scores, self.min_score, self.max_score)
    
    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler to file.
        
        Args:
            filepath: Path to save the scaler
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'is_fitted': self.is_fitted,
                    'min_score': self.min_score,
                    'max_score': self.max_score,
                    'original_min': self.original_min,
                    'original_max': self.original_max
                }, f)
                
            logger.info(f"Severity scaler saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save scaler: {str(e)}")
            raise
    
    def load_scaler(self, filepath: str):
        """
        Load a fitted scaler from file.
        
        Args:
            filepath: Path to load the scaler from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']
            self.min_score = data['min_score']
            self.max_score = data['max_score']
            self.original_min = data.get('original_min')
            self.original_max = data.get('original_max')
            
            logger.info(f"Severity scaler loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load scaler: {str(e)}")
            raise
    
    def get_scaler_info(self) -> dict:
        """
        Get information about the scaler.
        
        Returns:
            Dictionary with scaler information
        """
        return {
            'is_fitted': self.is_fitted,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'original_min': self.original_min,
            'original_max': self.original_max,
            'score_range': self.max_score - self.min_score
        }