import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union
import pickle
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TFIDFFeatureExtractor:
    """Extract TF-IDF features from text data."""
    
    def __init__(self, max_features: int = 5000, 
                 min_df: int = 2, 
                 max_df: float = 0.95,
                 ngram_range: tuple = (1, 2)):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=None,  # We handle stopwords in preprocessing
            lowercase=False,  # We handle lowercasing in preprocessing
            token_pattern=r'\b\w+\b'
        )
        
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, texts: List[str]) -> 'TFIDFFeatureExtractor':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            Self for method chaining
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
            
        try:
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                raise ValueError("No valid texts found for fitting")
                
            self.vectorizer.fit(valid_texts)
            self.is_fitted = True
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            logger.info(f"TF-IDF vectorizer fitted. Features: {len(self.feature_names)}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit TF-IDF vectorizer: {str(e)}")
            raise
    
    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: Single text or list of texts to transform
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
            
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([]).reshape(0, len(self.feature_names))
            
        try:
            # Handle empty texts
            processed_texts = []
            for text in texts:
                if text and text.strip():
                    processed_texts.append(text)
                else:
                    processed_texts.append(" ")  # Use single space for empty texts
                    
            tfidf_matrix = self.vectorizer.transform(processed_texts)
            
            logger.info(f"Transformed {len(texts)} texts to TF-IDF features. Shape: {tfidf_matrix.shape}")
            return tfidf_matrix.toarray()
            
        except Exception as e:
            logger.error(f"Failed to transform texts: {str(e)}")
            raise
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts: List of texts to fit and transform
            
        Returns:
            TF-IDF feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary).
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        return list(self.feature_names)
    
    def get_top_features(self, text: str, top_k: int = 10) -> List[tuple]:
        """
        Get top TF-IDF features for a single text.
        
        Args:
            text: Input text
            top_k: Number of top features to return
            
        Returns:
            List of tuples (feature_name, tfidf_score)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        try:
            tfidf_vector = self.transform([text])[0]
            feature_names = self.get_feature_names()
            
            # Get indices of top features
            top_indices = np.argsort(tfidf_vector)[-top_k:][::-1]
            
            top_features = [
                (feature_names[idx], tfidf_vector[idx])
                for idx in top_indices
                if tfidf_vector[idx] > 0
            ]
            
            return top_features
            
        except Exception as e:
            logger.error(f"Failed to get top features: {str(e)}")
            return []
    
    def get_vocabulary_stats(self) -> dict:
        """
        Get statistics about the vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self.is_fitted:
            return {}
            
        vocabulary = self.vectorizer.vocabulary_
        
        return {
            'vocabulary_size': len(vocabulary),
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'actual_features': len(self.feature_names)
        }
    
    def save_vectorizer(self, filepath: str):
        """
        Save the fitted vectorizer to file.
        
        Args:
            filepath: Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'is_fitted': self.is_fitted,
                    'feature_names': self.feature_names,
                    'max_features': self.max_features,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'ngram_range': self.ngram_range
                }, f)
                
            logger.info(f"TF-IDF vectorizer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save vectorizer: {str(e)}")
            raise
    
    def load_vectorizer(self, filepath: str):
        """
        Load a fitted vectorizer from file.
        
        Args:
            filepath: Path to load the vectorizer from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.vectorizer = data['vectorizer']
            self.is_fitted = data['is_fitted']
            self.feature_names = data['feature_names']
            self.max_features = data['max_features']
            self.min_df = data['min_df']
            self.max_df = data['max_df']
            self.ngram_range = data['ngram_range']
            
            logger.info(f"TF-IDF vectorizer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {str(e)}")
            raise
    
    def analyze_feature_importance(self, texts: List[str], 
                                 labels: List[float]) -> pd.DataFrame:
        """
        Analyze feature importance based on correlation with labels.
        
        Args:
            texts: List of texts
            labels: List of corresponding labels (severity scores)
            
        Returns:
            DataFrame with feature importance analysis
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        try:
            tfidf_matrix = self.transform(texts)
            feature_names = self.get_feature_names()
            
            # Calculate correlation between each feature and labels
            correlations = []
            for i, feature_name in enumerate(feature_names):
                feature_values = tfidf_matrix[:, i]
                correlation = np.corrcoef(feature_values, labels)[0, 1]
                correlations.append({
                    'feature': feature_name,
                    'correlation': correlation if not np.isnan(correlation) else 0,
                    'abs_correlation': abs(correlation) if not np.isnan(correlation) else 0,
                    'mean_tfidf': np.mean(feature_values),
                    'std_tfidf': np.std(feature_values)
                })
            
            df = pd.DataFrame(correlations)
            df = df.sort_values('abs_correlation', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {str(e)}")
            return pd.DataFrame()