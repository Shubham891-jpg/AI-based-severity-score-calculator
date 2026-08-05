import os
# Force Hugging Face cache to be in the workspace directory so it persists in the Render run container
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HOME"] = os.path.join(project_root, ".cache", "huggingface")

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import pickle
try:
    import psutil
except ImportError:
    psutil = None
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MultilingualEmbeddings:
    """Generate multilingual sentence embeddings for IT tickets."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the multilingual embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            sample_embedding = self.model.encode(["test"])
            self.embedding_dim = sample_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_texts(self, texts: Union[str, List[str]], 
                    batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([])
            
        try:
            logger.info(f"Encoding {len(texts)} texts into embeddings")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Successfully encoded texts. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            1D numpy array representing the embedding
        """
        self._load_model()
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
            
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]  # Return 1D array
            
        except Exception as e:
            logger.error(f"Failed to encode single text: {str(e)}")
            return np.zeros(self.embedding_dim)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        self._load_model()
        try:
            embeddings = self.encode_texts([text1, text2], show_progress=False)
            
            if embeddings.shape[0] != 2:
                return 0.0
                
            # Compute cosine similarity
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_text: str, 
                         candidate_texts: List[str], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar texts to a query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top similar texts to return
            
        Returns:
            List of tuples (text, similarity_score) sorted by similarity
        """
        self._load_model()
        if not candidate_texts:
            return []
            
        try:
            # Encode query and candidates
            all_texts = [query_text] + candidate_texts
            embeddings = self.encode_texts(all_texts, show_progress=False)
            
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # Compute similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding, candidate_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                )
                similarities.append((candidate_texts[i], float(similarity)))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar texts: {str(e)}")
            return []
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage if DEBUG mode is enabled."""
        if os.getenv("DEBUG", "False").lower() == "true":
            if psutil is not None:
                process = psutil.Process(os.getpid())
                return {"memory_mb": process.memory_info().rss / (1024 * 1024)}
            return {"error": "psutil module not available"}
        return {"error": "DEBUG mode not enabled"}
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save the embeddings
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim
                }, f)
                
            logger.info(f"Embeddings saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
            
        Returns:
            Numpy array of embeddings
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            embeddings = data['embeddings']
            model_name = data.get('model_name', 'unknown')
            
            logger.info(f"Embeddings loaded from {filepath}. Model: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> dict:
        """
        Get statistics about embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        if embeddings.size == 0:
            return {}
            
        return {
            'shape': embeddings.shape,
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1)))
        }