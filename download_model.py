import os
import nltk
from sentence_transformers import SentenceTransformer

def pre_download_assets():
    print("⏳ Pre-downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK data pre-downloaded successfully.")
    
    print("⏳ Pre-downloading SentenceTransformer model (paraphrase-multilingual-MiniLM-L12-v2)...")
    # This downloads and caches the model in the build cache
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ SentenceTransformer model downloaded and cached successfully.")

if __name__ == "__main__":
    pre_download_assets()
