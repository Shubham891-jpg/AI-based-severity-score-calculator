import os
import nltk

# Force Hugging Face cache to be in the workspace directory so it persists in the Render run container
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(project_dir, ".cache", "huggingface")

from sentence_transformers import SentenceTransformer

def pre_download_assets():
    # Setup local NLTK data directory
    nltk_data_path = os.path.join(project_dir, ".cache", "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    
    print(f"⏳ Pre-downloading NLTK data to {nltk_data_path}...")
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    print("✅ NLTK data pre-downloaded successfully.")
    
    print(f"⏳ Pre-downloading SentenceTransformer model to {os.environ['HF_HOME']}...")
    # This downloads and caches the model in the workspace cache
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ SentenceTransformer model downloaded and cached successfully.")

if __name__ == "__main__":
    pre_download_assets()
