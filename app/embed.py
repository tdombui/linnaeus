"""
Generate embeddings and build FAISS index for ticket text.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for ticket text using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.faiss_index = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name}: {e}")
    
    def create_composite_text(self, row: pd.Series) -> str:
        """Create composite text from subject and description."""
        subject = str(row.get("subject_redacted", row.get("subject", "")))
        description = str(row.get("description_redacted", row.get("description", "")))
        
        # Combine with newline separator
        composite = f"{subject}\n{description}".strip()
        return composite
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Generate embeddings for all tickets in the DataFrame."""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(df)} tickets")
        
        # Create composite text for each ticket
        texts = df.apply(self.create_composite_text, axis=1).tolist()
        
        # Generate embeddings
        logger.info("Computing embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        logger.info("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_embeddings(self, embeddings: np.ndarray, output_dir: Path):
        """Save embeddings to numpy file."""
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
    
    def save_faiss_index(self, index: faiss.Index, output_dir: Path):
        """Save FAISS index to file."""
        index_path = output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
    
    def load_embeddings(self, embeddings_path: Path) -> np.ndarray:
        """Load embeddings from numpy file."""
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def load_faiss_index(self, index_path: Path) -> faiss.Index:
        """Load FAISS index from file."""
        index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def process_embeddings(input_path: str, output_name: str = "dataset") -> Tuple[str, str]:
    """
    Process embeddings for ticket data.
    
    Args:
        input_path: Path to input Parquet file (redacted data)
        output_name: Name for output files (without extension)
    
    Returns:
        Tuple of (embeddings_path, faiss_index_path)
    """
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Get embedding model from config
    model_name = config.get("nlp", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    logger.info(f"Loading data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} rows for embedding generation")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name=model_name)
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(df)
    
    # Build FAISS index
    faiss_index = generator.build_faiss_index(embeddings)
    
    # Save embeddings and index
    embeddings_path = models_dir / f"{output_name}_embeddings.npy"
    index_path = models_dir / f"{output_name}_faiss_index.bin"
    
    generator.save_embeddings(embeddings, models_dir)
    generator.save_faiss_index(faiss_index, models_dir)
    
    # Rename files to include output_name
    final_embeddings_path = models_dir / f"{output_name}_embeddings.npy"
    final_index_path = models_dir / f"{output_name}_faiss_index.bin"
    
    if embeddings_path != final_embeddings_path:
        embeddings_path.rename(final_embeddings_path)
    if index_path != final_index_path:
        index_path.rename(final_index_path)
    
    logger.info(f"Embeddings saved to: {final_embeddings_path}")
    logger.info(f"FAISS index saved to: {final_index_path}")
    
    return str(final_embeddings_path), str(final_index_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.embed <parquet_file> [output_name]")
        print("Example: python -m app.embed data/redacted/dataset_redacted.parquet dataset")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    
    try:
        embeddings_path, index_path = process_embeddings(input_path, output_name)
        print(f"Successfully generated embeddings: {embeddings_path}")
        print(f"Successfully built FAISS index: {index_path}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
