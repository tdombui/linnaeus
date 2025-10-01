"""
Topic discovery using BERTopic or UMAP+HDBSCAN clustering.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from bertopic import BERTopic
from keybert import KeyBERT
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicDiscoverer:
    """Discover topics using BERTopic or UMAP+HDBSCAN."""
    
    def __init__(self, use_bertopic: bool = True, min_cluster_size: int = 25):
        self.use_bertopic = use_bertopic
        self.min_cluster_size = min_cluster_size
        self.topic_model = None
        self.keybert_model = None
        
    def load_models(self):
        """Load BERTopic and KeyBERT models."""
        if self.use_bertopic:
            logger.info("Initializing BERTopic model...")
            self.topic_model = BERTopic(
                min_topic_size=self.min_cluster_size,
                calculate_probabilities=True,
                verbose=True
            )
        
        logger.info("Initializing KeyBERT model...")
        self.keybert_model = KeyBERT()
    
    def discover_topics_bertopic(self, texts: List[str], embeddings: np.ndarray) -> Tuple[List[int], Dict]:
        """Discover topics using BERTopic."""
        if self.topic_model is None:
            self.load_models()
        
        logger.info("Running BERTopic clustering...")
        
        # Fit BERTopic model
        topics, probs = self.topic_model.fit_transform(texts, embeddings)
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        logger.info(f"Discovered {len(topic_info)} topics")
        
        return topics, {
            'topic_info': topic_info,
            'probabilities': probs,
            'model': self.topic_model
        }
    
    def discover_topics_umap_hdbscan(self, embeddings: np.ndarray) -> Tuple[List[int], Dict]:
        """Discover topics using UMAP + HDBSCAN."""
        from umap import UMAP
        import hdbscan
        
        logger.info("Running UMAP dimensionality reduction...")
        
        # UMAP dimensionality reduction
        umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        logger.info("Running HDBSCAN clustering...")
        
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric='euclidean'
        )
        topics = clusterer.fit_predict(umap_embeddings)
        
        logger.info(f"Discovered {len(set(topics)) - (1 if -1 in topics else 0)} topics")
        
        return topics, {
            'umap_model': umap_model,
            'clusterer': clusterer,
            'umap_embeddings': umap_embeddings
        }
    
    def extract_keywords(self, texts: List[str], topics: List[int], 
                        topic_assignments: Dict = None) -> Dict[int, List[str]]:
        """Extract keywords for each topic using KeyBERT."""
        if self.keybert_model is None:
            self.load_models()
        
        logger.info("Extracting keywords for each topic...")
        
        topic_keywords = {}
        unique_topics = sorted(set(topics))
        
        for topic_id in unique_topics:
            if topic_id == -1:  # Skip noise cluster
                continue
                
            # Get texts for this topic
            topic_texts = [texts[i] for i, t in enumerate(topics) if t == topic_id]
            
            if len(topic_texts) < 3:  # Need at least 3 texts for keyword extraction
                continue
            
            # Combine texts for keyword extraction
            combined_text = " ".join(topic_texts[:50])  # Limit to first 50 texts
            
            # Extract keywords
            try:
                keywords = self.keybert_model.extract_keywords(
                    combined_text, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=20,
                    top_n=5
                )
                topic_keywords[topic_id] = [kw[0] for kw in keywords]
            except Exception as e:
                logger.warning(f"Failed to extract keywords for topic {topic_id}: {e}")
                topic_keywords[topic_id] = []
        
        return topic_keywords
    
    def get_topic_exemplars(self, df: pd.DataFrame, topics: List[int], 
                           num_exemplars: int = 3) -> Dict[int, List[str]]:
        """Get exemplar case IDs for each topic."""
        topic_exemplars = {}
        unique_topics = sorted(set(topics))
        
        for topic_id in unique_topics:
            if topic_id == -1:  # Skip noise cluster
                continue
            
            # Get case IDs for this topic
            topic_mask = np.array(topics) == topic_id
            topic_case_ids = df.loc[topic_mask, 'case_id'].tolist()
            
            # Take first few as exemplars
            topic_exemplars[topic_id] = topic_case_ids[:num_exemplars]
        
        return topic_exemplars

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def discover_topics(input_path: str, embeddings_path: str, output_name: str = "dataset") -> str:
    """
    Discover topics in ticket data.
    
    Args:
        input_path: Path to input Parquet file (redacted data)
        embeddings_path: Path to embeddings numpy file
        output_name: Name for output files (without extension)
    
    Returns:
        Path to created topics JSON file
    """
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Get clustering parameters from config
    use_bertopic = config.get("nlp", {}).get("use_bertopic", True)
    min_cluster_size = config.get("nlp", {}).get("min_cluster_size", 25)
    
    logger.info(f"Loading data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} rows for topic discovery")
    
    # Load embeddings
    try:
        embeddings = np.load(embeddings_path)
    except Exception as e:
        raise ValueError(f"Failed to load embeddings: {e}")
    
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Create composite text for each ticket
    def create_composite_text(row):
        subject = str(row.get("subject_redacted", row.get("subject", "")))
        description = str(row.get("description_redacted", row.get("description", "")))
        return f"{subject}\n{description}".strip()
    
    texts = df.apply(create_composite_text, axis=1).tolist()
    
    # Initialize topic discoverer
    discoverer = TopicDiscoverer(use_bertopic=use_bertopic, min_cluster_size=min_cluster_size)
    
    # Discover topics
    if use_bertopic:
        topics, topic_data = discoverer.discover_topics_bertopic(texts, embeddings)
    else:
        topics, topic_data = discoverer.discover_topics_umap_hdbscan(embeddings)
    
    # Extract keywords
    topic_keywords = discoverer.extract_keywords(texts, topics)
    
    # Get exemplars
    topic_exemplars = discoverer.get_topic_exemplars(df, topics)
    
    # Create topic report
    topic_report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_tickets': len(df),
            'num_topics': len(set(topics)) - (1 if -1 in topics else 0),
            'min_cluster_size': min_cluster_size,
            'method': 'bertopic' if use_bertopic else 'umap_hdbscan'
        },
        'topics': []
    }
    
    # Build topic information
    unique_topics = sorted(set(topics))
    for topic_id in unique_topics:
        if topic_id == -1:  # Skip noise cluster
            continue
        
        topic_size = sum(1 for t in topics if t == topic_id)
        keywords = topic_keywords.get(topic_id, [])
        exemplars = topic_exemplars.get(topic_id, [])
        
        topic_info = {
            'cluster_id': int(topic_id),
            'size': topic_size,
            'keywords': keywords,
            'exemplar_case_ids': exemplars
        }
        
        topic_report['topics'].append(topic_info)
    
    # Sort topics by size (largest first)
    topic_report['topics'].sort(key=lambda x: x['size'], reverse=True)
    
    # Save topics JSON
    topics_json_path = reports_dir / f"{output_name}_topics.json"
    with open(topics_json_path, 'w') as f:
        json.dump(topic_report, f, indent=2)
    
    logger.info(f"Saved topics report to {topics_json_path}")
    
    # Create HTML report
    html_path = create_html_report(topic_report, reports_dir / f"{output_name}_topics.html")
    logger.info(f"Saved HTML report to {html_path}")
    
    return str(topics_json_path)

def create_html_report(topic_report: Dict, output_path: Path) -> str:
    """Create HTML report for topic discovery results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Discovery Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .topic {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .topic h3 {{ color: #333; margin-top: 0; }}
            .keywords {{ color: #666; font-style: italic; }}
            .exemplars {{ color: #888; font-size: 0.9em; }}
            .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Topic Discovery Report</h1>
            <div class="metadata">
                <p><strong>Generated:</strong> {topic_report['metadata']['timestamp']}</p>
                <p><strong>Total Tickets:</strong> {topic_report['metadata']['total_tickets']}</p>
                <p><strong>Number of Topics:</strong> {topic_report['metadata']['num_topics']}</p>
                <p><strong>Method:</strong> {topic_report['metadata']['method']}</p>
                <p><strong>Min Cluster Size:</strong> {topic_report['metadata']['min_cluster_size']}</p>
            </div>
        </div>
        
        <h2>Discovered Topics</h2>
    """
    
    for topic in topic_report['topics']:
        keywords_str = ", ".join(topic['keywords']) if topic['keywords'] else "No keywords extracted"
        exemplars_str = ", ".join(topic['exemplar_case_ids']) if topic['exemplar_case_ids'] else "No exemplars"
        
        html_content += f"""
        <div class="topic">
            <h3>Topic {topic['cluster_id']} (Size: {topic['size']})</h3>
            <div class="keywords"><strong>Keywords:</strong> {keywords_str}</div>
            <div class="exemplars"><strong>Exemplar Case IDs:</strong> {exemplars_str}</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m app.discover <parquet_file> <embeddings_file> [output_name]")
        print("Example: python -m app.discover data/redacted/dataset_redacted.parquet artifacts/models/dataset_embeddings.npy dataset")
        sys.exit(1)
    
    input_path = sys.argv[1]
    embeddings_path = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else "dataset"
    
    try:
        topics_path = discover_topics(input_path, embeddings_path, output_name)
        print(f"Successfully discovered topics: {topics_path}")
    except Exception as e:
        logger.error(f"Topic discovery failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
