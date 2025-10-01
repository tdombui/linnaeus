"""
Ensemble classification combining rules and classical ML models.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleClassifier:
    """Ensemble classifier combining rules and ML models."""
    
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.ml_model = None
        self.label_encoder = None
        self.is_trained = False
        
    def prepare_training_data(self, df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from DataFrame with existing categories."""
        # Filter rows with existing categories
        labeled_mask = df['current_category'].notna() & (df['current_category'] != '')
        labeled_df = df[labeled_mask]
        labeled_embeddings = embeddings[labeled_mask]
        
        if len(labeled_df) == 0:
            logger.warning("No labeled data found for training")
            return None, None
        
        logger.info(f"Found {len(labeled_df)} labeled examples for training")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labeled_df['current_category'])
        
        return labeled_embeddings, y_encoded
    
    def train_model(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train the ML model on embeddings and labels."""
        if len(embeddings) < 10:
            logger.warning("Insufficient training data for ML model")
            return
        
        logger.info(f"Training ML model on {len(embeddings)} examples")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Try multiple models and pick the best
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            logger.info(f"{name} validation accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.ml_model = best_model
        self.is_trained = True
        
        logger.info(f"Best model accuracy: {best_score:.3f}")
        
        # Print classification report
        y_pred = self.ml_model.predict(X_val)
        report = classification_report(y_val, y_pred, target_names=self.label_encoder.classes_)
        logger.info(f"Classification report:\n{report}")
    
    def predict_with_ml(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained ML model."""
        if not self.is_trained or self.ml_model is None:
            return None, None
        
        # Get predictions and probabilities
        predictions = self.ml_model.predict(embeddings)
        probabilities = self.ml_model.predict_proba(embeddings)
        
        # Get maximum probability for confidence
        max_probs = np.max(probabilities, axis=1)
        
        # Decode predictions
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        
        return decoded_predictions, max_probs
    
    def classify_tickets(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """Classify tickets using ensemble approach."""
        logger.info(f"Classifying {len(df)} tickets using ensemble approach")
        
        result_df = df.copy()
        
        # Initialize result columns
        result_df['predicted_code'] = None
        result_df['predicted_name'] = None
        result_df['confidence'] = 0.0
        result_df['model_strategy'] = 'none'
        
        # Strategy 1: Apply rules first
        rule_matches = 0
        for idx, row in result_df.iterrows():
            if pd.notna(row.get('rule_code')):
                result_df.at[idx, 'predicted_code'] = row['rule_code']
                result_df.at[idx, 'predicted_name'] = row['rule_name']
                result_df.at[idx, 'confidence'] = row['rule_confidence']
                result_df.at[idx, 'model_strategy'] = 'rules'
                rule_matches += 1
        
        logger.info(f"Rules classified {rule_matches} tickets")
        
        # Strategy 2: Use ML model for remaining tickets
        if self.is_trained:
            # Get tickets not classified by rules
            unclassified_mask = result_df['model_strategy'] == 'none'
            unclassified_indices = result_df[unclassified_mask].index
            unclassified_embeddings = embeddings[unclassified_mask]
            
            if len(unclassified_embeddings) > 0:
                logger.info(f"Using ML model for {len(unclassified_embeddings)} unclassified tickets")
                
                ml_predictions, ml_confidences = self.predict_with_ml(unclassified_embeddings)
                
                if ml_predictions is not None:
                    # Apply ML predictions with confidence threshold
                    for i, idx in enumerate(unclassified_indices):
                        if ml_confidences[i] >= self.confidence_threshold:
                            result_df.at[idx, 'predicted_code'] = ml_predictions[i]
                            result_df.at[idx, 'predicted_name'] = ml_predictions[i]  # Assuming code == name for now
                            result_df.at[idx, 'confidence'] = ml_confidences[i]
                            result_df.at[idx, 'model_strategy'] = 'ml_model'
        
        # Count final classifications
        total_classified = sum(1 for strategy in result_df['model_strategy'] if strategy != 'none')
        rules_classified = sum(1 for strategy in result_df['model_strategy'] if strategy == 'rules')
        ml_classified = sum(1 for strategy in result_df['model_strategy'] if strategy == 'ml_model')
        
        logger.info(f"Final classification results:")
        logger.info(f"  Total classified: {total_classified} ({total_classified/len(df)*100:.1f}%)")
        logger.info(f"  Rules: {rules_classified}")
        logger.info(f"  ML model: {ml_classified}")
        logger.info(f"  Unclassified: {len(df) - total_classified}")
        
        return result_df
    
    def save_model(self, output_dir: Path):
        """Save the trained model and label encoder."""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        model_path = output_dir / "ensemble_classifier.pkl"
        encoder_path = output_dir / "label_encoder.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ml_model, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved label encoder to {encoder_path}")
    
    def load_model(self, model_path: Path, encoder_path: Path):
        """Load a trained model and label encoder."""
        try:
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def classify_tickets(input_path: str, embeddings_path: str, output_name: str = "dataset") -> str:
    """
    Classify tickets using ensemble approach.
    
    Args:
        input_path: Path to input Parquet file (with rules applied)
        embeddings_path: Path to embeddings numpy file
        output_name: Name for output files (without extension)
    
    Returns:
        Path to created parquet file with classifications
    """
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    redacted_dir = Path(config["paths"]["redacted_dir"])
    
    models_dir.mkdir(parents=True, exist_ok=True)
    redacted_dir.mkdir(parents=True, exist_ok=True)
    
    # Get confidence threshold from config
    confidence_threshold = config.get("classifier", {}).get("confidence_threshold", 0.70)
    
    logger.info(f"Loading data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} rows for classification")
    
    # Load embeddings
    try:
        embeddings = np.load(embeddings_path)
    except Exception as e:
        raise ValueError(f"Failed to load embeddings: {e}")
    
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Initialize classifier
    classifier = EnsembleClassifier(confidence_threshold=confidence_threshold)
    
    # Prepare training data if current_category exists
    if 'current_category' in df.columns:
        X_train, y_train = classifier.prepare_training_data(df, embeddings)
        if X_train is not None:
            classifier.train_model(X_train, y_train)
            classifier.save_model(models_dir)
    
    # Classify all tickets
    classified_df = classifier.classify_tickets(df, embeddings)
    
    # Add metadata columns
    classified_df['version'] = '1.0'
    classified_df['decided_by'] = 'ensemble_classifier'
    classified_df['decided_at_utc'] = datetime.now().isoformat()
    
    # Save results
    output_path = redacted_dir / f"{output_name}_classified.parquet"
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Convert to PyArrow table for better type preservation
        table = pa.Table.from_pandas(classified_df)
        pq.write_table(table, output_path)
        logger.info(f"Saved classified data to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to save Parquet: {e}")
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m app.classify <parquet_file> <embeddings_file> [output_name]")
        print("Example: python -m app.classify data/redacted/dataset_with_rules.parquet artifacts/models/dataset_embeddings.npy dataset")
        sys.exit(1)
    
    input_path = sys.argv[1]
    embeddings_path = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else "dataset"
    
    try:
        output_path = classify_tickets(input_path, embeddings_path, output_name)
        print(f"Successfully classified tickets: {output_path}")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
