"""
PII redaction using regex patterns and spaCy NER.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIRedactor:
    """PII redaction using regex patterns and spaCy NER."""
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None
        
        # Regex patterns for common PII
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'vin': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),  # VIN pattern
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        }
        
        # Initialize spaCy if requested
        if self.use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_spacy = False
                self.nlp = None
    
    def redact_regex(self, text: str) -> str:
        """Apply regex-based PII redaction."""
        if not isinstance(text, str):
            return text
            
        redacted = text
        
        # Email addresses
        redacted = self.patterns['email'].sub('[EMAIL]', redacted)
        
        # Phone numbers
        redacted = self.patterns['phone'].sub('[PHONE]', redacted)
        
        # VIN numbers
        redacted = self.patterns['vin'].sub('[VIN]', redacted)
        
        # SSN
        redacted = self.patterns['ssn'].sub('[SSN]', redacted)
        
        # Credit card numbers
        redacted = self.patterns['credit_card'].sub('[CREDIT_CARD]', redacted)
        
        return redacted
    
    def redact_spacy(self, text: str) -> str:
        """Apply spaCy NER-based PII redaction."""
        if not self.nlp or not isinstance(text, str):
            return text
            
        doc = self.nlp(text)
        redacted = text
        
        # Replace named entities that might be PII
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:  # Person, Organization, Geopolitical entity
                redacted = redacted.replace(ent.text, f'[{ent.label_}]')
        
        return redacted
    
    def redact_text(self, text: str) -> str:
        """Apply both regex and spaCy redaction."""
        if not isinstance(text, str):
            return text
            
        # Apply regex redaction first
        redacted = self.redact_regex(text)
        
        # Apply spaCy redaction if available
        if self.use_spacy:
            redacted = self.redact_spacy(redacted)
        
        return redacted

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def redact_dataframe(df: pd.DataFrame, redactor: PIIRedactor) -> pd.DataFrame:
    """Apply PII redaction to a DataFrame."""
    df_redacted = df.copy()
    
    # Redact text columns
    text_columns = ['subject', 'description']
    
    for col in text_columns:
        if col in df_redacted.columns:
            logger.info(f"Redacting PII in column: {col}")
            df_redacted[f"{col}_redacted"] = df_redacted[col].apply(redactor.redact_text)
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    
    return df_redacted

def redact_parquet(input_path: str, output_name: str = "dataset") -> str:
    """
    Load Parquet file, apply PII redaction, and save redacted version.
    
    Args:
        input_path: Path to input Parquet file
        output_name: Name for output parquet file (without extension)
    
    Returns:
        Path to created redacted parquet file
    """
    config = load_config()
    redacted_dir = Path(config["paths"]["redacted_dir"])
    redacted_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PII redaction is enabled
    if not config.get("privacy", {}).get("redact_pii", True):
        logger.info("PII redaction is disabled in config")
        return input_path
    
    logger.info(f"Loading data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} rows for redaction")
    
    # Initialize redactor
    redactor = PIIRedactor(use_spacy=True)
    
    # Apply redaction
    df_redacted = redact_dataframe(df, redactor)
    
    # Save redacted data
    output_path = redacted_dir / f"{output_name}_redacted.parquet"
    
    try:
        # Convert to PyArrow table for better type preservation
        table = pa.Table.from_pandas(df_redacted)
        pq.write_table(table, output_path)
        logger.info(f"Saved redacted data to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to save redacted Parquet: {e}")
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.redact <parquet_file> [output_name]")
        print("Example: python -m app.redact data/warehouse/dataset.parquet dataset")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    
    try:
        output_path = redact_parquet(input_path, output_name)
        print(f"Successfully redacted data to: {output_path}")
    except Exception as e:
        logger.error(f"Redaction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
