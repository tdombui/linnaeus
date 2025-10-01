"""
Rule-based classification using regex patterns from CSV rules.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Rule:
    """Represents a single classification rule."""
    
    def __init__(self, rule_id: str, enabled: bool, pattern: str, 
                 case_sensitive: bool, assign_code: str, assign_name: str, 
                 confidence: float, notes: str = ""):
        self.rule_id = rule_id
        self.enabled = enabled
        self.pattern = pattern
        self.case_sensitive = case_sensitive
        self.assign_code = assign_code
        self.assign_name = assign_name
        self.confidence = confidence
        self.notes = notes
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            self.compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            logger.warning(f"Invalid regex pattern in rule {rule_id}: {pattern}. Error: {e}")
            self.compiled_pattern = None
    
    def matches(self, text: str) -> bool:
        """Check if the rule matches the given text."""
        if not self.enabled or self.compiled_pattern is None:
            return False
        
        if not isinstance(text, str):
            return False
        
        return bool(self.compiled_pattern.search(text))

class RuleEngine:
    """Engine for applying classification rules."""
    
    def __init__(self, rules_file: str = "rules/keyword_rules.csv"):
        self.rules_file = rules_file
        self.rules: List[Rule] = []
        self.load_rules()
    
    def load_rules(self):
        """Load rules from CSV file."""
        rules_path = Path(self.rules_file)
        if not rules_path.exists():
            logger.warning(f"Rules file not found: {rules_path}")
            return
        
        try:
            df = pd.read_csv(rules_path)
            logger.info(f"Loading {len(df)} rules from {rules_path}")
            
            for _, row in df.iterrows():
                rule = Rule(
                    rule_id=str(row.get('rule_id', '')),
                    enabled=bool(row.get('enabled', True)),
                    pattern=str(row.get('pattern', '')),
                    case_sensitive=bool(row.get('case_sensitive', False)),
                    assign_code=str(row.get('assign_code', '')),
                    assign_name=str(row.get('assign_name', '')),
                    confidence=float(row.get('confidence', 0.0)),
                    notes=str(row.get('notes', ''))
                )
                self.rules.append(rule)
            
            enabled_count = sum(1 for rule in self.rules if rule.enabled)
            logger.info(f"Loaded {len(self.rules)} rules ({enabled_count} enabled)")
            
        except Exception as e:
            logger.error(f"Failed to load rules from {rules_path}: {e}")
    
    def apply_rules(self, text: str) -> Optional[Tuple[str, str, float, str]]:
        """
        Apply all rules to the given text.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (assign_code, assign_name, confidence, rule_id) if match found, None otherwise
        """
        if not isinstance(text, str):
            return None
        
        # Try rules in order (first match wins)
        for rule in self.rules:
            if rule.matches(text):
                logger.debug(f"Rule {rule.rule_id} matched for text: {text[:100]}...")
                return (rule.assign_code, rule.assign_name, rule.confidence, rule.rule_id)
        
        return None
    
    def apply_rules_to_dataframe(self, df: pd.DataFrame, 
                                text_columns: List[str] = None) -> pd.DataFrame:
        """
        Apply rules to all rows in a DataFrame.
        
        Args:
            df: DataFrame with ticket data
            text_columns: List of column names to search in (default: auto-detect)
            
        Returns:
            DataFrame with rule-based classification results
        """
        if text_columns is None:
            # Auto-detect text columns
            text_columns = []
            for col in ['subject_redacted', 'subject', 'description_redacted', 'description']:
                if col in df.columns:
                    text_columns.append(col)
        
        if not text_columns:
            logger.warning("No text columns found for rule application")
            return df
        
        logger.info(f"Applying rules to {len(df)} tickets using columns: {text_columns}")
        
        # Create result columns
        df_result = df.copy()
        df_result['rule_code'] = None
        df_result['rule_name'] = None
        df_result['rule_confidence'] = None
        df_result['rule_id'] = None
        
        matches_found = 0
        
        for idx, row in df_result.iterrows():
            # Combine text from all specified columns
            combined_text = " ".join([
                str(row.get(col, "")) for col in text_columns
            ])
            
            # Apply rules
            result = self.apply_rules(combined_text)
            if result:
                assign_code, assign_name, confidence, rule_id = result
                df_result.at[idx, 'rule_code'] = assign_code
                df_result.at[idx, 'rule_name'] = assign_name
                df_result.at[idx, 'rule_confidence'] = confidence
                df_result.at[idx, 'rule_id'] = rule_id
                matches_found += 1
        
        logger.info(f"Rules matched {matches_found} out of {len(df)} tickets ({matches_found/len(df)*100:.1f}%)")
        
        return df_result
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded rules."""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules if rule.enabled)
        valid_patterns = sum(1 for rule in self.rules if rule.compiled_pattern is not None)
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'valid_patterns': valid_patterns,
            'rules_file': self.rules_file
        }

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def apply_rules_to_data(input_path: str, output_name: str = "dataset") -> str:
    """
    Apply rules to ticket data and save results.
    
    Args:
        input_path: Path to input Parquet file (redacted data)
        output_name: Name for output parquet file (without extension)
    
    Returns:
        Path to created parquet file with rule results
    """
    config = load_config()
    redacted_dir = Path(config["paths"]["redacted_dir"])
    redacted_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} rows for rule application")
    
    # Initialize rule engine
    rule_engine = RuleEngine()
    
    # Print rule statistics
    stats = rule_engine.get_rule_stats()
    logger.info(f"Rule engine stats: {stats}")
    
    # Apply rules
    df_with_rules = rule_engine.apply_rules_to_dataframe(df)
    
    # Save results
    output_path = redacted_dir / f"{output_name}_with_rules.parquet"
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Convert to PyArrow table for better type preservation
        table = pa.Table.from_pandas(df_with_rules)
        pq.write_table(table, output_path)
        logger.info(f"Saved data with rule results to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to save Parquet: {e}")
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.rules <parquet_file> [output_name]")
        print("Example: python -m app.rules data/redacted/dataset_redacted.parquet dataset")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    
    try:
        output_path = apply_rules_to_data(input_path, output_name)
        print(f"Successfully applied rules to: {output_path}")
    except Exception as e:
        logger.error(f"Rule application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
