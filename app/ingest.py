"""
CSV to Parquet ingestion with schema validation and deduplication.
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketSchema(BaseModel):
    """Schema for ticket data validation."""
    case_id: str = Field(..., description="Unique case identifier")
    created_at: str = Field(..., description="Timestamp in UTC")
    subject: str = Field(..., description="Ticket subject")
    description: str = Field(..., description="Ticket description")
    current_category: Optional[str] = Field(None, description="Current category if available")
    resolution_code: Optional[str] = Field(None, description="Resolution code if available")
    channel: str = Field(..., description="Support channel")
    product_line: str = Field(..., description="Product line")
    region: str = Field(..., description="Geographic region")
    language: str = Field(..., description="Language")
    severity: str = Field(..., description="Severity level")
    status: str = Field(..., description="Ticket status")
    close_reason: Optional[str] = Field(None, description="Close reason if available")
    agent_team: Optional[str] = Field(None, description="Agent team if available")

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that required columns are present."""
    required_columns = [
        "case_id", "created_at", "subject", "description", 
        "channel", "product_line", "region", "language", 
        "severity", "status"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate types."""
    # Ensure case_id is string and unique
    df["case_id"] = df["case_id"].astype(str)
    
    # Convert created_at to datetime
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    
    # Ensure text fields are strings
    text_columns = ["subject", "description", "channel", "product_line", 
                   "region", "language", "severity", "status"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Optional fields
    optional_columns = ["current_category", "resolution_code", "close_reason", "agent_team"]
    for col in optional_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def create_content_hash(row: pd.Series) -> str:
    """Create hash for deduplication based on normalized content."""
    # Normalize text by lowercasing and removing extra whitespace
    subject = str(row.get("subject", "")).lower().strip()
    description = str(row.get("description", "")).lower().strip()
    
    # Combine and hash
    content = f"{subject}\n{description}"
    return hashlib.md5(content.encode()).hexdigest()

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate tickets based on content hash."""
    logger.info(f"Starting with {len(df)} tickets")
    
    # Create content hash for deduplication
    df["content_hash"] = df.apply(create_content_hash, axis=1)
    
    # Keep first occurrence of each unique content
    df_deduped = df.drop_duplicates(subset=["content_hash"], keep="first")
    
    # Remove the temporary hash column
    df_deduped = df_deduped.drop(columns=["content_hash"])
    
    duplicates_removed = len(df) - len(df_deduped)
    logger.info(f"Removed {duplicates_removed} duplicate tickets")
    logger.info(f"Final dataset: {len(df_deduped)} tickets")
    
    return df_deduped

def ingest_csv(csv_path: str, output_name: str = "dataset") -> str:
    """
    Ingest CSV file and convert to Parquet with validation.
    
    Args:
        csv_path: Path to input CSV file
        output_name: Name for output parquet file (without extension)
    
    Returns:
        Path to created parquet file
    """
    config = load_config()
    warehouse_dir = Path(config["paths"]["warehouse_dir"])
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading CSV from {csv_path}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
    
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Validate required columns
    validate_required_columns(df)
    logger.info("Required columns validation passed")
    
    # Cast types
    df = cast_types(df)
    logger.info("Type casting completed")
    
    # Deduplicate
    df = deduplicate(df)
    
    # Save to Parquet
    output_path = warehouse_dir / f"{output_name}.parquet"
    
    try:
        # Convert to PyArrow table for better type preservation
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        logger.info(f"Saved {len(df)} tickets to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to save Parquet: {e}")
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest <csv_file> [output_name]")
        print("Example: python -m app.ingest data/raw/tickets.csv tickets")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    
    try:
        output_path = ingest_csv(csv_path, output_name)
        print(f"Successfully ingested data to: {output_path}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
