"""
Quick script to prepare your CSV with column mapping and run the pipeline.
"""
import pandas as pd
import sys
from pathlib import Path

def prepare_csv(input_csv: str, output_csv: str):
    """Prepare CSV with column mapping."""
    print(f"📥 Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"📊 Found {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Create a copy
    prepared_df = df.copy()
    
    # Add case_id if missing (use index)
    if 'case_id' not in prepared_df.columns:
        prepared_df['case_id'] = prepared_df.index.map(lambda x: f"TICKET_{x:05d}")
        print("✅ Created case_id column")
    
    # Add created_at if missing
    if 'created_at' not in prepared_df.columns:
        prepared_df['created_at'] = pd.Timestamp.now().isoformat()
        print("✅ Created created_at column")
    
    # Combine taxonomy fields into description
    text_fields = []
    for col in ['Main Category', 'Issue', 'Detail', 'description']:
        if col in prepared_df.columns:
            text_fields.append(col)
    
    if text_fields:
        print(f"✅ Combining text fields: {', '.join(text_fields)}")
        prepared_df['description'] = prepared_df[text_fields].fillna('').agg(' | '.join, axis=1)
    
    # Add subject if missing (use Issue or first 50 chars of description)
    if 'subject' not in prepared_df.columns:
        if 'Issue' in prepared_df.columns:
            prepared_df['subject'] = prepared_df['Issue']
        elif 'description' in prepared_df.columns:
            prepared_df['subject'] = prepared_df['description'].str[:50]
        print("✅ Created subject column")
    
    # Add default values for optional fields
    defaults = {
        'channel': 'unknown',
        'product_line': prepared_df.get('Main Category', 'unknown'),
        'region': 'unknown',
        'language': 'en',
        'severity': 'medium',
        'status': 'open'
    }
    
    for col, value in defaults.items():
        if col not in prepared_df.columns:
            prepared_df[col] = value
            print(f"✅ Added {col} column with default value")
    
    # Save prepared CSV
    prepared_df.to_csv(output_csv, index=False)
    print(f"✅ Saved prepared CSV to {output_csv}")
    print(f"📊 Final columns: {', '.join(prepared_df.columns)}")
    
    return output_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_and_run.py <your_tickets.csv>")
        print("Example: python prepare_and_run.py my_tickets.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "data/raw/prepared_tickets.csv"
    
    # Ensure directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    prepare_csv(input_file, output_file)
    
    print("\n" + "="*60)
    print("🚀 Now run the pipeline with these commands:")
    print("="*60)
    print()
    print(".venv\\Scripts\\activate")
    print("python -m app.ingest data/raw/prepared_tickets.csv my_tickets")
    print("python -m app.redact data/warehouse/my_tickets.parquet my_tickets")
    print("python -m app.embed data/redacted/my_tickets_redacted.parquet my_tickets")
    print("python -m app.discover data/redacted/my_tickets_redacted.parquet artifacts/models/embeddings.npy my_tickets")
    print("python -m app.rules data/redacted/my_tickets_redacted.parquet my_tickets")
    print("python -m app.classify data/redacted/my_tickets_with_rules.parquet artifacts/models/embeddings.npy my_tickets")
    print("python -m app.export data/redacted/my_tickets_classified.parquet my_tickets --with-eval")
    print()
    print("Or run them all at once (copy and paste):")
    print()


