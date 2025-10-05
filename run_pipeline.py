"""
Simple script to run the entire pipeline on your CSV file.
"""
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import ingest, redact, embed, discover, rules, classify, export

def run_pipeline(csv_path: str, dataset_name: str = "my_data"):
    """Run the complete pipeline."""
    print("="*60)
    print("LINNAEUS - Ticket Taxonomy Tool")
    print("="*60)
    print()
    
    # Read and prepare CSV
    print("Step 0: Preparing data...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # Auto-prepare data
    prepared_df = df.copy()
    
    # Add case_id if missing
    if 'case_id' not in prepared_df.columns:
        prepared_df['case_id'] = prepared_df.index.map(lambda x: f"TICKET_{x:05d}")
        print("   [OK] Created case_id column")
    
    # Add created_at if missing
    if 'created_at' not in prepared_df.columns:
        prepared_df['created_at'] = pd.Timestamp.now().isoformat()
        print("   [OK] Created created_at column")
    
    # Combine text fields
    text_fields = []
    for col in ['Main Category', 'Issue', 'Detail', 'description', 'subject']:
        if col in prepared_df.columns:
            text_fields.append(col)
    
    if text_fields:
        print(f"   [OK] Combining fields: {', '.join(text_fields)}")
        prepared_df['description'] = prepared_df[text_fields].fillna('').agg(' | '.join, axis=1)
    
    # Add subject if missing
    if 'subject' not in prepared_df.columns:
        if 'Issue' in prepared_df.columns:
            prepared_df['subject'] = prepared_df['Issue']
        else:
            prepared_df['subject'] = prepared_df['description'].str[:50]
        print("   [OK] Created subject column")
    
    # Add defaults
    defaults = {
        'channel': 'unknown',
        'product_line': prepared_df.get('Main Category', pd.Series(['unknown'] * len(prepared_df))),
        'region': 'unknown',
        'language': 'en',
        'severity': 'medium',
        'status': 'open'
    }
    
    for col, value in defaults.items():
        if col not in prepared_df.columns:
            prepared_df[col] = value
    
    # Save prepared file
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    prepared_path = f"data/raw/{dataset_name}.csv"
    prepared_df.to_csv(prepared_path, index=False)
    print(f"   [OK] Saved to {prepared_path}")
    print()
    
    # Step 1: Ingest
    print("Step 1/7: Ingesting data...")
    warehouse_path = ingest.ingest_csv(prepared_path, dataset_name)
    print()
    
    # Step 2: Redact
    print("Step 2/7: Redacting PII...")
    redacted_path = redact.redact_parquet(warehouse_path, dataset_name)
    print()
    
    # Step 3: Embed
    print("Step 3/7: Generating embeddings...")
    embeddings_path, faiss_path = embed.process_embeddings(redacted_path, dataset_name)
    print()
    
    # Step 4: Discover
    print("Step 4/7: Discovering topics...")
    topics_report = discover.discover_topics(redacted_path, "artifacts/models/embeddings.npy", dataset_name)
    print()
    
    # Step 5: Rules
    print("Step 5/7: Applying rules...")
    with_rules_path = rules.apply_rules_to_data(redacted_path, dataset_name)
    print()
    
    # Step 6: Classify
    print("Step 6/7: Classifying tickets...")
    classified_path = classify.classify_tickets(with_rules_path, "artifacts/models/embeddings.npy", dataset_name)
    print()
    
    # Step 7: Export
    print("Step 7/7: Exporting results...")
    export_paths = export.export_with_evaluation(classified_path, dataset_name)
    print()
    
    print("="*60)
    print("[OK] PIPELINE COMPLETE!")
    print("="*60)
    print()
    print("Your results are ready:")
    for name, path in export_paths.items():
        if path:
            print(f"   - {name}: {path}")
    print()
    print("Open the HTML files in your browser to view results!")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <your_tickets.csv> [dataset_name]")
        print()
        print("Example: python run_pipeline.py my_tickets.csv")
        print("Example: python run_pipeline.py C:\\path\\to\\tickets.csv my_data")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "my_data"
    
    run_pipeline(csv_file, name)

