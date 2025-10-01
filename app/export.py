"""
Export classification results to CSV and Parquet formats.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def create_labels_export(df: pd.DataFrame) -> pd.DataFrame:
    """Create the final labels export DataFrame with data quality checks."""
    # Select and rename columns for the final export
    export_columns = {
        'case_id': 'case_id',
        'predicted_code': 'predicted_code',
        'predicted_name': 'predicted_name',
        'confidence': 'confidence',
        'model_strategy': 'model_strategy',
        'version': 'version',
        'decided_by': 'decided_by',
        'decided_at_utc': 'decided_at_utc'
    }
    
    # Create export DataFrame
    export_df = pd.DataFrame()
    for export_col, source_col in export_columns.items():
        if source_col in df.columns:
            export_df[export_col] = df[source_col]
        else:
            logger.warning(f"Column {source_col} not found in DataFrame")
            export_df[export_col] = None
    
    # Data quality checks and cleaning
    logger.info("Performing data quality checks...")
    
    # Ensure case_id is not null
    null_case_ids = export_df['case_id'].isnull().sum()
    if null_case_ids > 0:
        logger.warning(f"Found {null_case_ids} null case_ids - these will be excluded")
        export_df = export_df[export_df['case_id'].notnull()]
    
    # Ensure confidence is between 0 and 1
    invalid_confidence = ((export_df['confidence'] < 0) | (export_df['confidence'] > 1)).sum()
    if invalid_confidence > 0:
        logger.warning(f"Found {invalid_confidence} invalid confidence scores - clipping to [0,1]")
        export_df['confidence'] = export_df['confidence'].clip(0, 1)
    
    # Ensure predicted_code is not empty string
    empty_codes = (export_df['predicted_code'] == '').sum()
    if empty_codes > 0:
        logger.warning(f"Found {empty_codes} empty predicted_code values - setting to None")
        export_df.loc[export_df['predicted_code'] == '', 'predicted_code'] = None
    
    # Fill missing decided_by with 'system'
    missing_decided_by = export_df['decided_by'].isnull().sum()
    if missing_decided_by > 0:
        logger.info(f"Filling {missing_decided_by} missing decided_by values with 'system'")
        export_df['decided_by'] = export_df['decided_by'].fillna('system')
    
    # Fill missing version with '1.0'
    missing_version = export_df['version'].isnull().sum()
    if missing_version > 0:
        logger.info(f"Filling {missing_version} missing version values with '1.0'")
        export_df['version'] = export_df['version'].fillna('1.0')
    
    # Ensure decided_at_utc is properly formatted
    if export_df['decided_at_utc'].dtype == 'object':
        # Try to convert to datetime
        try:
            export_df['decided_at_utc'] = pd.to_datetime(export_df['decided_at_utc'])
            export_df['decided_at_utc'] = export_df['decided_at_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        except:
            logger.warning("Could not convert decided_at_utc to datetime - keeping as is")
    
    logger.info(f"Export DataFrame ready: {len(export_df)} rows, {len(export_df.columns)} columns")
    
    return export_df

def create_summary_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a summary report of classification results."""
    total_tickets = len(df)
    
    # Count by strategy
    strategy_counts = df['model_strategy'].value_counts().to_dict()
    
    # Count by predicted category
    category_counts = df['predicted_code'].value_counts().to_dict()
    
    # Confidence statistics
    confidence_stats = df['confidence'].describe().to_dict()
    
    # Coverage statistics
    classified_count = sum(1 for strategy in df['model_strategy'] if strategy != 'none')
    coverage_percentage = (classified_count / total_tickets * 100) if total_tickets > 0 else 0
    
    # High confidence predictions
    high_confidence_count = sum(1 for conf in df['confidence'] if conf >= 0.7)
    high_confidence_percentage = (high_confidence_count / total_tickets * 100) if total_tickets > 0 else 0
    
    summary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_tickets': total_tickets,
            'classified_tickets': classified_count,
            'coverage_percentage': round(coverage_percentage, 2),
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_percentage': round(high_confidence_percentage, 2)
        },
        'strategy_breakdown': strategy_counts,
        'category_breakdown': category_counts,
        'confidence_statistics': confidence_stats
    }
    
    return summary

def export_results(input_path: str, output_name: str = "labels") -> Dict[str, str]:
    """
    Export classification results to CSV and Parquet formats.
    
    Args:
        input_path: Path to input Parquet file (classified data)
        output_name: Name for output files (without extension)
    
    Returns:
        Dictionary with paths to created files
    """
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading classified data from {input_path}")
    
    # Load Parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}")
    
    logger.info(f"Loaded {len(df)} classified tickets")
    
    # Create labels export
    labels_df = create_labels_export(df)
    
    # Save CSV
    csv_path = reports_dir / f"{output_name}.csv"
    labels_df.to_csv(csv_path, index=False)
    logger.info(f"Saved labels CSV to {csv_path}")
    
    # Save Parquet
    parquet_path = reports_dir / f"{output_name}.parquet"
    table = pa.Table.from_pandas(labels_df)
    pq.write_table(table, parquet_path)
    logger.info(f"Saved labels Parquet to {parquet_path}")
    
    # Create and save summary report
    summary = create_summary_report(df)
    
    import json
    summary_path = reports_dir / f"{output_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary report to {summary_path}")
    
    # Create HTML summary report
    html_path = create_html_summary(summary, reports_dir / f"{output_name}_summary.html")
    logger.info(f"Saved HTML summary to {html_path}")
    
    return {
        'csv_path': str(csv_path),
        'parquet_path': str(parquet_path),
        'summary_json_path': str(summary_path),
        'summary_html_path': str(html_path)
    }

def create_html_summary(summary: Dict[str, Any], output_path: Path) -> str:
    """Create HTML summary report."""
    metadata = summary['metadata']
    strategy_breakdown = summary['strategy_breakdown']
    category_breakdown = summary['category_breakdown']
    confidence_stats = summary['confidence_statistics']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classification Results Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .section h3 {{ color: #333; margin-top: 0; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .stat-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
            .breakdown {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .breakdown-item {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            .breakdown-item h4 {{ margin-top: 0; color: #2c3e50; }}
            .breakdown-list {{ list-style: none; padding: 0; }}
            .breakdown-list li {{ padding: 5px 0; border-bottom: 1px solid #ecf0f1; }}
            .breakdown-list li:last-child {{ border-bottom: none; }}
            .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Classification Results Summary</h1>
            <div class="metadata">
                <p><strong>Generated:</strong> {metadata['generated_at']}</p>
                <p><strong>Total Tickets:</strong> {metadata['total_tickets']:,}</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Coverage Statistics</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{metadata['coverage_percentage']:.1f}%</div>
                    <div class="stat-label">Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metadata['classified_tickets']:,}</div>
                    <div class="stat-label">Classified Tickets</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metadata['high_confidence_percentage']:.1f}%</div>
                    <div class="stat-label">High Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metadata['high_confidence_predictions']:,}</div>
                    <div class="stat-label">High Confidence Predictions</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Classification Breakdown</h3>
            <div class="breakdown">
                <div class="breakdown-item">
                    <h4>By Strategy</h4>
                    <ul class="breakdown-list">
    """
    
    for strategy, count in strategy_breakdown.items():
        percentage = (count / metadata['total_tickets'] * 100) if metadata['total_tickets'] > 0 else 0
        html_content += f"<li><strong>{strategy}:</strong> {count:,} ({percentage:.1f}%)</li>"
    
    html_content += """
                    </ul>
                </div>
                <div class="breakdown-item">
                    <h4>By Category</h4>
                    <ul class="breakdown-list">
    """
    
    for category, count in list(category_breakdown.items())[:10]:  # Top 10 categories
        percentage = (count / metadata['total_tickets'] * 100) if metadata['total_tickets'] > 0 else 0
        html_content += f"<li><strong>{category}:</strong> {count:,} ({percentage:.1f}%)</li>"
    
    if len(category_breakdown) > 10:
        html_content += f"<li><em>... and {len(category_breakdown) - 10} more categories</em></li>"
    
    html_content += f"""
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Confidence Statistics</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{confidence_stats.get('mean', 0):.3f}</div>
                    <div class="stat-label">Mean Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{confidence_stats.get('std', 0):.3f}</div>
                    <div class="stat-label">Std Deviation</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{confidence_stats.get('min', 0):.3f}</div>
                    <div class="stat-label">Min Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{confidence_stats.get('max', 0):.3f}</div>
                    <div class="stat-label">Max Confidence</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path)

def export_with_evaluation(input_path: str, output_name: str = "labels") -> Dict[str, str]:
    """
    Export labels and run evaluation in one command.
    
    Args:
        input_path: Path to classified data
        output_name: Name for output files
    
    Returns:
        Dictionary with paths to all created files
    """
    logger.info("Starting comprehensive export with evaluation...")
    
    # First export the labels
    export_paths = export_results(input_path, output_name)
    
    # Then run evaluation
    try:
        from .evaluate import evaluate_classification
        eval_html_path = evaluate_classification(input_path, f"{output_name}_eval")
        export_paths['evaluation_html'] = eval_html_path
        logger.info(f"Evaluation completed: {eval_html_path}")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        export_paths['evaluation_html'] = None
    
    return export_paths

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.export <classified_parquet_file> [output_name] [--with-eval]")
        print("Example: python -m app.export data/redacted/dataset_classified.parquet labels")
        print("Example: python -m app.export data/redacted/dataset_classified.parquet labels --with-eval")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "labels"
    with_eval = "--with-eval" in sys.argv
    
    try:
        if with_eval:
            result_paths = export_with_evaluation(input_path, output_name)
            print("Successfully exported results with evaluation:")
        else:
            result_paths = export_results(input_path, output_name)
            print("Successfully exported results:")
        
        for file_type, path in result_paths.items():
            if path:  # Only show non-None paths
                print(f"  {file_type}: {path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
