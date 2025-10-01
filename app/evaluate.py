"""
Evaluation metrics for classification results.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.metrics import ConfusionMatrixDisplay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationEvaluator:
    """Evaluator for classification results."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_coverage_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate coverage and basic metrics."""
        total_tickets = len(df)
        classified_tickets = df['predicted_code'].notna().sum()
        unclassified_tickets = total_tickets - classified_tickets
        
        coverage_percentage = (classified_tickets / total_tickets * 100) if total_tickets > 0 else 0
        
        # Confidence statistics
        confidence_stats = {k: float(v) for k, v in df['confidence'].describe().to_dict().items()}
        
        # High confidence predictions (>= 0.7)
        high_confidence = (df['confidence'] >= 0.7).sum()
        high_confidence_percentage = (high_confidence / total_tickets * 100) if total_tickets > 0 else 0
        
        # Strategy breakdown
        strategy_breakdown = {k: int(v) for k, v in df['model_strategy'].value_counts().to_dict().items()}
        
        # Category distribution
        category_breakdown = {k: int(v) for k, v in df['predicted_code'].value_counts().to_dict().items()}
        
        return {
            'total_tickets': int(total_tickets),
            'classified_tickets': int(classified_tickets),
            'unclassified_tickets': int(unclassified_tickets),
            'coverage_percentage': round(float(coverage_percentage), 2),
            'high_confidence_predictions': int(high_confidence),
            'high_confidence_percentage': round(float(high_confidence_percentage), 2),
            'confidence_statistics': confidence_stats,
            'strategy_breakdown': strategy_breakdown,
            'category_breakdown': category_breakdown
        }
    
    def calculate_confusion_metrics(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate confusion matrix if ground truth exists."""
        # Check if we have ground truth (current_category column)
        if 'current_category' not in df.columns:
            logger.info("No ground truth available (current_category column missing)")
            return None
        
        # Filter to rows with both predictions and ground truth
        comparison_df = df[
            df['predicted_code'].notna() & 
            df['current_category'].notna() & 
            (df['current_category'] != '')
        ].copy()
        
        if len(comparison_df) == 0:
            logger.info("No rows with both predictions and ground truth")
            return None
        
        logger.info(f"Evaluating {len(comparison_df)} tickets with ground truth")
        
        y_true = comparison_df['current_category']
        y_pred = comparison_df['predicted_code']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(set(y_true.tolist() + y_pred.tolist()))
        
        return {
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'classification_report': class_report,
            'support': len(comparison_df)
        }
    
    def create_coverage_visualization(self, metrics: Dict[str, Any], output_path: Path):
        """Create coverage visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Coverage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Coverage pie chart
        coverage_data = [
            metrics['classified_tickets'], 
            metrics['unclassified_tickets']
        ]
        labels = ['Classified', 'Unclassified']
        colors = ['#2ecc71', '#e74c3c']
        
        ax1.pie(coverage_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Coverage: {metrics["coverage_percentage"]}%')
        
        # 2. Strategy breakdown
        strategy_data = list(metrics['strategy_breakdown'].values())
        strategy_labels = list(metrics['strategy_breakdown'].keys())
        
        ax2.bar(strategy_labels, strategy_data, color=['#3498db', '#f39c12', '#9b59b6'])
        ax2.set_title('Classification Strategy Breakdown')
        ax2.set_ylabel('Number of Tickets')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Category distribution
        category_data = list(metrics['category_breakdown'].values())[:10]  # Top 10
        category_labels = list(metrics['category_breakdown'].keys())[:10]
        
        ax3.barh(category_labels, category_data, color='#1abc9c')
        ax3.set_title('Top Categories (Predicted)')
        ax3.set_xlabel('Number of Tickets')
        
        # 4. Confidence distribution
        confidence_stats = metrics['confidence_statistics']
        ax4.hist([confidence_stats['mean']], bins=10, color='#e67e22', alpha=0.7)
        ax4.axvline(confidence_stats['mean'], color='red', linestyle='--', 
                   label=f'Mean: {confidence_stats["mean"]:.3f}')
        ax4.set_title('Confidence Distribution')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Coverage visualization saved to {output_path}")
    
    def create_confusion_matrix_plot(self, confusion_metrics: Dict[str, Any], output_path: Path):
        """Create confusion matrix visualization."""
        cm = np.array(confusion_metrics['confusion_matrix'])
        labels = confusion_metrics['labels']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix\nAccuracy: {confusion_metrics["accuracy"]:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {output_path}")
    
    def generate_evaluation_report(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        # Calculate coverage metrics
        coverage_metrics = self.calculate_coverage_metrics(df)
        
        # Calculate confusion metrics if ground truth exists
        confusion_metrics = self.calculate_confusion_metrics(df)
        
        # Create visualizations
        coverage_viz_path = output_dir / "coverage_analysis.png"
        self.create_coverage_visualization(coverage_metrics, coverage_viz_path)
        
        if confusion_metrics:
            confusion_viz_path = output_dir / "confusion_matrix.png"
            self.create_confusion_matrix_plot(confusion_metrics, confusion_viz_path)
        
        # Compile results
        evaluation_results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_tickets': int(coverage_metrics['total_tickets']),
                'evaluation_type': 'confusion_matrix' if confusion_metrics else 'coverage_only'
            },
            'coverage_metrics': coverage_metrics,
            'confusion_metrics': confusion_metrics,
            'visualizations': {
                'coverage_analysis': str(coverage_viz_path),
                'confusion_matrix': str(confusion_viz_path) if confusion_metrics else None
            }
        }
        
        # Save results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return evaluation_results

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    import toml
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return toml.load(config_path)

def evaluate_classification(input_path: str, output_name: str = "evaluation") -> str:
    """
    Evaluate classification results.
    
    Args:
        input_path: Path to classified data
        output_name: Name for output files
    
    Returns:
        Path to evaluation results JSON
    """
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading classified data from {input_path}")
    
    # Load classified data
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load classified data: {e}")
    
    logger.info(f"Loaded {len(df)} tickets for evaluation")
    
    # Initialize evaluator
    evaluator = ClassificationEvaluator()
    
    # Generate evaluation report
    results = evaluator.generate_evaluation_report(df, reports_dir)
    
    # Create HTML report
    html_path = create_html_evaluation_report(results, reports_dir / f"{output_name}_report.html")
    
    logger.info(f"Evaluation complete. Results saved to {reports_dir}")
    
    return str(html_path)

def create_html_evaluation_report(results: Dict[str, Any], output_path: Path) -> str:
    """Create HTML evaluation report."""
    metadata = results['metadata']
    coverage = results['coverage_metrics']
    confusion = results['confusion_metrics']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classification Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .section h3 {{ color: #333; margin-top: 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
            .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Classification Evaluation Report</h1>
            <div class="metadata">
                <p><strong>Generated:</strong> {metadata['generated_at']}</p>
                <p><strong>Total Tickets:</strong> {metadata['total_tickets']:,}</p>
                <p><strong>Evaluation Type:</strong> {metadata['evaluation_type'].replace('_', ' ').title()}</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Coverage Metrics</h3>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{coverage['coverage_percentage']:.1f}%</div>
                    <div class="metric-label">Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['classified_tickets']:,}</div>
                    <div class="metric-label">Classified Tickets</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['high_confidence_percentage']:.1f}%</div>
                    <div class="metric-label">High Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['high_confidence_predictions']:,}</div>
                    <div class="metric-label">High Confidence Predictions</div>
                </div>
            </div>
        </div>
    """
    
    if confusion:
        html_content += f"""
        <div class="section">
            <h3>Confusion Matrix Metrics</h3>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{confusion['accuracy']:.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{confusion['precision']:.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{confusion['recall']:.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{confusion['f1_score']:.3f}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>
        """
    
    html_content += """
        <div class="section">
            <h3>Visualizations</h3>
            <div class="visualization">
                <h4>Coverage Analysis</h4>
                <img src="coverage_analysis.png" alt="Coverage Analysis">
            </div>
    """
    
    if confusion:
        html_content += """
            <div class="visualization">
                <h4>Confusion Matrix</h4>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path)

def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.evaluate <classified_parquet_file> [output_name]")
        print("Example: python -m app.evaluate data/redacted/dataset_classified.parquet evaluation")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "evaluation"
    
    try:
        results_path = evaluate_classification(input_path, output_name)
        print(f"Successfully generated evaluation report: {results_path}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
