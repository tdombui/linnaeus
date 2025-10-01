"""
Streamlit UI for reviewing clusters and auto-labels.
"""
import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import toml

# Configure page
st.set_page_config(
    page_title="Linnaeus",
    page_icon="✨",
    layout="wide"
)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    config_path = Path("configs/config.toml")
    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        return {}
    return toml.load(config_path)

def load_taxonomy() -> Dict[str, Any]:
    """Load taxonomy from YAML file."""
    import yaml
    taxonomy_path = Path("configs/taxonomy.yaml")
    if not taxonomy_path.exists():
        return {}
    
    try:
        with open(taxonomy_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load taxonomy: {e}")
        return {}

def load_topics_report(reports_dir: Path) -> Optional[Dict[str, Any]]:
    """Load topics report from JSON file."""
    reports_dir = Path(reports_dir)
    topics_files = list(reports_dir.glob("*_topics.json"))
    if not topics_files:
        return None
    
    # Use the most recent file
    latest_file = max(topics_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load topics report: {e}")
        return None

def load_classified_data(redacted_dir: Path) -> Optional[pd.DataFrame]:
    """Load classified data from Parquet file."""
    redacted_dir = Path(redacted_dir)
    classified_files = list(redacted_dir.glob("*_classified.parquet"))
    if not classified_files:
        return None
    
    # Use the most recent file
    latest_file = max(classified_files, key=lambda x: x.stat().st_mtime)
    
    try:
        return pd.read_parquet(latest_file)
    except Exception as e:
        st.error(f"Failed to load classified data: {e}")
        return None

def save_training_labels(df: pd.DataFrame, reports_dir: Path):
    """Save training labels from UI corrections."""
    training_path = reports_dir / "training_labels.parquet"
    
    # Select only the columns we need for training
    training_columns = ['case_id', 'predicted_code', 'predicted_name', 'confidence', 'model_strategy']
    training_df = df[training_columns].copy()
    
    # Add metadata
    training_df['version'] = '1.0'
    training_df['decided_by'] = 'ui_review'
    training_df['decided_at_utc'] = pd.Timestamp.now().isoformat()
    
    try:
        training_df.to_parquet(training_path, index=False)
        st.success(f"Saved training labels to {training_path}")
    except Exception as e:
        st.error(f"Failed to save training labels: {e}")

def clusters_explorer_page(topics_report: Dict[str, Any], classified_df: pd.DataFrame):
    """Page for exploring discovered clusters."""
    st.header("🎯 Clusters Explorer")
    
    if not topics_report:
        st.warning("No topics report found. Run topic discovery first.")
        return
    
    metadata = topics_report.get('metadata', {})
    topics = topics_report.get('topics', [])
    
    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickets", f"{metadata.get('total_tickets', 0):,}")
    with col2:
        st.metric("Number of Topics", metadata.get('num_topics', 0))
    with col3:
        st.metric("Method", metadata.get('method', 'unknown'))
    with col4:
        st.metric("Min Cluster Size", metadata.get('min_cluster_size', 0))
    
    st.divider()
    
    # Display topics
    for i, topic in enumerate(topics):
        with st.expander(f"Topic {topic['cluster_id']} (Size: {topic['size']})", expanded=i < 3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Keywords")
                keywords = topic.get('keywords', [])
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No keywords extracted")
                
                st.subheader("Exemplar Case IDs")
                exemplars = topic.get('exemplar_case_ids', [])
                if exemplars:
                    for case_id in exemplars:
                        st.write(f"• {case_id}")
                else:
                    st.write("No exemplars available")
            
            with col2:
                st.subheader("Sample Tickets")
                if classified_df is not None and exemplars:
                    # Show sample tickets from this cluster
                    sample_tickets = classified_df[classified_df['case_id'].isin(exemplars)]
                    for _, ticket in sample_tickets.iterrows():
                        st.write(f"**{ticket['case_id']}**")
                        st.write(f"Subject: {ticket.get('subject_redacted', ticket.get('subject', 'N/A'))[:100]}...")
                        st.write(f"Predicted: {ticket.get('predicted_name', 'N/A')}")
                        st.write("---")

def auto_labels_review_page(classified_df: pd.DataFrame, taxonomy: Dict[str, Any]):
    """Page for reviewing and correcting auto-labels."""
    st.header("🏷️ Auto-labels Review")
    
    if classified_df is None:
        st.warning("No classified data found. Run classification first.")
        return
    
    # Get taxonomy categories for dropdown
    categories = taxonomy.get('categories', [])
    category_options = {cat['code']: cat['name'] for cat in categories}
    
    # Add "Unclassified" option
    category_options['UNCLASSIFIED'] = 'Unclassified'
    
    # Filter options
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_filter = st.selectbox(
            "Model Strategy",
            ["All"] + list(classified_df['model_strategy'].unique()),
            key="strategy_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.0, 0.1,
            key="confidence_filter"
        )
    
    with col3:
        limit = st.number_input("Number of tickets to review", 1, 1000, 50)
    
    # Apply filters
    filtered_df = classified_df.copy()
    
    if strategy_filter != "All":
        filtered_df = filtered_df[filtered_df['model_strategy'] == strategy_filter]
    
    filtered_df = filtered_df[filtered_df['confidence'] >= confidence_filter]
    
    # Limit results
    filtered_df = filtered_df.head(limit)
    
    st.write(f"Showing {len(filtered_df)} tickets for review")
    
    # Create editable dataframe
    st.subheader("Review and Correct Labels")
    
    # Initialize session state for corrections
    if 'corrections' not in st.session_state:
        st.session_state.corrections = {}
    
    # Display tickets for review
    for idx, (_, ticket) in enumerate(filtered_df.iterrows()):
        with st.container():
            st.write("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Case ID:** {ticket['case_id']}")
                st.write(f"**Subject:** {ticket.get('subject_redacted', ticket.get('subject', 'N/A'))}")
                st.write(f"**Description:** {ticket.get('description_redacted', ticket.get('description', 'N/A'))[:200]}...")
                
                # Current prediction
                st.write(f"**Current Prediction:** {ticket.get('predicted_name', 'N/A')} (Confidence: {ticket.get('confidence', 0):.2f})")
                st.write(f"**Strategy:** {ticket.get('model_strategy', 'N/A')}")
            
            with col2:
                # Correction interface
                case_id = ticket['case_id']
                
                # Get current correction or use original prediction
                current_correction = st.session_state.corrections.get(case_id, ticket.get('predicted_code', 'UNCLASSIFIED'))
                
                # Category selection
                corrected_category = st.selectbox(
                    "Correct Category",
                    list(category_options.keys()),
                    index=list(category_options.keys()).index(current_correction) if current_correction in category_options else 0,
                    key=f"category_{case_id}"
                )
                
                # Confidence adjustment
                corrected_confidence = st.slider(
                    "Confidence",
                    0.0, 1.0, ticket.get('confidence', 0.5),
                    key=f"confidence_{case_id}"
                )
                
                # Save correction
                if st.button("Save Correction", key=f"save_{case_id}"):
                    st.session_state.corrections[case_id] = corrected_category
                    st.success("Correction saved!")
    
    # Summary and export
    st.divider()
    st.subheader("Review Summary")
    
    num_corrections = len(st.session_state.corrections)
    st.write(f"Number of corrections made: {num_corrections}")
    
    if num_corrections > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Training Labels"):
                # Apply corrections to the dataframe
                corrected_df = classified_df.copy()
                
                for case_id, corrected_code in st.session_state.corrections.items():
                    mask = corrected_df['case_id'] == case_id
                    corrected_df.loc[mask, 'predicted_code'] = corrected_code
                    corrected_df.loc[mask, 'predicted_name'] = category_options.get(corrected_code, corrected_code)
                    corrected_df.loc[mask, 'model_strategy'] = 'ui_corrected'
                
                # Save training labels
                config = load_config()
                reports_dir = Path(config["paths"]["reports_dir"])
                save_training_labels(corrected_df, reports_dir)
        
        with col2:
            if st.button("Clear All Corrections"):
                st.session_state.corrections = {}
                st.rerun()

def main():
    """Main Streamlit app."""
    st.title("🎫 Ticket Taxonomy Tool")
    st.markdown("Review clusters and correct auto-labels for ticket classification")
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Load data
    reports_dir = Path(config["paths"]["reports_dir"])
    redacted_dir = Path(config["paths"]["redacted_dir"])
    
    # Load topics report
    topics_report = load_topics_report(reports_dir)
    
    # Load classified data
    classified_df = load_classified_data(redacted_dir)
    
    # Load taxonomy
    taxonomy = load_taxonomy()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Clusters Explorer", "🏷️ Auto-labels Review"])
    
    with tab1:
        clusters_explorer_page(topics_report, classified_df)
    
    with tab2:
        auto_labels_review_page(classified_df, taxonomy)
    
    # Sidebar with data info
    with st.sidebar:
        st.header("Data Status")
        
        if topics_report:
            st.success("✅ Topics report loaded")
            st.write(f"Topics: {topics_report.get('metadata', {}).get('num_topics', 0)}")
        else:
            st.error("❌ No topics report found")
        
        if classified_df is not None:
            st.success("✅ Classified data loaded")
            st.write(f"Tickets: {len(classified_df):,}")
            
            # Show strategy breakdown
            strategy_counts = classified_df['model_strategy'].value_counts()
            st.write("**Strategy Breakdown:**")
            for strategy, count in strategy_counts.items():
                st.write(f"• {strategy}: {count:,}")
        else:
            st.error("❌ No classified data found")
        
        if taxonomy:
            st.success("✅ Taxonomy loaded")
            st.write(f"Categories: {len(taxonomy.get('categories', []))}")
        else:
            st.warning("⚠️ No taxonomy found")

if __name__ == "__main__":
    main()
