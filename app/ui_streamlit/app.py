"""
Streamlit UI for ticket classification with file upload and processing.
"""
import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import toml
import tempfile
import io

# Configure page
st.set_page_config(
    page_title="Linnaeus - Ticket Taxonomy Tool",
    page_icon="🎫",
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
            taxonomy_data = yaml.safe_load(f)
            return taxonomy_data if taxonomy_data else {}
    except Exception as e:
        st.error(f"Failed to load taxonomy: {e}")
        return {}

def process_uploaded_file(uploaded_file, dataset_name: str) -> bool:
    """Process an uploaded CSV file through the entire pipeline."""
    try:
        # Import pipeline modules directly
        import sys
        import os
        # Add parent directory to Python path
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try to setup spaCy model first
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except OSError:
            st.warning("Setting up spaCy model... This may take a moment on first run.")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                st.success("spaCy model ready!")
            except Exception as e:
                st.warning(f"Could not setup spaCy model: {e}. PII redaction will use regex only.")
        
        # Import modules directly from the app directory
        import importlib.util
        
        # Define module paths
        modules_to_import = ['ingest', 'redact', 'embed', 'discover', 'rules', 'classify', 'export']
        imported_modules = {}
        
        for module_name in modules_to_import:
            module_path = os.path.join(parent_dir, 'app', f'{module_name}.py')
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                imported_modules[module_name] = module
            else:
                st.error(f"Could not find module: {module_name}")
                return False
        
        # Extract modules
        ingest = imported_modules['ingest']
        redact = imported_modules['redact']
        embed = imported_modules['embed']
        discover = imported_modules['discover']
        rules = imported_modules['rules']
        classify = imported_modules['classify']
        export = imported_modules['export']
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded file temporarily
        status_text.text("📥 Saving uploaded file...")
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = raw_dir / f"{dataset_name}.csv"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        progress_bar.progress(10)
        
        # Step 1: Ingest
        status_text.text("1/7 - 📊 Ingesting CSV data...")
        warehouse_path = ingest.ingest_csv(str(csv_path), dataset_name)
        progress_bar.progress(20)
        
        # Step 2: Redact PII (optional)
        status_text.text("2/7 - 🔒 Redacting PII...")
        
        # Check if user wants to skip PII redaction for faster processing
        if st.session_state.get('skip_pii_redaction', False):
            redacted_path = warehouse_path
            st.info("⏭️ Skipped PII redaction for faster processing")
        else:
            try:
                redacted_path = redact.redact_parquet(warehouse_path, dataset_name)
            except Exception as e:
                st.warning(f"PII redaction failed: {e}. Continuing without redaction...")
                redacted_path = warehouse_path
        
        progress_bar.progress(35)
        
        # Step 3: Generate embeddings
        status_text.text("3/7 - 🧠 Generating embeddings...")
        try:
            embeddings_path, faiss_path = embed.process_embeddings(redacted_path, dataset_name)
            st.success(f"Embeddings generated: {embeddings_path}")
        except Exception as e:
            st.error(f"Embeddings generation failed: {e}")
            return False
        progress_bar.progress(50)
        
        # Step 4: Discover topics
        status_text.text("4/7 - 🎯 Discovering topics...")
        try:
            topics_report = discover.discover_topics(redacted_path, embeddings_path, dataset_name)
            st.success("Topic discovery completed")
        except Exception as e:
            st.error(f"Topic discovery failed: {e}")
            return False
        progress_bar.progress(65)
        
        # Step 5: Apply rules
        status_text.text("5/7 - 📋 Applying classification rules...")
        try:
            with_rules_path = rules.apply_rules_to_data(redacted_path, dataset_name)
            st.success("Classification rules applied")
        except Exception as e:
            st.error(f"Rules application failed: {e}")
            return False
        progress_bar.progress(75)
        
        # Step 6: Classify tickets
        status_text.text("6/7 - 🏷️ Classifying tickets...")
        try:
            classified_path = classify.classify_tickets(with_rules_path, embeddings_path, dataset_name)
            st.success("Ticket classification completed")
        except Exception as e:
            st.error(f"Classification failed: {e}")
            return False
        progress_bar.progress(85)
        
        # Step 7: Export results
        status_text.text("7/7 - 📤 Exporting results...")
        try:
            export_paths = export.export_results(classified_path, dataset_name)
            st.success("Results exported successfully")
        except Exception as e:
            st.error(f"Export failed: {e}")
            return False
        progress_bar.progress(100)
        
        status_text.text("✅ Processing complete!")
        return True
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)
        return False

def prepare_data_for_processing(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Prepare uploaded data by mapping columns and combining text fields."""
    processed_df = df.copy()
    
    # Map columns according to user selection
    for required_col, user_col in column_mapping.items():
        if user_col and user_col in processed_df.columns:
            if required_col != user_col:
                processed_df[required_col] = processed_df[user_col]
    
    # Combine multiple text fields for better analysis
    text_fields_to_combine = []
    
    # Always include description if available
    if 'description' in processed_df.columns:
        text_fields_to_combine.append('description')
    
    # Add taxonomy fields if they exist
    for col in ['Main Category', 'Issue', 'Detail', 'subject']:
        if col in processed_df.columns and col not in text_fields_to_combine:
            text_fields_to_combine.append(col)
    
    # Create combined text field for analysis
    if text_fields_to_combine:
        processed_df['combined_text'] = processed_df[text_fields_to_combine].fillna('').agg(' | '.join, axis=1)
        # Use combined text as description for embedding
        if 'combined_text' in processed_df.columns:
            processed_df['description'] = processed_df['combined_text']
    
    # Fill missing required columns with defaults
    defaults = {
        'channel': 'unknown',
        'product_line': 'unknown',
        'region': 'unknown',
        'language': 'en',
        'severity': 'medium',
        'status': 'open'
    }
    
    for col, default_value in defaults.items():
        if col not in processed_df.columns:
            processed_df[col] = default_value
    
    return processed_df

def upload_page():
    """Page for uploading and processing CSV files."""
    st.header("📤 Upload & Process Tickets")
    
    st.markdown("""
    Upload a CSV file with your support tickets. The tool will help you map your columns to the required format.
    
    **Flexible Upload**: Works with any CSV structure - just map your columns!
    """)
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your support tickets"
    )
    
    if uploaded_file is not None:
        # Show file preview
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 File Preview")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {', '.join(df.columns)}")
            
            st.dataframe(df.head(5), use_container_width=True)
            
            st.divider()
            
            # Column mapping interface
            st.subheader("🔗 Map Your Columns")
            st.markdown("Match your CSV columns to the required fields. Leave blank if you don't have that field.")
            
            col1, col2 = st.columns(2)
            
            column_mapping = {}
            available_columns = [''] + list(df.columns)
            
            with col1:
                st.markdown("**🔑 Essential Fields:**")
                column_mapping['case_id'] = st.selectbox(
                    "Ticket ID / Case ID *",
                    available_columns,
                    index=available_columns.index('case_id') if 'case_id' in available_columns else 0,
                    help="Unique identifier for each ticket"
                )
                
                column_mapping['subject'] = st.selectbox(
                    "Subject / Title",
                    available_columns,
                    index=available_columns.index('subject') if 'subject' in available_columns else 
                          available_columns.index('Issue') if 'Issue' in available_columns else 0,
                    help="Brief description or title"
                )
                
                column_mapping['description'] = st.selectbox(
                    "Description / Details",
                    available_columns,
                    index=available_columns.index('description') if 'description' in available_columns else
                          available_columns.index('Detail') if 'Detail' in available_columns else 0,
                    help="Detailed ticket description"
                )
                
                column_mapping['created_at'] = st.selectbox(
                    "Created Date / Timestamp",
                    available_columns,
                    index=available_columns.index('created_at') if 'created_at' in available_columns else 0,
                    help="When the ticket was created"
                )
            
            with col2:
                st.markdown("**📊 Optional Fields:**")
                column_mapping['channel'] = st.selectbox(
                    "Channel / Source",
                    available_columns,
                    index=available_columns.index('channel') if 'channel' in available_columns else 0
                )
                
                column_mapping['product_line'] = st.selectbox(
                    "Product / Category",
                    available_columns,
                    index=available_columns.index('product_line') if 'product_line' in available_columns else
                          available_columns.index('Main Category') if 'Main Category' in available_columns else 0
                )
                
                column_mapping['severity'] = st.selectbox(
                    "Severity / Priority",
                    available_columns,
                    index=available_columns.index('severity') if 'severity' in available_columns else 0
                )
                
                column_mapping['status'] = st.selectbox(
                    "Status",
                    available_columns,
                    index=available_columns.index('status') if 'status' in available_columns else 0
                )
            
            st.divider()
            
            # Show which taxonomy columns will be included
            taxonomy_cols = [col for col in ['Main Category', 'Issue', 'Detail'] if col in df.columns]
            if taxonomy_cols:
                st.success(f"✅ Found taxonomy columns: **{', '.join(taxonomy_cols)}** - These will be included in the analysis!")
            
            # Validate essential mappings
            if not column_mapping.get('case_id'):
                st.error("❌ Please map the Case ID field (required)")
            elif not (column_mapping.get('subject') or column_mapping.get('description')):
                st.error("❌ Please map at least Subject or Description (required for text analysis)")
            else:
                st.success("✅ Ready to process!")
                
                st.divider()
                
                # Dataset name input
                dataset_name = st.text_input(
                    "Dataset Name",
                    value="uploaded_data",
                    help="A name for this dataset (will be used in file names)"
                )
                
                # Processing options
                col1, col2 = st.columns(2)
                with col1:
                    skip_pii = st.checkbox(
                        "Skip PII Redaction", 
                        value=False,
                        help="Skip PII redaction for faster processing (less privacy protection)"
                    )
                with col2:
                    st.info("💡 Tip: Skip PII redaction for faster processing on large files")
                
                # Process button
                if st.button("🚀 Process Dataset", type="primary", use_container_width=True):
                    # Store the skip_pii option in session state
                    st.session_state['skip_pii_redaction'] = skip_pii
                    
                    # Prepare data with column mapping
                    prepared_df = prepare_data_for_processing(df, column_mapping)
                    
                    # Save prepared data
                    raw_dir = Path("data/raw")
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = raw_dir / f"{dataset_name}.csv"
                    prepared_df.to_csv(csv_path, index=False)
                    
                    # Reset file pointer and create a new file object with prepared data
                    prepared_file = io.BytesIO()
                    prepared_df.to_csv(prepared_file, index=False)
                    prepared_file.seek(0)
                    
                    with st.spinner("Processing dataset... This may take a few minutes."):
                        success = process_uploaded_file(prepared_file, dataset_name)
                    
                    if success:
                        st.success("✅ Dataset processed successfully!")
                        st.info("📊 Navigate to the **Clusters Explorer** or **Auto-labels Review** tabs to see results.")
                        # Force a rerun to refresh the data
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.exception(e)

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
        st.success(f"✅ Saved training labels to {training_path}")
        
        # Also offer CSV download
        csv = training_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Labels (CSV)",
            data=csv,
            file_name="training_labels.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Failed to save training labels: {e}")

def clusters_explorer_page(topics_report: Dict[str, Any], classified_df: pd.DataFrame):
    """Page for exploring discovered clusters."""
    st.header("🎯 Clusters Explorer")
    
    if not topics_report:
        st.warning("⚠️ No topics report found. Upload and process a CSV file first.")
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
        with st.expander(f"📌 Topic {topic['cluster_id']} - Size: {topic['size']} tickets", expanded=i < 3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔑 Keywords")
                keywords = topic.get('keywords', [])
                if keywords:
                    # Display keywords as tags
                    keyword_html = " ".join([f'<span style="background-color: #e1f5ff; padding: 5px 10px; margin: 2px; border-radius: 5px; display: inline-block;">{kw}</span>' for kw in keywords])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                else:
                    st.write("No keywords extracted")
                
                st.subheader("📋 Exemplar Case IDs")
                exemplars = topic.get('exemplar_case_ids', [])
                if exemplars:
                    for case_id in exemplars[:5]:  # Show max 5
                        st.write(f"• `{case_id}`")
                else:
                    st.write("No exemplars available")
            
            with col2:
                st.subheader("📝 Sample Tickets")
                if classified_df is not None and exemplars:
                    # Show sample tickets from this cluster
                    sample_tickets = classified_df[classified_df['case_id'].isin(exemplars[:3])]
                    for _, ticket in sample_tickets.iterrows():
                        with st.container():
                            st.write(f"**{ticket['case_id']}**")
                            subject = ticket.get('subject_redacted', ticket.get('subject', 'N/A'))
                            st.write(f"📧 {subject[:80]}{'...' if len(subject) > 80 else ''}")
                            pred_name = ticket.get('predicted_name', 'N/A')
                            conf = ticket.get('confidence', 0)
                            st.write(f"🏷️ {pred_name} ({conf:.2f})")
                            st.markdown("---")

def auto_labels_review_page(classified_df: pd.DataFrame, taxonomy: Dict[str, Any]):
    """Page for reviewing and correcting auto-labels."""
    st.header("🏷️ Auto-labels Review")
    
    if classified_df is None:
        st.warning("⚠️ No classified data found. Upload and process a CSV file first.")
        return
    
    # Get available categories from the classified data
    unique_codes = classified_df['predicted_code'].dropna().unique()
    unique_names = classified_df['predicted_name'].dropna().unique()
    
    # Build category options from actual data
    category_options = {}
    for code, name in zip(classified_df['predicted_code'].fillna('UNCLASSIFIED'), 
                         classified_df['predicted_name'].fillna('Unclassified')):
        if code not in category_options:
            category_options[code] = name
    
    # Add from taxonomy if available
    if taxonomy and 'categories' in taxonomy:
        for cat in taxonomy['categories']:
            if 'code' in cat and 'name' in cat:
                category_options[cat['code']] = cat['name']
    
    # Ensure we have at least an unclassified option
    if 'UNCLASSIFIED' not in category_options:
        category_options['UNCLASSIFIED'] = 'Unclassified'
    
    # Filter options
    st.subheader("🔍 Filter Options")
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
        limit = st.number_input("Number of tickets to review", 1, 100, 20)
    
    # Apply filters
    filtered_df = classified_df.copy()
    
    if strategy_filter != "All":
        filtered_df = filtered_df[filtered_df['model_strategy'] == strategy_filter]
    
    filtered_df = filtered_df[filtered_df['confidence'] >= confidence_filter]
    
    # Limit results
    filtered_df = filtered_df.head(limit)
    
    st.info(f"📊 Showing {len(filtered_df)} tickets for review")
    
    # Initialize session state for corrections
    if 'corrections' not in st.session_state:
        st.session_state.corrections = {}
    
    st.divider()
    
    # Display tickets for review
    for idx, (_, ticket) in enumerate(filtered_df.iterrows()):
        case_id = ticket['case_id']
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### 🎫 {case_id}")
                st.write(f"**📧 Subject:** {ticket.get('subject_redacted', ticket.get('subject', 'N/A'))}")
                
                description = ticket.get('description_redacted', ticket.get('description', 'N/A'))
                st.write(f"**📝 Description:** {description[:300]}{'...' if len(description) > 300 else ''}")
                
                # Current prediction with color coding
                confidence = ticket.get('confidence', 0)
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.markdown(f"**Current:** <span style='color: {color};'>{ticket.get('predicted_name', 'N/A')} ({confidence:.2f})</span>", unsafe_allow_html=True)
                st.write(f"**Strategy:** {ticket.get('model_strategy', 'N/A')}")
            
            with col2:
                # Get current correction or use original prediction
                current_correction = st.session_state.corrections.get(case_id, ticket.get('predicted_code', 'UNCLASSIFIED'))
                
                # Category selection
                corrected_category = st.selectbox(
                    "Correct Category",
                    list(category_options.keys()),
                    format_func=lambda x: f"{x}: {category_options[x]}",
                    index=list(category_options.keys()).index(current_correction) if current_correction in category_options else 0,
                    key=f"category_{case_id}"
                )
                
                # Save correction button
                if st.button("💾 Save", key=f"save_{case_id}", use_container_width=True):
                    st.session_state.corrections[case_id] = corrected_category
                    st.success("Saved!")
            
            st.divider()
    
    # Summary and export
    st.subheader("📊 Review Summary")
    
    num_corrections = len(st.session_state.corrections)
    st.write(f"**Corrections made:** {num_corrections}")
    
    if num_corrections > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Training Labels", type="primary", use_container_width=True):
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
            if st.button("🗑️ Clear All Corrections", use_container_width=True):
                st.session_state.corrections = {}
                st.rerun()

def main():
    """Main Streamlit app."""
    st.title("🎫 Linnaeus - Ticket Taxonomy Tool")
    st.markdown("*Automatically classify support tickets using AI and rule-based matching*")
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Ensure directories exist
    for dir_key in ['raw_dir', 'warehouse_dir', 'redacted_dir', 'models_dir', 'reports_dir']:
        Path(config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)
    
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
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🎯 Clusters Explorer", "🏷️ Auto-labels Review"])
    
    with tab1:
        upload_page()
    
    with tab2:
        clusters_explorer_page(topics_report, classified_df)
    
    with tab3:
        auto_labels_review_page(classified_df, taxonomy)
    
    # Sidebar with data info
    with st.sidebar:
        st.header("📊 Data Status")
        
        st.write("---")
        
        if topics_report:
            st.success("✅ Topics report loaded")
            st.write(f"**Topics:** {topics_report.get('metadata', {}).get('num_topics', 0)}")
        else:
            st.warning("⚠️ No topics report")
            st.caption("Upload a CSV file to get started")
        
        if classified_df is not None:
            st.success("✅ Classified data loaded")
            st.write(f"**Tickets:** {len(classified_df):,}")
            
            # Show strategy breakdown
            strategy_counts = classified_df['model_strategy'].value_counts()
            st.write("**Strategy Breakdown:**")
            for strategy, count in strategy_counts.items():
                percentage = (count / len(classified_df) * 100)
                st.write(f"• {strategy}: {count:,} ({percentage:.1f}%)")
        else:
            st.warning("⚠️ No classified data")
            st.caption("Upload a CSV file to get started")
        
        if taxonomy and taxonomy.get('categories'):
            st.success("✅ Taxonomy loaded")
            st.write(f"**Categories:** {len(taxonomy.get('categories', []))}")
        else:
            st.info("ℹ️ No taxonomy configured")
        
        st.write("---")
        st.caption("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
