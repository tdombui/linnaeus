# Ticket Taxonomy Tool

A comprehensive tool for automatically classifying support tickets using a combination of rule-based matching and machine learning. The tool processes CSV ticket data, applies PII redaction, generates embeddings, discovers topic clusters, and provides an interactive UI for reviewing and correcting classifications.

---

## 🚀 Quick Start for Beginners (Total Code Noobs Welcome!)

### Step 1: Set Up Your Environment (One-Time Setup)

**Windows PowerShell:**
```powershell
# Navigate to the project folder
cd C:\Users\tdomb\Desktop\Code\linnaeus

# Create a virtual environment (isolated Python environment)
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install all required packages
pip install -e .

# Download the language model for PII detection
python -m spacy download en_core_web_sm
```

**What this does:** Sets up an isolated Python environment with all the tools needed.

### Step 2: Run the Complete Pipeline (Process Your Tickets)

**Option A: Use the sample data we created**
```powershell
# Make sure you're still in the project folder and venv is activated
.venv\Scripts\activate

# Step 1: Import the CSV file
python -m app.ingest data/raw/sample_tickets.csv sample_run

# Step 2: Remove personal information (PII redaction)
python -m app.redact data/warehouse/sample_run.parquet sample_run

# Step 3: Generate embeddings (convert text to numbers for AI)
python -m app.embed data/redacted/sample_run_redacted.parquet sample_run

# Step 4: Discover topics (find patterns in tickets)
python -m app.discover data/redacted/sample_run_redacted.parquet artifacts/models/embeddings.npy sample_run

# Step 5: Apply classification rules
python -m app.rules data/redacted/sample_run_redacted.parquet sample_run

# Step 6: Classify all tickets
python -m app.classify data/redacted/sample_run_with_rules.parquet artifacts/models/embeddings.npy sample_run

# Step 7: Export results with evaluation (THE MAGIC COMMAND!)
python -m app.export data/redacted/sample_run_classified.parquet sample_run_final --with-eval
```

**What you get:** All your results in `artifacts/reports/` folder!

### Step 3: View Your Results

You have **two options** for viewing results:

#### **Option A: View HTML Reports (Easiest - No Server Needed)**
Static web-based dashboards that open in your browser. Perfect for viewing metrics and sharing with stakeholders.

```powershell
# Open the summary report in your browser
explorer artifacts\reports\sample_run_final_summary.html

# Open the evaluation report
explorer artifacts\reports\sample_run_final_eval_report.html

# View the visualization dashboard
explorer artifacts\reports\coverage_analysis.png
```

**What you get:** Professional dashboards showing coverage, confidence metrics, category distribution, and visualizations.

---

#### **Option B: Launch Interactive Streamlit Web UI (For Making Corrections)**
Web-based application for reviewing and correcting ticket classifications interactively.

**Step 1: Configure Streamlit (One-Time Setup)**
```powershell
# Create Streamlit config directory
mkdir .streamlit

# Create config file to skip email prompt
echo [browser] > .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [server] >> .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo port = 8501 >> .streamlit\config.toml
```

**Step 2: Start the Web Server**
```powershell
# Make sure virtual environment is activated
.venv\Scripts\activate

# Start the Streamlit web server
.venv\Scripts\python.exe -m streamlit run app/ui_streamlit/app.py

# You should see output like:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
```

**Step 3: Open in Browser**
- Open your browser and go to: **`http://localhost:8501`**
- You should see the Ticket Taxonomy Tool web interface

**What you get:**
- **Clusters Explorer Tab**: Browse discovered topics with keywords and examples
- **Auto-labels Review Tab**: Review and correct 10 classified tickets
- **Interactive Corrections**: Use dropdown to fix misclassifications
- **Export Training Labels**: Generate `training_labels.parquet` from your corrections

**To Stop the Server:**
```powershell
# Press Ctrl+C in the terminal, or:
taskkill /F /IM python.exe
```

---

**🎯 Which Option Should You Use?**

| Option | Best For | Interactive? |
|--------|----------|--------------|
| **HTML Reports** | Viewing metrics, sharing results | ❌ Static |
| **Streamlit UI** | Correcting classifications, active learning | ✅ Interactive |

**Typical Workflow:**
1. Run the command-line pipeline (Step 2) to process all tickets
2. View HTML reports (Option A) to see overall performance
3. Launch Streamlit UI (Option B) to review and correct specific tickets
4. Export corrected labels for model improvement

### Step 4: Use Your Own Data

1. Put your CSV file in `data/raw/your_tickets.csv`
2. Make sure it has these columns: `case_id,created_at,subject,description,channel,product_line,region,language,severity,status`
3. Replace `sample_run` with `your_tickets` in all the commands above

### Common Issues & Solutions

**Problem: "No module named 'streamlit'"**
- Solution: Make sure you activated the virtual environment first: `.venv\Scripts\activate`

**Problem: "Streamlit won't start"**
- Solution: Use the full path: `.venv\Scripts\python.exe -m streamlit run app/ui_streamlit/app.py`
- When it asks for email, just press Enter

**Problem: "localhost refused to connect"**
- Solution: The server isn't running. Check your terminal where you ran streamlit - it might be waiting for you to press Enter

**Problem: "Command not found"**
- Solution: Make sure you're in the project folder: `cd C:\Users\tdomb\Desktop\Code\linnaeus`

### What Each Step Does (ELI5)

1. **Ingest**: Reads your CSV and converts it to a faster format (Parquet)
2. **Redact**: Removes personal info like emails and phone numbers
3. **Embed**: Converts text into numbers that computers can understand
4. **Discover**: Finds groups of similar tickets (like organizing a messy desk)
5. **Rules**: Applies your predefined classification rules (if ticket mentions "CarPlay", it's a connection issue)
6. **Classify**: Uses AI to categorize tickets that didn't match any rules
7. **Export**: Creates reports and files ready to use in Excel, BI tools, or Salesforce

---

## Features

- **CSV Ingestion**: Validates and converts CSV data to Parquet format with schema validation
- **PII Redaction**: Automatically redacts emails, phones, VINs, and other PII using regex and spaCy NER
- **Topic Discovery**: Uses BERTopic or UMAP+HDBSCAN to discover natural topic clusters
- **Rule-based Classification**: Applies regex rules for high-confidence classifications
- **ML Classification**: Trains ensemble models on embeddings for remaining tickets
- **Interactive UI**: Streamlit interface for reviewing clusters and correcting labels
- **Export Ready**: Generates CSV/Parquet files ready for BI tools and Salesforce

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install dependencies
make setup

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Or manually install
pip install -e .
python -m spacy download en_core_web_sm
```

### 2. Run Complete Pipeline

```bash
# Run full pipeline on your CSV file
make pipeline CSV=data/raw/your_tickets.csv NAME=tickets

# Or test with sample data
make test
make pipeline CSV=data/raw/sample.csv NAME=sample
```

### 3. Review Results

```bash
# Launch interactive UI
make ui
```

## Project Structure

```
.
├─ app/
│  ├─ ingest.py            # CSV→Parquet + schema validation + dedupe
│  ├─ redact.py            # spaCy/regex PII masking → *_redacted fields
│  ├─ embed.py             # build embeddings + FAISS index
│  ├─ discover.py          # UMAP/HDBSCAN or BERTopic → clusters + keywords
│  ├─ rules.py             # load/apply regex rules (CSV)
│  ├─ classify.py          # ensemble: rules → classical (LogReg/XGB)
│  ├─ export.py            # write labels + reports to artifacts/
│  └─ ui_streamlit/
│     └─ app.py            # minimal review UI
├─ configs/
│  ├─ config.toml
│  └─ taxonomy.yaml        # starter taxonomy
├─ rules/
│  └─ keyword_rules.csv
├─ data/
│  ├─ raw/                 # drop CSVs here
│  ├─ warehouse/           # Parquet canonical
│  └─ redacted/            # Parquet redacted
├─ artifacts/
│  ├─ models/
│  └─ reports/
├─ pyproject.toml
├─ Makefile
└─ README.md
```

## Input Format

Your CSV should have these required columns:

- `case_id` (str, unique, required)
- `created_at` (timestamp, UTC)
- `subject` (str), `description` (str)
- `current_category` (str, optional) - for training
- `resolution_code` (str, optional)
- `channel` (str), `product_line` (str), `region` (str), `language` (str)
- `severity` (str), `status` (str)
- `close_reason` (str, optional), `agent_team` (str, optional)

## Output Formats

### Topics Report (`artifacts/reports/*_topics.json`)
```json
{
  "metadata": {
    "total_tickets": 1000,
    "num_topics": 15,
    "method": "bertopic"
  },
  "topics": [
    {
      "cluster_id": 0,
      "size": 150,
      "keywords": ["carplay", "android auto", "connection"],
      "exemplar_case_ids": ["T001", "T002", "T003"]
    }
  ]
}
```

### Labels Export (`artifacts/reports/labels.csv`)
```csv
case_id,predicted_code,predicted_name,confidence,model_strategy,version,decided_by,decided_at_utc
T001,CONN.CARPLAY_AA,CarPlay/Android Auto,0.92,rules,1.0,ensemble_classifier,2024-01-01T10:00:00Z
T002,NAV.GPS,GPS Accuracy,0.88,rules,1.0,ensemble_classifier,2024-01-01T10:00:00Z
```

## Step-by-Step Usage

### 1. Ingest Data
```bash
make ingest CSV=data/raw/tickets.csv NAME=tickets
```

### 2. Apply PII Redaction
```bash
make redact INPUT=data/warehouse/tickets.parquet NAME=tickets
```

### 3. Generate Embeddings & Discover Topics
```bash
make discover INPUT=data/redacted/tickets_redacted.parquet NAME=tickets
```

### 4. Apply Rules & Classification
```bash
make classify INPUT=data/redacted/tickets_redacted.parquet EMBEDDINGS=artifacts/models/tickets_embeddings.npy NAME=tickets
```

### 5. Launch Review UI
```bash
make ui
```

## Configuration

### `configs/config.toml`
```toml
[paths]
raw_dir = "data/raw"
warehouse_dir = "data/warehouse"
redacted_dir = "data/redacted"
artifacts_dir = "artifacts"

[privacy]
redact_pii = true
send_data_externally = false

[nlp]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
use_bertopic = true
min_cluster_size = 25

[classifier]
confidence_threshold = 0.70
```

### `rules/keyword_rules.csv`
```csv
rule_id,enabled,pattern,case_sensitive,assign_code,assign_name,confidence,notes
R001,true,"(carplay|apple carplay|android auto)",false,"CONN.CARPLAY_AA","CarPlay/Android Auto",0.92,"Projection"
R002,true,"(usb.*map.*update|map.*update.*usb)",false,"NAV.UPDATE_FAIL","Map Update Failure",0.90,"USB map update"
```

## Streamlit UI

The UI provides two main tabs:

1. **Clusters Explorer**: Browse discovered topic clusters with keywords and exemplar tickets
2. **Auto-labels Review**: Review and correct classification results with dropdown taxonomy

### Key Features:
- Filter by model strategy and confidence threshold
- Correct predictions with dropdown taxonomy
- Export training labels for model improvement
- Real-time statistics and coverage metrics

## Dependencies

- Python 3.11+
- pandas, pyarrow, pydantic
- sentence-transformers, faiss-cpu
- bertopic, umap-learn, hdbscan
- scikit-learn, spacy
- streamlit, toml

## Performance

- **Ingestion**: ~1k tickets/second
- **Redaction**: ~500 tickets/second
- **Embeddings**: ~100 tickets/second (depends on model)
- **Clustering**: ~50 tickets/second
- **Classification**: ~1k tickets/second

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Memory issues with large datasets**
   - Reduce `min_cluster_size` in config
   - Use smaller embedding model
   - Process in batches

3. **No topics discovered**
   - Check if embeddings were generated
   - Reduce `min_cluster_size`
   - Try different clustering method

### Logs
All modules use Python logging. Set log level in code or environment:
```bash
export PYTHONPATH=.
python -m app.ingest data/raw/tickets.csv
```

## Development

### Adding New Rules
1. Edit `rules/keyword_rules.csv`
2. Add regex patterns with appropriate confidence scores
3. Test with `make classify`

### Customizing Taxonomy
1. Edit `configs/taxonomy.yaml`
2. Add new categories with codes and descriptions
3. Update rules to use new codes

### Extending Classification
1. Modify `app/classify.py` to add new models
2. Update ensemble logic in `classify_tickets()`
3. Test with existing data

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Open an issue with sample data and error details
