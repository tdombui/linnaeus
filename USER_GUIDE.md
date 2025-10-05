# 🎫 Linnaeus - Ticket Taxonomy Tool

**A beginner-friendly tool to automatically organize and categorize your support tickets using AI.**

## 🌟 What is Linnaeus?

Linnaeus automatically reads your support tickets and:
- 🔍 **Discovers topics** - Finds natural groupings in your data
- 🏷️ **Auto-labels tickets** - Suggests categories for each ticket
- 📊 **Creates reports** - Generates insights about your support patterns
- 🎯 **Improves over time** - Learns from your feedback

Think of it as an AI assistant that helps you understand what your customers are asking about most.

## 🚀 Quick Start (5 minutes)

### Option 1: Use Online Version (Easiest)
1. Go to **https://linnaeus.streamlit.app/**
2. Upload your CSV file
3. Map your columns
4. Click "Process Dataset"
5. View your results!

### Option 2: Run Locally
```bash
# 1. Download the code
git clone https://github.com/tdombui/linnaeus.git
cd linnaeus

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -e .

# 3. Launch the web interface
.venv\Scripts\python.exe -m streamlit run app/ui_streamlit/app.py
```

## 📊 Your Data Format

Linnaeus works with CSV files containing support tickets. Your CSV should have columns like:

| Required Columns | Optional Columns |
|------------------|------------------|
| `case_id` or `ticket_id` | `channel` (email, phone, chat) |
| `subject` or `title` | `priority` (high, medium, low) |
| `description` or `details` | `status` (open, closed, pending) |
| `created_at` or `timestamp` | `category` or `product` |

**Example CSV structure:**
```csv
case_id,subject,description,created_at,priority
12345,"GPS not working","Customer reports GPS navigation is inaccurate",2024-01-15,high
12346,"Bluetooth connection issues","Phone won't connect to car stereo",2024-01-16,medium
```

## 🎯 How to Use the Tool

### Step 1: Upload Your Data
1. **Go to "Upload & Process" tab**
2. **Drag and drop your CSV file**
3. **Preview your data** - Make sure it looks right
4. **Map your columns** - Tell the tool which column is which

### Step 2: Processing Options
- **Skip PII Redaction**: ✅ Check this for faster processing on large files
- **Dataset Name**: Give your dataset a memorable name

### Step 3: Wait for Processing
The tool will:
1. 📊 **Ingest** your CSV data
2. 🔒 **Redact PII** (optional) - Remove sensitive information
3. 🧠 **Generate embeddings** - Convert text to numbers for analysis
4. 🎯 **Discover topics** - Find natural groupings
5. 🏷️ **Apply rules** - Use predefined patterns to label tickets
6. 🤖 **Classify** - Use AI to suggest categories
7. 📤 **Export results** - Generate reports and files

### Step 4: Explore Your Results

## 📈 Understanding Your Results

### 📊 Summary Report
**What it shows**: High-level overview of your data
- Total tickets processed
- Number of topics discovered
- Classification coverage
- Top categories found

**How to read**: 
- **Coverage**: % of tickets that got auto-labeled
- **High confidence**: Tickets the AI is very sure about
- **Topics**: Natural groupings found in your data

### 🎯 Clusters Explorer
**What it does**: Shows discovered topic clusters
**Why useful**: Find patterns in your support tickets

**How to use**:
1. **Browse clusters** - Each cluster represents a group of similar tickets
2. **Read representative tickets** - See examples from each cluster
3. **Understand patterns** - Notice common themes and issues
4. **Export clusters** - Save interesting clusters for further analysis

**Example clusters you might see**:
- 🔌 "Bluetooth connection issues"
- 🗺️ "GPS navigation problems" 
- 📱 "CarPlay/Android Auto issues"
- 🔊 "Audio system problems"

### 🏷️ Auto-labels Review
**What it does**: Shows tickets that were automatically classified
**Why useful**: Review and improve the auto-labeling system

**How to use**:
1. **Review suggested labels** - See what the AI suggested
2. **Accept good labels** - Click ✅ for correct suggestions
3. **Reject bad labels** - Click ❌ for incorrect suggestions
4. **Improve the system** - Your feedback helps the AI learn

**Label types**:
- **Rules-based**: Labels from predefined patterns (e.g., "bluetooth" → "Connectivity")
- **ML-based**: Labels from machine learning (more flexible)
- **Unclassified**: Tickets that need manual review

## 📁 Output Files

After processing, you'll get several files:

### 📊 Reports (HTML - Open in browser)
- **`dataset_summary.html`** - Overview dashboard
- **`dataset_eval_report.html`** - Detailed analysis
- **`dataset_topics.html`** - Topic discovery results

### 📋 Data Files
- **`dataset.csv`** - Classified tickets with labels
- **`dataset.parquet`** - Same data in efficient format
- **`dataset_summary.json`** - Machine-readable summary

### 🎯 Topic Files
- **`dataset_topics.json`** - Discovered topics and clusters
- **`coverage_analysis.png`** - Visualization of classification coverage

## 🔧 Advanced Features

### 📝 Custom Rules
You can add your own classification rules in `rules/keyword_rules.csv`:

```csv
category,keywords,pattern,enabled
Connectivity,bluetooth;wifi;connection,bluetooth|wifi|connection,true
Navigation,gps;maps;navigation,gps|maps|navigation,true
Audio,sound;speaker;audio,sound|speaker|audio,true
```

### 🎨 Custom Taxonomy
Define your own categories in `configs/taxonomy.yaml`:

```yaml
categories:
  - name: "Technical Issues"
    subcategories:
      - "Software Problems"
      - "Hardware Issues"
  - name: "User Experience"
    subcategories:
      - "Interface Issues"
      - "Performance Problems"
```

## 🚨 Troubleshooting

### Common Issues

**❌ "Processing failed"**
- **Cause**: Usually a function name error
- **Fix**: The tool is being updated automatically

**❌ "spaCy model not found"**
- **Cause**: Missing language model
- **Fix**: Tool will use regex-only PII redaction (still works!)

**❌ "File too large"**
- **Cause**: Very large CSV files
- **Fix**: Use local version or split your file

**❌ "No topics found"**
- **Cause**: Data too diverse or small
- **Fix**: Try with more data or adjust clustering parameters

### Performance Tips

**For Large Files (50MB+)**:
1. ✅ Check "Skip PII Redaction"
2. 🏠 Use local version instead of cloud
3. 📊 Split into smaller files if needed

**For Better Results**:
1. 📝 Clean your data (remove duplicates, fix encoding)
2. 🎯 Include good descriptions in your tickets
3. 📊 Process at least 1000+ tickets for meaningful clusters

## 🔄 Updating the Tool

### Online Version
- **Automatic**: Pushes to GitHub automatically update the cloud version
- **Manual**: Visit https://linnaeus.streamlit.app/ - updates appear in ~3 minutes

### Local Version
```bash
git pull  # Get latest updates
pip install -e .  # Reinstall if needed
```

## 📞 Getting Help

### 🐛 Found a Bug?
1. Check the troubleshooting section above
2. Look at the terminal/console for error messages
3. Try the local version if cloud version has issues

### 💡 Want to Improve the Tool?
- **Add custom rules** in `rules/keyword_rules.csv`
- **Adjust settings** in `configs/config.toml`
- **Contribute code** by forking the repository

### 📚 Need More Help?
- **Check the logs** in your terminal for detailed error messages
- **Try with sample data** first to test the tool
- **Start small** with a subset of your data

## 🎉 Success Stories

**Example Results**:
- 📊 **61,888 tickets** processed
- 🎯 **2,751 topics** discovered
- 🏷️ **6.1% auto-classified** with rules
- ⚡ **Processing time**: ~10 minutes for large datasets

**What You'll Learn**:
- 🔍 Most common customer issues
- 📈 Trending problems over time
- 🎯 Which categories need attention
- 📊 Support team workload patterns

---

**Ready to get started?** 🚀

1. **Upload your CSV** to https://linnaeus.streamlit.app/
2. **Follow the steps** above
3. **Explore your results** and discover insights!

**Happy analyzing!** 📊✨
