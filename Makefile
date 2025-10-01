# Ticket Taxonomy Tool Makefile

.PHONY: setup install clean ingest redact discover classify ui help

# Default target
help:
	@echo "Ticket Taxonomy Tool - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Create virtual environment and install dependencies"
	@echo "  install        - Install the package in development mode"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Data Processing Pipeline:"
	@echo "  ingest         - Ingest CSV to Parquet with validation"
	@echo "  redact         - Apply PII redaction to data"
	@echo "  discover       - Generate embeddings and discover topics"
	@echo "  classify       - Apply rules and ML classification"
	@echo "  export         - Export final labels and reports"
	@echo "  export-eval    - Export labels + evaluation in one command"
	@echo "  evaluate       - Evaluate classification results and generate reports"
	@echo ""
	@echo "UI:"
	@echo "  ui             - Launch Streamlit UI for review"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  pipeline       - Run complete pipeline (ingest -> export)"
	@echo ""
	@echo "Examples:"
	@echo "  make ingest CSV=data/raw/tickets.csv"
	@echo "  make discover"
	@echo "  make evaluate INPUT=data/redacted/dataset_classified.parquet"
	@echo "  make ui"

# Setup virtual environment and install dependencies
setup:
	python -m venv .venv
	.venv\Scripts\activate && python -m pip install -U pip
	.venv\Scripts\activate && pip install -e .
	.venv\Scripts\activate && python -m spacy download en_core_web_sm
	@echo "Setup complete! Activate with: .venv\Scripts\activate"

# Install package in development mode
install:
	pip install -e .

# Clean up generated files
clean:
	rm -rf data/warehouse/*.parquet
	rm -rf data/redacted/*.parquet
	rm -rf artifacts/models/*.npy
	rm -rf artifacts/models/*.bin
	rm -rf artifacts/models/*.pkl
	rm -rf artifacts/reports/*.json
	rm -rf artifacts/reports/*.html
	rm -rf artifacts/reports/*.csv
	rm -rf artifacts/reports/*.parquet
	@echo "Cleaned up generated files"

# Ingest CSV to Parquet
ingest:
	@if [ -z "$(CSV)" ]; then \
		echo "Usage: make ingest CSV=path/to/file.csv [NAME=dataset_name]"; \
		echo "Example: make ingest CSV=data/raw/tickets.csv NAME=tickets"; \
		exit 1; \
	fi
	python -m app.ingest $(CSV) $(NAME)

# Apply PII redaction
redact:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make redact INPUT=path/to/input.parquet [NAME=dataset_name]"; \
		echo "Example: make redact INPUT=data/warehouse/dataset.parquet NAME=dataset"; \
		exit 1; \
	fi
	python -m app.redact $(INPUT) $(NAME)

# Generate embeddings and discover topics
discover:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make discover INPUT=path/to/redacted.parquet EMBEDDINGS=path/to/embeddings.npy [NAME=dataset_name]"; \
		echo "Example: make discover INPUT=data/redacted/dataset_redacted.parquet EMBEDDINGS=artifacts/models/dataset_embeddings.npy NAME=dataset"; \
		exit 1; \
	fi
	python -m app.embed $(INPUT) $(NAME)
	python -m app.discover $(INPUT) artifacts/models/$(NAME)_embeddings.npy $(NAME)

# Apply rules and classification
classify:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make classify INPUT=path/to/with_rules.parquet EMBEDDINGS=path/to/embeddings.npy [NAME=dataset_name]"; \
		echo "Example: make classify INPUT=data/redacted/dataset_with_rules.parquet EMBEDDINGS=artifacts/models/dataset_embeddings.npy NAME=dataset"; \
		exit 1; \
	fi
	python -m app.rules $(INPUT) $(NAME)
	python -m app.classify data/redacted/$(NAME)_with_rules.parquet $(EMBEDDINGS) $(NAME)
	python -m app.export data/redacted/$(NAME)_classified.parquet $(NAME)

# Launch Streamlit UI
ui:
	streamlit run app/ui_streamlit/app.py

# Evaluate classification results
evaluate:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make evaluate INPUT=path/to/classified.parquet [OUTPUT=evaluation]"; \
		echo "Example: make evaluate INPUT=data/redacted/dataset_classified.parquet OUTPUT=dataset_eval"; \
		exit 1; \
	fi
	python -m app.evaluate $(INPUT) $(OUTPUT)

# Export with evaluation (one command for labels + evaluation)
export-eval:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make export-eval INPUT=path/to/classified.parquet [OUTPUT=labels]"; \
		echo "Example: make export-eval INPUT=data/redacted/dataset_classified.parquet OUTPUT=dataset"; \
		exit 1; \
	fi
	python -m app.export $(INPUT) $(OUTPUT) --with-eval

# Run complete pipeline
pipeline:
	@if [ -z "$(CSV)" ]; then \
		echo "Usage: make pipeline CSV=path/to/file.csv [NAME=dataset_name]"; \
		echo "Example: make pipeline CSV=data/raw/tickets.csv NAME=tickets"; \
		exit 1; \
	fi
	@echo "Running complete pipeline for $(CSV)..."
	@echo "Step 1: Ingesting CSV..."
	make ingest CSV=$(CSV) NAME=$(NAME)
	@echo "Step 2: Applying PII redaction..."
	make redact INPUT=data/warehouse/$(NAME).parquet NAME=$(NAME)
	@echo "Step 3: Generating embeddings and discovering topics..."
	make discover INPUT=data/redacted/$(NAME)_redacted.parquet NAME=$(NAME)
	@echo "Step 4: Applying rules and classification..."
	make classify INPUT=data/redacted/$(NAME)_redacted.parquet EMBEDDINGS=artifacts/models/$(NAME)_embeddings.npy NAME=$(NAME)
	@echo "Pipeline complete! Check artifacts/reports/ for results."

# Quick test with sample data
test:
	@echo "Creating sample data for testing..."
	@mkdir -p data/raw
	@echo "case_id,created_at,subject,description,channel,product_line,region,language,severity,status" > data/raw/sample.csv
	@echo "T001,2024-01-01T10:00:00Z,CarPlay not working,My CarPlay connection keeps disconnecting,phone,infotainment,US,en,high,open" >> data/raw/sample.csv
	@echo "T002,2024-01-01T11:00:00Z,GPS showing wrong location,The GPS is showing me in the wrong location,nav,navigation,US,en,medium,open" >> data/raw/sample.csv
	@echo "T003,2024-01-01T12:00:00Z,Bluetooth issues,Bluetooth keeps disconnecting from my phone,phone,infotainment,US,en,high,open" >> data/raw/sample.csv
	@echo "T004,2024-01-01T13:00:00Z,Map update failed,USB map update failed to install,nav,navigation,US,en,medium,open" >> data/raw/sample.csv
	@echo "T005,2024-01-01T14:00:00Z,Screen frozen,The infotainment screen is frozen,display,infotainment,US,en,high,open" >> data/raw/sample.csv
	@echo "Sample data created at data/raw/sample.csv"
	@echo "Run: make pipeline CSV=data/raw/sample.csv NAME=sample"
