"""
Setup script to download spaCy model for Streamlit Cloud deployment.
"""
import subprocess
import sys

def setup_spacy():
    """Download and install the spaCy English model."""
    try:
        print("Downloading spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("spaCy model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download spaCy model: {e}")
        return False

if __name__ == "__main__":
    setup_spacy()
