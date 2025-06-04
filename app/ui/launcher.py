"""
Launcher script to start the Streamlit UI for the AI Research Agent.

Usage:
    python -m app.ui.launcher
"""

import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the directory of this script
    current_dir = Path(__file__).parent
    
    # Set the Streamlit app file path
    streamlit_app = str(current_dir / "streamlit_app.py")
    
    # Check if the file exists
    if not os.path.exists(streamlit_app):
        print(f"Error: Streamlit app file not found at {streamlit_app}")
        sys.exit(1)
    
    # Construct Streamlit CLI args
    sys.argv = [
        "streamlit", 
        "run", 
        streamlit_app,
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--browser.serverAddress=localhost"
    ]
    
    # Launch the app
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
