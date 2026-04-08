"""
EntryPoint for defining WMS Dashboard
Usage: streamlit run dashboard.py
"""
import sys
import os
# Ensure the src module can be resolved
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ui.app import main

if __name__ == "__main__":
    main()
