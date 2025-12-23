"""
Helper script to run the Streamlit application.

This script provides a convenient way to launch the hotel cancellation
prediction web application.

Usage:
    python run_app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    
    app_path = Path("app/streamlit_app.py")
    
    if not app_path.exists():
        print(f"❌ Error: Application file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Hotel Cancellation Predictor...")
    print("📍 The application will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
