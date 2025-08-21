"""
Commodity Trading Analytics - Single Script Runner

This script runs the complete analytics pipeline and launches the dashboard.
Just run this one file to get everything working!
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully")
    except subprocess.CalledProcessError:
        print("Failed to install packages. Please run: pip install -r requirements.txt")
        return False
    return True

def force_refresh():
    """Force refresh by removing all processed data"""
    print("🔄 Force refreshing all data...")
    
    # Remove processed data files
    files_to_remove = [
        "data/raw/market_data_clean.csv",
        "data/processed/normalized_data.csv",
        "outputs/arb_results.csv",
        "outputs/simulation_results.csv",
        "outputs/monte_carlo_distributions.png",
        "outputs/risk_comparison.png",
        "reports/analytics_summary.txt"
    ]
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"   Removed: {file_path}")
    
    print("✅ All processed data cleared")

def run_pipeline():
    """Run the main analytics pipeline"""
    print("\nRunning analytics pipeline...")
    try:
        subprocess.check_call([sys.executable, "main.py"])
        print("Analytics pipeline completed")
        return True
    except subprocess.CalledProcessError:
        print("Pipeline failed. Check the error messages above.")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\nLaunching dashboard...")
    print("Dashboard will open in your browser at: http://localhost:8501")
    print("To stop the dashboard, press Ctrl+C in this terminal")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\nDashboard stopped")

def main():
    """Main function to run everything"""
    print("Commodity Trading Analytics Platform")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("Error: main.py not found!")
        print(" Please run this script from the project root directory")
        return
    
    # Check for force refresh argument
    if len(sys.argv) > 1 and sys.argv[1] == "--force-refresh":
        force_refresh()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run pipeline
    if not run_pipeline():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
