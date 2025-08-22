# Commodity Trading Analytics WebApp

## Reflections
This project was built to further understand the structure and intricacies of physical commodity trading.


### 1. Run Everything
```bash
python run_analytics.py
```

This single command will:
- Install all required packages
- Run the complete analytics pipeline
- Launch the interactive dashboard

### 2. View Dashboard
The dashboard will automatically open in your browser at `http://localhost:8501`

### 3. Stop Dashboard
Press `Ctrl+C` in the terminal to stop the dashboard

## Force Refresh (When Data Changes)

If you've updated `ticker_metadata.csv` and want to completely reprocess everything:

```bash
python run_analytics.py --force-refresh
```

This will:
- Remove all processed data files
- Reprocess everything from scratch
- Launch the dashboard with fresh data

##  **Project Structure (Simplified)**

```
📁 Commodity Trading Analytics
├──  run_analytics.py          # ONE FILE TO RUN EVERYTHING
├──  dashboard.py              # Interactive dashboard
├──  main.py                   # Analytics pipeline
├──  data/
│   ├── raw/                     # Your Excel files go here
│   └── processed/               # Clean data
├──  outputs/                  # Results and charts
└──  requirements.txt          # Dependencies
```

##  **What You Get**

### Dashboard Features
- **Trading View**: Interactive arbitrage analysis with charts
- **Under the Hood**: Data transparency and methodology
- **Export**: Download results as CSV
- **Refresh Button**: Force reload data in dashboard
