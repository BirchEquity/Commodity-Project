# 🎯 Commodity Trading Analytics Platform

A complete analytics platform for commodity arbitrage analysis with interactive dashboard.

## 🚀 **Quick Start (3 Steps)**

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

## 🔄 **Force Refresh (When Data Changes)**

If you've updated `ticker_metadata.csv` and want to completely reprocess everything:

```bash
python run_analytics.py --force-refresh
```

This will:
- Remove all processed data files
- Reprocess everything from scratch
- Launch the dashboard with fresh data

## 📁 **Project Structure (Simplified)**

```
📁 Commodity Trading Analytics
├── 🚀 run_analytics.py          # ONE FILE TO RUN EVERYTHING
├── 📊 dashboard.py              # Interactive dashboard
├── 🔧 main.py                   # Analytics pipeline
├── 📁 data/
│   ├── raw/                     # Your Excel files go here
│   └── processed/               # Clean data
├── 📁 outputs/                  # Results and charts
└── 📋 requirements.txt          # Dependencies
```

## 📊 **What You Get**

### Dashboard Features
- **Trading View**: Interactive arbitrage analysis with charts
- **Under the Hood**: Data transparency and methodology
- **Export**: Download results as CSV
- **Refresh Button**: Force reload data in dashboard

### Key Results
- **Dubai-Brent**: 38.6% probability of arbitrage opening
- **Risk Analysis**: Monte Carlo simulations with probability distributions
- **Historical Data**: 2,776 trading days analyzed

## 🔧 **Updating Data**

### To Update Ticker Metadata
1. Edit `ticker_metadata.csv`
2. Run `python run_analytics.py --force-refresh` to completely reprocess
3. Dashboard will automatically update

### To Add New Data
1. Put new Excel files in `data/` folder
2. Update `ticker_metadata.csv` if needed
3. Run `python run_analytics.py --force-refresh`

## 🎯 **For Recruiters**

This platform demonstrates:
- **Market Knowledge**: Real arbitrage analysis
- **Technical Skills**: Python, data science, web development
- **Business Acumen**: Risk management and trading insights
- **Professional Standards**: Clean, production-ready code

## 🆘 **Troubleshooting**

### Dashboard Won't Start
```bash
pip install streamlit plotly pandas numpy matplotlib seaborn
python run_analytics.py
```

### Data Not Updating
1. Make sure you're in the project root directory
2. Run `python run_analytics.py --force-refresh`
3. Check that `ticker_metadata.csv` is saved

### Old Data Still Showing
Use the force refresh option:
```bash
python run_analytics.py --force-refresh
```

### Port Already in Use
The dashboard will automatically find an available port.

## 📈 **Key Files Explained**

- **`run_analytics.py`**: The only file you need to run
- **`dashboard.py`**: Interactive web interface
- **`main.py`**: Analytics pipeline (runs automatically)
- **`ticker_metadata.csv`**: Configure which tickers to analyze

---

**That's it! Just run `python run_analytics.py` and you're done!** 🎉
