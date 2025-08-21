import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def load_metadata_from_csv():
    """Load metadata from ticker_metadata.csv file"""
    try:
        df = pd.read_csv("ticker_metadata.csv")
        metadata = {}
        for _, row in df.iterrows():
            metadata[row['Ticker']] = {
                "Commodity": row['Commodity'],
                "Region": row['Region'],
                "Unit": row['Unit']
            }
        print(f"Loaded metadata for {len(metadata)} tickers from ticker_metadata.csv")
        return metadata
    except FileNotFoundError:
        print("Warning: ticker_metadata.csv not found. Using default metadata.")
        # Fallback to default metadata if CSV doesn't exist
        return get_default_metadata()

def get_default_metadata():
    """Default metadata dictionaries as fallback"""
    # Commodity mapping
    commodity_meta = {
        "CL1 Comdty": {"Commodity": "Crude Oil", "Region": "USA", "Unit": "USD/bbl"},
        "CO1 Comdty": {"Commodity": "Crude Oil", "Region": "North Sea", "Unit": "USD/bbl"},
        "DBL1 Comdty": {"Commodity": "Crude Oil", "Region": "Middle East", "Unit": "USD/bbl"},
        "W61 Comdty": {"Commodity": "Gasoil", "Region": "Singapore", "Unit": "USD/ton"},
        "QSU5 Comdty": {"Commodity": "Gasoil", "Region": "Rotterdam", "Unit": "USD/ton"},
        "XB1 Comdty": {"Commodity": "Gasoline", "Region": "US Gulf", "Unit": "USD/ton"},
        "IHW1 Comdty": {"Commodity": "Gasoline", "Region": "Europe", "Unit": "USD/ton"},
        "NGA Comdty": {"Commodity": "LNG", "Region": "USA", "Unit": "USD/MMBtu"},
        "TZTA Comdty": {"Commodity": "LNG", "Region": "Europe", "Unit": "USD/MMBtu"},
    }

    # Freight mapping
    freight_meta = {
        "BIDY Index": {"Commodity": "Freight", "Region": "Baltic Dirty Tanker", "Unit": "Index"},
        "CGD1 Comdty": {"Commodity": "Freight", "Region": "Coal Route", "Unit": "USD/ton"},
    }

    # FX mapping
    fx_meta = {
        "USDEUR Curncy": {"Commodity": "FX", "Region": "Europe/US", "Unit": "EUR"},
        "USDSGD Curncy": {"Commodity": "FX", "Region": "Singapore/US", "Unit": "SGD"},
        "JPYUSD Curncy": {"Commodity": "FX", "Region": "Japan/US", "Unit": "USD"},
        "GBPUSD Curncy": {"Commodity": "FX", "Region": "UK/US", "Unit": "USD"},
        "EURSGD Curncy": {"Commodity": "FX", "Region": "Europe/Singapore", "Unit": "SGD"},
    }
    
    # Combine all metadata
    all_meta = {**commodity_meta, **freight_meta, **fx_meta}
    return all_meta

def clean_price_value(value):
    """Clean price values, handling '#N/A N/A' and other problematic values"""
    if pd.isna(value) or value == "#N/A N/A" or value == "#N/A" or value == "N/A":
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def reshape_file(file_path, meta_dict, date_column="Dates"):
    """Load and reshape Excel file from wide to long format"""
    print(f"Processing {file_path}...")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Check if date column exists
    if date_column not in df.columns:
        print(f"Warning: {date_column} column not found in {file_path}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Convert to long format
    df_long = df.melt(id_vars=[date_column], var_name="Ticker", value_name="Price")
    
    # Clean date column
    df_long[date_column] = pd.to_datetime(df_long[date_column], errors='coerce')
    
    # Clean price values
    df_long["Price"] = df_long["Price"].apply(clean_price_value)
    
    # Add metadata
    df_long["Commodity"] = df_long["Ticker"].map(lambda x: meta_dict.get(x, {}).get("Commodity"))
    df_long["Region"] = df_long["Ticker"].map(lambda x: meta_dict.get(x, {}).get("Region"))
    df_long["Unit"] = df_long["Ticker"].map(lambda x: meta_dict.get(x, {}).get("Unit"))
    
    # Rename date column to standard "Date"
    df_long = df_long.rename(columns={date_column: "Date"})
    
    # Remove rows with missing metadata (tickers not in our mapping)
    missing_tickers = df_long[df_long["Commodity"].isna()]["Ticker"].unique()
    if len(missing_tickers) > 0:
        print(f"Warning: Found tickers not in metadata: {missing_tickers}")
    
    # Remove rows with missing dates or prices
    df_long = df_long.dropna(subset=["Date", "Price"])
    
    print(f"  - Loaded {len(df_long)} rows")
    print(f"  - Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")
    print(f"  - Tickers: {df_long['Ticker'].nunique()}")
    
    return df_long

def main():
    """Main processing function"""
    print("Starting market data processing...")
    print("=" * 50)
    
    # Load metadata from CSV file (or use defaults if not found)
    all_metadata = load_metadata_from_csv()
    
    # Step 2: Load & Reshape Each File
    commodities = reshape_file("data/commodity.xlsx", all_metadata)
    freight = reshape_file("data/freight.xlsx", all_metadata)
    fx = reshape_file("data/fx.xlsx", all_metadata)
    
    # Step 3: Combine All Data
    print("\nCombining all datasets...")
    all_data = pd.concat([commodities, freight, fx], ignore_index=True)
    
    # Sort by date and ticker
    all_data = all_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    
    # Step 4: Final Structure
    print(f"\nFinal dataset summary:")
    print(f"  - Total rows: {len(all_data):,}")
    print(f"  - Date range: {all_data['Date'].min()} to {all_data['Date'].max()}")
    print(f"  - Unique tickers: {all_data['Ticker'].nunique()}")
    print(f"  - Commodities: {all_data['Commodity'].unique()}")
    print(f"  - Regions: {all_data['Region'].nunique()}")
    
    # Show sample of final data
    print(f"\nSample of final data:")
    print(all_data.head(10))
    
    # Ensure data/raw directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Save clean dataset to the correct location
    output_file = "data/raw/market_data_clean.csv"
    all_data.to_csv(output_file, index=False)
    print(f"\nClean dataset saved to: {output_file}")
    
    return all_data

if __name__ == "__main__":
    result = main()
