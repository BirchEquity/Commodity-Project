"""
Unit Normalization Module for Commodity Trading Analytics

This module normalizes all commodity prices into a common comparable unit (USD per barrel).
This is essential for arbitrage analysis and cross-commodity comparisons.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class UnitNormalizer:
    """Handles conversion of various commodity units to USD per barrel equivalent"""
    
    def __init__(self):
        # Conversion factors for different units to USD/bbl
        self.conversion_factors = {
            # Crude Oil - already in USD/bbl
            'USD/bbl': 1.0,
            
            # Refined products - convert USD/ton to USD/bbl
            'USD/ton': 1/7.45,  # 1 ton ≈ 7.45 barrels
            
            # Gasoline - convert USD/gallon to USD/bbl
            'USD/gal': 0.42,  # 1 barrel = 42 US gallons
            
            # LNG - convert USD/MMBtu to USD/bbl equivalent
            'USD/MMBtu': 0.172,  # 1 MMBtu ≈ 0.172 bbl equivalent
            
            # FX - leave unchanged for now
            'USD': 1.0,
            'EUR': 1.0,
            'SGD': 1.0
        }
        
        # Using BDTI correlation to benchmark Worldscale rates
        self.ws_conversion_params = {
            'BIDY Index': {  # Baltic Dirty Tanker Index (VLCC routes)
                'benchmark_route': 'TD3C',  # Arabian Gulf to China
                'vessel_type': 'VLCC',
                'cargo_size_mt': 270000,  # 270k metric tons
                'flat_rate_usd_mt': 39.50,  # 2025 TD3C flat rate
                'bbl_per_mt': 7.3,  # Standard conversion for medium-gravity crudes
                'bdi_to_ws_correlation': 0.06,  # BDI 1000 ≈ WS 60
                'route_factor': 1.0  # Base route adjustment
            },
            'BITY Index': {  # Baltic Clean Tanker Index (Product tanker routes)
                'benchmark_route': 'TC2',  # Rotterdam to New York
                'vessel_type': 'LR2',
                'cargo_size_mt': 75000,  # 75k metric tons
                'flat_rate_usd_mt': 28.50,  # 2025 TC2 flat rate (estimated)
                'bbl_per_mt': 7.45,  # Refined products conversion
                'bdi_to_ws_correlation': 0.08,  # BDI 1000 ≈ WS 80 (higher for clean)
                'route_factor': 1.0  # Base route adjustment
            }
        }
    
    def convert_worldscale_to_usd_bbl(self, ws_value, ticker):
        """
        Convert Worldscale points to USD/bbl using Trader's Proxy Method
        
        Step 1: Correlate BDTI to benchmark Worldscale rate
        Step 2: Calculate freight cost in $/metric ton
        Step 3: Convert from $/metric ton to $/barrel
        
        Args:
            ws_value (float): Baltic Index value (BDTI/BDCI)
            ticker (str): Ticker symbol (BIDY or BITY)
            
        Returns:
            float: Freight cost in USD/bbl
        """
        if ticker not in self.ws_conversion_params:
            # Fallback for unknown tickers
            return ws_value * 0.0035  # Rough estimate: BDI 1000 ≈ $3.50/bbl
        
        params = self.ws_conversion_params[ticker]
        
        # Step 1: Correlate BDTI to benchmark Worldscale rate
        # BDI level correlates to WS points for benchmark route
        ws_points = ws_value * params['bdi_to_ws_correlation']
        
        # Step 2: Calculate freight cost in $/metric ton
        # Freight Cost ($/mt) = (Worldscale Points / 100) * Flat Rate ($/mt)
        freight_cost_mt = (ws_points / 100) * params['flat_rate_usd_mt']
        
        # Step 3: Convert from $/metric ton to $/barrel
        # Freight Cost ($/bbl) = Freight Cost ($/mt) / Barrels per Ton
        freight_cost_bbl = freight_cost_mt / params['bbl_per_mt']
        
        # Apply route factor if needed
        freight_cost_bbl = freight_cost_bbl * params['route_factor']
        
        return freight_cost_bbl
    
    def normalize_price(self, price, unit, commodity, ticker=None):
        """
        Convert price to USD per barrel equivalent
        
        Args:
            price (float): Original price
            unit (str): Original unit
            commodity (str): Commodity type for special handling
            ticker (str): Ticker symbol for freight indices
            
        Returns:
            float: Price in USD/bbl equivalent
        """
        if pd.isna(price) or pd.isna(unit):
            return np.nan
        
        # Special handling for freight indices (Trader's Proxy Method)
        if unit == 'Index' and commodity == 'Freight' and ticker:
            return self.convert_worldscale_to_usd_bbl(price, ticker)
        
        # Get conversion factor for other units
        factor = self.conversion_factors.get(unit, 1.0)
        
        # Apply conversion
        normalized_price = price * factor
        
        return normalized_price
    
    def process_data(self, input_file, output_file):
        """
        Process the input CSV and add normalized price column
        
        Args:
            input_file (str): Path to input CSV
            output_file (str): Path to output CSV
        """
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Original data shape: {df.shape}")
        print(f"Commodities: {df['Commodity'].unique()}")
        print(f"Units: {df['Unit'].unique()}")
        
        # Add normalized price column with ticker information
        print("Normalizing prices to USD/bbl equivalent...")
        print("Applying Trader's Proxy Method for freight indices...")
        
        df['Price_USD_bbl'] = df.apply(
            lambda row: self.normalize_price(
                row['Price'], 
                row['Unit'], 
                row['Commodity'], 
                row['Ticker']
            ), 
            axis=1
        )
        
        # Add conversion factor column for transparency
        df['Conversion_Factor'] = df['Unit'].map(self.conversion_factors)
        
        # Add conversion method for freight indices
        df['Conversion_Method'] = df.apply(
            lambda row: 'Trader\'s Proxy Method' if row['Unit'] == 'Index' and row['Commodity'] == 'Freight' 
                       else 'Standard Factor', 
            axis=1
        )
        
        # Summary statistics
        print("\nNormalization Summary:")
        for commodity in df['Commodity'].unique():
            subset = df[df['Commodity'] == commodity]
            if len(subset) > 0:
                avg_original = subset['Price'].mean()
                avg_normalized = subset['Price_USD_bbl'].mean()
                print(f"  {commodity}: {avg_original:.2f} → {avg_normalized:.4f} USD/bbl")
                
                # Show freight conversion details
                if commodity == 'Freight':
                    for ticker in subset['Ticker'].unique():
                        ticker_data = subset[subset['Ticker'] == ticker]
                        if len(ticker_data) > 0:
                            ws_value = ticker_data['Price'].iloc[0]
                            usd_bbl = ticker_data['Price_USD_bbl'].iloc[0]
                            print(f"    {ticker}: {ws_value:.0f} BDI → ${usd_bbl:.4f}/bbl")
        
        # Save normalized data
        print(f"\nSaving normalized data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print(f"Normalization complete! Output shape: {df.shape}")
        
        return df
    
    def validate_conversions(self, df):
        """
        Validate the conversions make sense
        
        Args:
            df (DataFrame): Normalized dataframe
        """
        print("\nValidation Summary:")
        
        # Check for any NaN values in normalized prices
        nan_count = df['Price_USD_bbl'].isna().sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values in normalized prices")
        
        # Check price ranges by commodity
        for commodity in df['Commodity'].unique():
            subset = df[df['Commodity'] == commodity]
            if len(subset) > 0:
                min_price = subset['Price_USD_bbl'].min()
                max_price = subset['Price_USD_bbl'].max()
                print(f"  {commodity}: ${min_price:.4f} - ${max_price:.4f} USD/bbl")
                
                # Show freight conversion validation
                if commodity == 'Freight':
                    print("    Freight conversion validation:")
                    for ticker in subset['Ticker'].unique():
                        ticker_data = subset[subset['Ticker'] == ticker]
                        if len(ticker_data) > 0:
                            avg_bdi = ticker_data['Price'].mean()
                            avg_usd_bbl = ticker_data['Price_USD_bbl'].mean()
                            print(f"      {ticker}: {avg_bdi:.0f} BDI avg → ${avg_usd_bbl:.4f}/bbl avg")
                            
                            # Show correlation validation
                            if ticker == 'BIDY Index':
                                print(f"        BDI 1000 benchmark → ~${avg_usd_bbl * (1000/avg_bdi):.2f}/bbl")
                            elif ticker == 'BITY Index':
                                print(f"        BDI 1000 benchmark → ~${avg_usd_bbl * (1000/avg_bdi):.2f}/bbl")

def main():
    """Main function to run unit normalization"""
    # File paths
    input_file = "data/raw/market_data_clean.csv"
    output_file = "data/processed/normalized_data.csv"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize normalizer
    normalizer = UnitNormalizer()
    
    # Process data
    normalized_df = normalizer.process_data(input_file, output_file)
    
    # Validate results
    normalizer.validate_conversions(normalized_df)
    
    print(f"\n Unit normalization complete!")
    print(f" Output saved to: {output_file}")
    
    return normalized_df

if __name__ == "__main__":
    main()
