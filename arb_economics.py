"""
Arbitrage Economics Module for Commodity Trading Analytics

This module calculates key arbitrage spreads that physical oil traders actually look at,
including freight and FX adjustments to determine if arbitrage windows are open.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class ArbitrageCalculator:
    """Calculates arbitrage opportunities across different commodity markets"""
    
    def __init__(self):
        # Define arbitrage pairs with their specifications
        # IMPORTANT: Origin = where we buy, Destination = where we sell
        # Arbitrage = Sell Price - Buy Price - Freight - FX - Storage - Financing - Quality - FX_Hedging
        self.arbitrage_pairs = {
            'Brent_WTI': {
                'description': 'Atlantic Basin Crude Arb (Brent - WTI)',
                'origin_ticker': 'CL1 Comdty',  # WTI (buy in US)
                'destination_ticker': 'CO1 Comdty',  # Brent (sell in Europe)
                'freight_index': 'BIDY Index',  # Baltic Dirty for crude
                'fx_rate': None,  # Both in USD
                'commodity_type': 'Crude Oil',
                'trade_direction': 'US to Europe',
                'storage_cost': 0.175,  # $0.10-0.25/bbl average
                'financing_cost': 0.30,  # $0.20-0.40/bbl average
                'quality_cost': 0.35,  # $0.20-0.50/bbl average
                'fx_hedging_cost': 0.0  # Negligible (both USD)
            },
            'Brent_Dubai': {
                'description': 'East vs West Crude Arb (Dubai - Brent)',
                'origin_ticker': 'DBL1 Comdty',  # Dubai (buy in Middle East)
                'destination_ticker': 'CO1 Comdty',  # Brent (sell in Europe)
                'freight_index': 'BIDY Index',  # Baltic Dirty for crude
                'fx_rate': None,  # Both in USD
                'commodity_type': 'Crude Oil',
                'trade_direction': 'Middle East to Europe',
                'storage_cost': 0.225,  # $0.15-0.30/bbl average
                'financing_cost': 0.45,  # $0.30-0.60/bbl average
                'quality_cost': 0.75,  # $0.50-1.00/bbl average
                'fx_hedging_cost': 0.0  # Negligible (both USD)
            },
            'Rotterdam_Singapore_Gasoil': {
                'description': 'Europe vs Asia Distillate Arb (Rotterdam - Singapore Gasoil)',
                'origin_ticker': 'W61 Comdty',  # Singapore Gasoil (buy in Asia)
                'destination_ticker': 'QSU5 Comdty',  # Rotterdam Gasoil (sell in Europe)
                'freight_index': 'BITY Index',  # Baltic Clean for refined products
                'fx_rate': None,  # Both in USD
                'commodity_type': 'Gasoil',
                'trade_direction': 'Asia to Europe',
                'storage_cost': 0.30,  # $0.20-0.40/bbl average
                'financing_cost': 0.60,  # $0.40-0.80/bbl average
                'quality_cost': 0.20,  # $0.10-0.30/bbl average
                'fx_hedging_cost': 0.125  # $0.05-0.20/bbl average
            },
            'Eurobob_USGC_Gasoline': {
                'description': 'Gasoline Arb Across Atlantic (Eurobob - USGC)',
                'origin_ticker': 'XB1 Comdty',  # USGC Gasoline (buy in US)
                'destination_ticker': 'IHW1 Comdty',  # Eurobob Gasoline (sell in Europe)
                'freight_index': 'BITY Index',  # Baltic Clean for refined products
                'fx_rate': 'USDEUR Curncy',  # Convert EUR to USD
                'commodity_type': 'Gasoline',
                'trade_direction': 'US to Europe',
                'storage_cost': 0.30,  # $0.20-0.40/bbl/month average
                'financing_cost': 0.225,  # $0.15-0.30/bbl average
                'quality_cost': 0.50,  # $0.30-0.70/bbl average
                'fx_hedging_cost': 0.10  # $0.05-0.15/bbl average
            }
        }
    
    def load_normalized_data(self, file_path):
        """Load normalized data and prepare for arbitrage calculations"""
        print(f"Loading normalized data from {file_path}...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pivot data to have tickers as columns for easier calculations
        df_pivot = df.pivot(index='Date', columns='Ticker', values='Price_USD_bbl')
        
        print(f"Data loaded: {df_pivot.shape[0]} dates, {df_pivot.shape[1]} tickers")
        return df, df_pivot
    
    def calculate_freight_adjustment(self, df_pivot, freight_index, date):
        """Calculate freight adjustment using properly converted freight costs"""
        if freight_index in df_pivot.columns:
            freight_cost = df_pivot.loc[date, freight_index]
            if pd.notna(freight_cost):
                # The freight cost is already converted to USD/bbl in the normalized data
                # No additional scaling needed - use the converted value directly
                return freight_cost
        return 0.0
    
    def calculate_fx_adjustment(self, df_pivot, fx_rate, origin_price, destination_price, date):
        """Calculate FX adjustment if needed"""
        if fx_rate and fx_rate in df_pivot.columns:
            fx_value = df_pivot.loc[date, fx_rate]
            if pd.notna(fx_value):
                # For EUR/USD, if destination is in EUR, we need to convert EUR price to USD
                # For other pairs, adjust accordingly
                if 'EUR' in fx_rate:
                    # If destination price is in EUR, convert to USD for comparison
                    return destination_price * (fx_value - 1)  # Adjustment for EUR pricing
        return 0.0
    
    def calculate_arbitrage_spread(self, df_pivot, arb_config, date):
        """Calculate arbitrage spread for a specific date and configuration"""
        origin_ticker = arb_config['origin_ticker']
        destination_ticker = arb_config['destination_ticker']
        freight_index = arb_config['freight_index']
        fx_rate = arb_config['fx_rate']
        
        # Get additional costs from config
        storage_cost = arb_config.get('storage_cost', 0.0)
        financing_cost = arb_config.get('financing_cost', 0.0)
        quality_cost = arb_config.get('quality_cost', 0.0)
        fx_hedging_cost = arb_config.get('fx_hedging_cost', 0.0)
        
        # Get prices
        origin_price = df_pivot.loc[date, origin_ticker] if origin_ticker in df_pivot.columns else np.nan
        destination_price = df_pivot.loc[date, destination_ticker] if destination_ticker in df_pivot.columns else np.nan
        
        if pd.isna(origin_price) or pd.isna(destination_price):
            return {
                'Raw_Spread': np.nan,
                'Freight_Adjustment': np.nan,
                'FX_Adjustment': np.nan,
                'Storage_Cost': np.nan,
                'Financing_Cost': np.nan,
                'Quality_Cost': np.nan,
                'FX_Hedging_Cost': np.nan,
                'Net_Arb': np.nan
            }
        
        # CORRECTED ARBITRAGE LOGIC:
        # Arbitrage = Sell Price - Buy Price - Freight - FX - Storage - Financing - Quality - FX_Hedging
        # Raw Spread = Destination (sell) - Origin (buy)
        raw_spread = destination_price - origin_price
        
        # Calculate freight adjustment using converted freight costs
        freight_adj = self.calculate_freight_adjustment(df_pivot, freight_index, date)
        
        # Calculate FX adjustment
        fx_adj = self.calculate_fx_adjustment(df_pivot, fx_rate, origin_price, destination_price, date)
        
        # Calculate net arbitrage: Sell - Buy - Freight - FX - Storage - Financing - Quality - FX_Hedging
        net_arb = raw_spread - freight_adj - fx_adj - storage_cost - financing_cost - quality_cost - fx_hedging_cost
        
        return {
            'Raw_Spread': raw_spread,
            'Freight_Adjustment': freight_adj,
            'FX_Adjustment': fx_adj,
            'Storage_Cost': storage_cost,
            'Financing_Cost': financing_cost,
            'Quality_Cost': quality_cost,
            'FX_Hedging_Cost': fx_hedging_cost,
            'Net_Arb': net_arb
        }
    
    def calculate_all_arbitrages(self, df_pivot):
        """Calculate arbitrage spreads for all pairs and dates"""
        print("Calculating arbitrage spreads...")
        
        results = []
        
        for arb_name, arb_config in self.arbitrage_pairs.items():
            print(f"  Processing {arb_name}: {arb_config['description']}")
            print(f"    Trade Direction: {arb_config['trade_direction']}")
            
            for date in df_pivot.index:
                arb_result = self.calculate_arbitrage_spread(df_pivot, arb_config, date)
                
                results.append({
                    'Date': date,
                    'Arb_Name': arb_name,
                    'Description': arb_config['description'],
                    'Origin_Ticker': arb_config['origin_ticker'],
                    'Destination_Ticker': arb_config['destination_ticker'],
                    'Freight_Index': arb_config['freight_index'],
                    'Commodity_Type': arb_config['commodity_type'],
                    'Trade_Direction': arb_config['trade_direction'],
                    'Raw_Spread': arb_result['Raw_Spread'],
                    'Freight_Adjustment': arb_result['Freight_Adjustment'],
                    'FX_Adjustment': arb_result['FX_Adjustment'],
                    'Storage_Cost': arb_result['Storage_Cost'],
                    'Financing_Cost': arb_result['Financing_Cost'],
                    'Quality_Cost': arb_result['Quality_Cost'],
                    'FX_Hedging_Cost': arb_result['FX_Hedging_Cost'],
                    'Net_Arb': arb_result['Net_Arb']
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by date and arbitrage name
        results_df = results_df.sort_values(['Date', 'Arb_Name']).reset_index(drop=True)
        
        print(f"Arbitrage analysis complete! Results shape: {results_df.shape}")
        
        return results_df
    
    def analyze_arbitrage_opportunities(self, arb_df):
        """Analyze arbitrage opportunities and print summary statistics"""
        print("\n" + "="*60)
        print("ARBITRAGE OPPORTUNITY ANALYSIS")
        print("="*60)
        
        for arb_name in arb_df['Arb_Name'].unique():
            subset = arb_df[arb_df['Arb_Name'] == arb_name]
            description = subset['Description'].iloc[0]
            trade_direction = subset['Trade_Direction'].iloc[0]
            
            # Remove NaN values for analysis
            valid_data = subset.dropna(subset=['Net_Arb'])
            
            if len(valid_data) == 0:
                print(f"\n{arb_name}: {description}")
                print("  No valid data available")
                continue
            
            # Calculate statistics
            total_days = len(valid_data)
            arb_open_days = len(valid_data[valid_data['Net_Arb'] > 0])
            arb_percentage = (arb_open_days / total_days) * 100
            
            avg_net_arb = valid_data['Net_Arb'].mean()
            max_net_arb = valid_data['Net_Arb'].max()
            min_net_arb = valid_data['Net_Arb'].min()
            
            # Calculate average freight and FX adjustments
            avg_freight = valid_data['Freight_Adjustment'].mean()
            avg_fx = valid_data['FX_Adjustment'].mean()
            
            # Get recent example for clarity
            recent_data = valid_data.tail(1).iloc[0]
            
            print(f"\n{arb_name}: {description}")
            print(f"  Trade Direction: {trade_direction}")
            print(f"  Total trading days: {total_days}")
            print(f"  Arbitrage open: {arb_open_days} days ({arb_percentage:.1f}%)")
            print(f"  Average Net Arb: ${avg_net_arb:.2f}/bbl")
            print(f"  Max Net Arb: ${max_net_arb:.2f}/bbl")
            print(f"  Min Net Arb: ${min_net_arb:.2f}/bbl")
            print(f"  Average Freight Cost: ${avg_freight:.4f}/bbl")
            print(f"  Average FX Adjustment: ${avg_fx:.4f}/bbl")
            
            # Show recent example
            print(f"  Recent Example ({recent_data['Date'].strftime('%Y-%m-%d')}):")
            print(f"    Raw Spread: ${recent_data['Raw_Spread']:.2f}/bbl")
            print(f"    Freight Cost: ${recent_data['Freight_Adjustment']:.2f}/bbl")
            print(f"    FX Adjustment: ${recent_data['FX_Adjustment']:.2f}/bbl")
            print(f"    Net Arbitrage: ${recent_data['Net_Arb']:.2f}/bbl")
    
    def process_data(self, input_file, output_file):
        """Process normalized data and calculate arbitrage opportunities"""
        # Load data
        df, df_pivot = self.load_normalized_data(input_file)
        
        # Calculate arbitrage spreads
        arb_results = self.calculate_all_arbitrages(df_pivot)
        
        # Analyze opportunities
        self.analyze_arbitrage_opportunities(arb_results)
        
        # Save results
        print(f"\nSaving arbitrage results to {output_file}...")
        arb_results.to_csv(output_file, index=False)
        
        return arb_results

def main():
    """Main function to run arbitrage analysis"""
    # File paths
    input_file = "data/processed/normalized_data.csv"
    output_file = "outputs/arb_results.csv"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator
    calculator = ArbitrageCalculator()
    
    # Process data
    arb_results = calculator.process_data(input_file, output_file)
    
    print(f"\n Arbitrage analysis complete!")
    print(f" Results saved to: {output_file}")
    
    return arb_results

if __name__ == "__main__":
    main()
