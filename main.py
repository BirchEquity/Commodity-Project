"""
Main Analytics Pipeline for Commodity Trading

This script orchestrates the entire analytics pipeline:
1. Data processing (if needed)
2. Unit normalization
3. Arbitrage economics analysis
4. Monte Carlo risk analysis
5. Summary reporting

Designed to impress commodity trading recruiters with understanding of physical trading and arbitrage economics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import subprocess
import sys

# Import our modules
from unit_normalization import UnitNormalizer
from arb_economics import ArbitrageCalculator
from monte_carlo import MonteCarloSimulator

class CommodityAnalyticsPipeline:
    """Main pipeline for commodity trading analytics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "data/raw",
            "data/processed", 
            "outputs",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("📁 Directory structure created")
    
    def check_and_process_data(self):
        """Check if data needs to be reprocessed and run if needed"""
        print("\n" + "="*60)
        print("STEP 0: DATA PROCESSING CHECK")
        print("="*60)
        
        # Check if market_data_clean.csv exists
        clean_data_file = "data/raw/market_data_clean.csv"
        metadata_file = "ticker_metadata.csv"
        
        needs_processing = False
        
        # Check if clean data file exists
        if not Path(clean_data_file).exists():
            print("📊 Clean data file not found. Running data processing...")
            needs_processing = True
        else:
            # Check if ticker_metadata.csv is newer than clean data
            if Path(metadata_file).exists():
                metadata_time = Path(metadata_file).stat().st_mtime
                clean_data_time = Path(clean_data_file).stat().st_mtime
                
                if metadata_time > clean_data_time:
                    print("📊 Ticker metadata updated. Reprocessing data...")
                    needs_processing = True
                else:
                    print("✅ Data is up to date")
            else:
                print("📊 Ticker metadata not found. Running data processing...")
                needs_processing = True
        
        if needs_processing:
            try:
                # Import and run the data processing
                from process_market_data import main as process_data
                process_data()
                print("✅ Data processing completed")
                return True
            except Exception as e:
                print(f"❌ Error during data processing: {e}")
                return False
        
        return True
    
    def run_unit_normalization(self):
        """Step 1: Normalize all prices to USD/bbl equivalent"""
        print("\n" + "="*60)
        print("STEP 1: UNIT NORMALIZATION")
        print("="*60)
        
        input_file = "data/raw/market_data_clean.csv"
        output_file = "data/processed/normalized_data.csv"
        
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"❌ Error: Input file {input_file} not found!")
            return False
        
        # Initialize normalizer and process data
        normalizer = UnitNormalizer()
        normalized_df = normalizer.process_data(input_file, output_file)
        
        # Validate results
        normalizer.validate_conversions(normalized_df)
        
        self.results['normalized_data'] = normalized_df
        print("✅ Unit normalization completed successfully")
        
        return True
    
    def run_arbitrage_analysis(self):
        """Step 2: Calculate arbitrage opportunities"""
        print("\n" + "="*60)
        print("STEP 2: ARBITRAGE ECONOMICS ANALYSIS")
        print("="*60)
        
        input_file = "data/processed/normalized_data.csv"
        output_file = "outputs/arb_results.csv"
        
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"❌ Error: Normalized data file {input_file} not found!")
            return False
        
        # Initialize calculator and process data
        calculator = ArbitrageCalculator()
        arb_results = calculator.process_data(input_file, output_file)
        
        self.results['arbitrage_results'] = arb_results
        print("✅ Arbitrage analysis completed successfully")
        
        return True
    
    def run_monte_carlo_simulation(self):
        """Step 3: Run Monte Carlo risk analysis"""
        print("\n" + "="*60)
        print("STEP 3: MONTE CARLO RISK ANALYSIS")
        print("="*60)
        
        input_file = "outputs/arb_results.csv"
        output_file = "outputs/simulation_results.csv"
        
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"❌ Error: Arbitrage results file {input_file} not found!")
            return False
        
        # Initialize simulator and run simulations
        simulator = MonteCarloSimulator(n_simulations=10000, horizon_days=30)
        results_df, summary_df = simulator.run_complete_simulation(input_file, output_file)
        
        self.results['monte_carlo_results'] = results_df
        self.results['monte_carlo_summary'] = summary_df
        print("✅ Monte Carlo simulation completed successfully")
        
        return True
    
    def generate_summary_report(self):
        """Step 4: Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("STEP 4: SUMMARY REPORT GENERATION")
        print("="*60)
        
        if 'arbitrage_results' not in self.results:
            print("❌ Error: Arbitrage results not available!")
            return False
        
        arb_df = self.results['arbitrage_results']
        
        # Generate comprehensive report
        report = self.create_analytics_report(arb_df)
        
        # Save report
        report_file = "reports/analytics_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_file}")
        print("✅ Summary report generated successfully")
        
        return True
    
    def create_analytics_report(self, arb_df):
        """Create a comprehensive analytics report"""
        report = []
        report.append("COMMODITY TRADING ANALYTICS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 20)
        total_observations = len(arb_df)
        valid_observations = len(arb_df.dropna(subset=['Net_Arb']))
        report.append(f"Total observations: {total_observations:,}")
        report.append(f"Valid arbitrage calculations: {valid_observations:,}")
        report.append(f"Data quality: {(valid_observations/total_observations)*100:.1f}%")
        report.append("")
        
        # Arbitrage opportunity summary
        report.append("ARBITRAGE OPPORTUNITY SUMMARY")
        report.append("-" * 30)
        
        for arb_name in arb_df['Arb_Name'].unique():
            subset = arb_df[arb_df['Arb_Name'] == arb_name]
            description = subset['Description'].iloc[0]
            
            # Remove NaN values for analysis
            valid_data = subset.dropna(subset=['Net_Arb'])
            
            if len(valid_data) == 0:
                report.append(f"\n{arb_name}: {description}")
                report.append("  No valid data available")
                continue
            
            # Calculate statistics
            total_days = len(valid_data)
            arb_open_days = len(valid_data[valid_data['Net_Arb'] > 0])
            arb_percentage = (arb_open_days / total_days) * 100
            
            avg_net_arb = valid_data['Net_Arb'].mean()
            max_net_arb = valid_data['Net_Arb'].max()
            min_net_arb = valid_data['Net_Arb'].min()
            
            report.append(f"\n{arb_name}: {description}")
            report.append(f"  Total trading days: {total_days}")
            report.append(f"  Arbitrage open: {arb_open_days} days ({arb_percentage:.1f}%)")
            report.append(f"  Average Net Arb: ${avg_net_arb:.2f}/bbl")
            report.append(f"  Max Net Arb: ${max_net_arb:.2f}/bbl")
            report.append(f"  Min Net Arb: ${min_net_arb:.2f}/bbl")
        
        # Monte Carlo results if available
        if 'monte_carlo_summary' in self.results:
            report.append("\n" + "=" * 50)
            report.append("MONTE CARLO RISK ANALYSIS")
            report.append("=" * 50)
            
            mc_summary = self.results['monte_carlo_summary']
            for _, row in mc_summary.iterrows():
                report.append(f"\n{row['Arb_Name']}:")
                report.append(f"  Current Net Arb: ${row['Current_Net_Arb']:.2f}/bbl")
                report.append(f"  Historical Volatility: {row['Historical_Volatility']:.3f}")
                report.append(f"  Probability Arb Open: {row['Probability_Arb_Open']:.1f}%")
                report.append(f"  Mean Net Arb: ${row['Mean_Net_Arb']:.2f}/bbl")
                report.append(f"  Risk Band (5th-95th): ${row['Percentile_5']:.2f} to ${row['Percentile_95']:.2f}/bbl")
                report.append(f"  Risk Band Width: ${row['Risk_Band']:.2f}/bbl")
        
        # Recent opportunities
        report.append("\n" + "=" * 50)
        report.append("RECENT ARBITRAGE OPPORTUNITIES")
        report.append("=" * 50)
        
        recent_data = arb_df.tail(40)  # Last 40 observations (10 days * 4 arbs)
        recent_opportunities = recent_data[recent_data['Net_Arb'] > 0]
        
        if len(recent_opportunities) > 0:
            report.append(f"Recent opportunities found: {len(recent_opportunities)}")
            for _, row in recent_opportunities.tail(10).iterrows():
                report.append(f"  {row['Date'].strftime('%Y-%m-%d')}: {row['Arb_Name']} = ${row['Net_Arb']:.2f}/bbl")
        else:
            report.append("No recent arbitrage opportunities found")
        
        # Performance metrics
        report.append("\n" + "=" * 50)
        report.append("PERFORMANCE METRICS")
        report.append("=" * 50)
        
        execution_time = time.time() - self.start_time
        report.append(f"Total execution time: {execution_time:.2f} seconds")
        report.append(f"Data processing efficiency: {valid_observations/execution_time:.0f} calculations/second")
        
        return "\n".join(report)
    
    def run_pipeline(self):
        """Run the complete analytics pipeline"""
        print("🚀 Starting Commodity Trading Analytics Pipeline")
        print("=" * 60)
        
        # Step 0: Check and process data if needed
        if not self.check_and_process_data():
            print("❌ Pipeline failed at data processing step")
            return False
        
        # Step 1: Unit normalization
        if not self.run_unit_normalization():
            print("❌ Pipeline failed at unit normalization step")
            return False
        
        # Step 2: Arbitrage analysis
        if not self.run_arbitrage_analysis():
            print("❌ Pipeline failed at arbitrage analysis step")
            return False
        
        # Step 3: Monte Carlo simulation
        if not self.run_monte_carlo_simulation():
            print("❌ Pipeline failed at Monte Carlo simulation step")
            return False
        
        # Step 4: Generate report
        if not self.generate_summary_report():
            print("❌ Pipeline failed at report generation step")
            return False
        
        # Final summary
        execution_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"📊 Arbitrage opportunities analyzed")
        print(f"🎲 Monte Carlo risk analysis completed")
        print(f"📁 Results saved in organized folder structure")
        print(f"📄 Comprehensive report generated")
        print("\n📈 Ready for interactive dashboard!")
        
        return True

def main():
    """Main function to run the analytics pipeline"""
    # Initialize pipeline
    pipeline = CommodityAnalyticsPipeline()
    
    # Run the complete pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\n✅ Analytics pipeline completed successfully!")
        print("📁 Check the following files:")
        print("   - data/processed/normalized_data.csv")
        print("   - outputs/arb_results.csv")
        print("   - outputs/simulation_results.csv")
        print("   - outputs/monte_carlo_distributions.png")
        print("   - outputs/risk_comparison.png")
        print("   - reports/analytics_summary.txt")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")
    
    return success

if __name__ == "__main__":
    main()
