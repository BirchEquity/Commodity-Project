"""
Monte Carlo Simulation Module for Commodity Trading Analytics

This module runs Monte Carlo simulations on arbitrage economics to demonstrate
risk management and volatility analysis. Shows understanding of uncertainty
in physical trading scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    """Monte Carlo simulation engine for arbitrage economics"""
    
    def __init__(self, n_simulations=10000, horizon_days=30):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations (int): Number of Monte Carlo simulations
            horizon_days (int): Time horizon for simulations
        """
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.results = {}
        self.historical_performance = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print(f"🎲 Monte Carlo Simulator initialized")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Horizon: {horizon_days} days")
    
    def load_arbitrage_data(self, file_path):
        """Load arbitrage results from previous analysis"""
        print(f"Loading arbitrage data from {file_path}...")
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get unique arbitrage pairs
        self.arb_pairs = df['Arb_Name'].unique()
        print(f"   Found {len(self.arb_pairs)} arbitrage pairs: {list(self.arb_pairs)}")
        
        return df
    
    def calculate_historical_volatility(self, df, arb_name):
        """Calculate historical volatility for an arbitrage pair"""
        subset = df[df['Arb_Name'] == arb_name].copy()
        
        # Calculate daily returns of raw spread
        subset['Spread_Return'] = subset['Raw_Spread'].pct_change()
        
        # Calculate volatility (std dev of returns)
        volatility = subset['Spread_Return'].std()
        
        # Handle NaN values
        if pd.isna(volatility) or volatility == 0:
            volatility = 0.02  # Default 2% daily volatility
        
        return volatility
    
    def simulate_price_shocks(self, current_spread, volatility):
        """Simulate price shocks using historical volatility"""
        # Generate random returns from normal distribution
        returns = np.random.normal(0, volatility, self.n_simulations)
        
        # Apply shocks to current spread
        shocked_spreads = current_spread * (1 + returns)
        
        return shocked_spreads
    
    def simulate_freight_shocks(self, current_freight):
        """Simulate freight cost shocks (±20% random noise)"""
        # Generate random freight adjustments
        freight_shocks = np.random.uniform(-0.20, 0.20, self.n_simulations)
        
        # Apply shocks to current freight
        shocked_freight = current_freight * (1 + freight_shocks)
        
        return shocked_freight
    
    def simulate_fx_shocks(self, current_fx_rate):
        """Simulate FX rate shocks (±5% random noise)"""
        # Generate random FX adjustments
        fx_shocks = np.random.uniform(-0.05, 0.05, self.n_simulations)
        
        # Apply shocks to current FX rate
        shocked_fx = current_fx_rate * (1 + fx_shocks)
        
        return shocked_fx
    
    def run_single_arbitrage_simulation(self, df, arb_name, simulation_date):
        """Run Monte Carlo simulation for a single arbitrage pair"""
        print(f"   Simulating {arb_name}...")
        
        # Get current values for this arbitrage pair
        current_data = df[(df['Arb_Name'] == arb_name) & 
                         (df['Date'] == simulation_date)]
        
        if len(current_data) == 0:
            print(f"     No data found for {arb_name} on {simulation_date}")
            return None
        
        current_data = current_data.iloc[0]
        
        current_spread = current_data['Raw_Spread']
        current_freight = current_data['Freight_Adjustment']
        current_fx = current_data['FX_Adjustment']
        
        # Check for NaN values
        if pd.isna(current_spread) or pd.isna(current_freight) or pd.isna(current_fx):
            print(f"     NaN values found for {arb_name}, skipping...")
            return None
        
        # Calculate historical volatility
        volatility = self.calculate_historical_volatility(df, arb_name)
        
        # Run simulations
        simulated_spreads = self.simulate_price_shocks(current_spread, volatility)
        simulated_freight = self.simulate_freight_shocks(current_freight)
        simulated_fx = self.simulate_fx_shocks(current_fx)
        
        # Calculate net arbitrage for each simulation
        net_arb_simulations = simulated_spreads - simulated_freight - simulated_fx
        
        # Calculate statistics
        prob_arb_open = np.mean(net_arb_simulations > 0) * 100
        mean_net_arb = np.mean(net_arb_simulations)
        std_net_arb = np.std(net_arb_simulations)
        percentile_5 = np.percentile(net_arb_simulations, 5)
        percentile_95 = np.percentile(net_arb_simulations, 95)
        
        # Store results
        simulation_results = {
            'Arb_Name': arb_name,
            'Simulation_Date': simulation_date,
            'Current_Net_Arb': current_data['Net_Arb'],
            'Historical_Volatility': volatility,
            'Probability_Arb_Open': prob_arb_open,
            'Mean_Net_Arb': mean_net_arb,
            'Std_Net_Arb': std_net_arb,
            'Percentile_5': percentile_5,
            'Percentile_95': percentile_95,
            'Risk_Band': percentile_95 - percentile_5,
            'Simulated_Spreads': simulated_spreads,
            'Simulated_Freight': simulated_freight,
            'Simulated_FX': simulated_fx,
            'Net_Arb_Simulations': net_arb_simulations
        }
        
        return simulation_results
    
    def run_all_simulations(self, df, simulation_date=None):
        """Run Monte Carlo simulations for all arbitrage pairs"""
        print(f"\n Starting Monte Carlo simulations...")
        
        # Use most recent date if not specified
        if simulation_date is None:
            simulation_date = df['Date'].max()
        
        print(f"   Simulation date: {simulation_date}")
        
        all_results = []
        
        for arb_name in self.arb_pairs:
            try:
                result = self.run_single_arbitrage_simulation(df, arb_name, simulation_date)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"   Error simulating {arb_name}: {e}")
                continue
        
        # If no results, try with an earlier date
        if len(all_results) == 0:
            print("   No results found for most recent date, trying earlier date...")
            earlier_date = df['Date'].max() - pd.Timedelta(days=7)
            for arb_name in self.arb_pairs:
                try:
                    result = self.run_single_arbitrage_simulation(df, arb_name, earlier_date)
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    print(f"   Error simulating {arb_name}: {e}")
                    continue
        
        self.results = all_results
        print(f" Completed {len(all_results)} arbitrage simulations")
        
        return all_results
    
    def generate_summary_statistics(self):
        """Generate summary statistics for all simulations"""
        print(f"\n Generating summary statistics...")
        
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Arb_Name': result['Arb_Name'],
                'Current_Net_Arb': result['Current_Net_Arb'],
                'Historical_Volatility': result['Historical_Volatility'],
                'Probability_Arb_Open': result['Probability_Arb_Open'],
                'Mean_Net_Arb': result['Mean_Net_Arb'],
                'Std_Net_Arb': result['Std_Net_Arb'],
                'Percentile_5': result['Percentile_5'],
                'Percentile_95': result['Percentile_95'],
                'Risk_Band': result['Risk_Band']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Print summary
        print("\n" + "="*80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['Arb_Name']}:")
            print(f"  Current Net Arb: ${row['Current_Net_Arb']:.2f}/bbl")
            print(f"  Historical Volatility: {row['Historical_Volatility']:.3f}")
            print(f"  Probability Arb Open: {row['Probability_Arb_Open']:.1f}%")
            print(f"  Mean Net Arb: ${row['Mean_Net_Arb']:.2f}/bbl")
            print(f"  Std Dev Net Arb: ${row['Std_Net_Arb']:.2f}/bbl")
            print(f"  Risk Band (5th-95th): ${row['Percentile_5']:.2f} to ${row['Percentile_95']:.2f}/bbl")
            print(f"  Risk Band Width: ${row['Risk_Band']:.2f}/bbl")
        
        return summary_df
    
    def create_visualizations(self, output_dir="outputs"):
        """Create visualization plots for simulation results"""
        print(f"\n Creating visualizations...")
        
        # Check if we have results to visualize
        if len(self.results) == 0:
            print("   No simulation results to visualize")
            return
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots for each arbitrage pair
        n_arbs = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, result in enumerate(self.results):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            arb_name = result['Arb_Name']
            net_arb_sims = result['Net_Arb_Simulations']
            
            # Check for valid data
            if len(net_arb_sims) == 0 or np.all(np.isnan(net_arb_sims)):
                ax.text(0.5, 0.5, f'No valid data for {arb_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(arb_name)
                continue
            
            # Create histogram
            ax.hist(net_arb_sims, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(result['Current_Net_Arb'], color='red', linestyle='--', 
                      label=f"Current: ${result['Current_Net_Arb']:.2f}")
            ax.axvline(result['Mean_Net_Arb'], color='green', linestyle='--', 
                      label=f"Mean: ${result['Mean_Net_Arb']:.2f}")
            ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
            
            ax.set_title(f'{arb_name}\nProbability Arb Open: {result["Probability_Arb_Open"]:.1f}%')
            ax.set_xlabel('Net Arbitrage ($/bbl)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monte_carlo_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        arb_names = [r['Arb_Name'] for r in self.results]
        risk_bands = [r['Risk_Band'] for r in self.results]
        prob_open = [r['Probability_Arb_Open'] for r in self.results]
        
        bars = ax.bar(arb_names, risk_bands, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add probability labels on bars
        for bar, prob in zip(bars, prob_open):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Risk Analysis: Arbitrage Opportunity Comparison')
        ax.set_ylabel('Risk Band Width ($/bbl)')
        ax.set_xlabel('Arbitrage Pair')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Plots saved to {output_dir}/")
    
    def save_results(self, output_file):
        """Save simulation results to CSV"""
        print(f"\n Saving results to {output_file}...")
        
        # Prepare data for CSV export
        export_data = []
        
        for result in self.results:
            # Add summary statistics
            export_data.append({
                'Arb_Name': result['Arb_Name'],
                'Simulation_Date': result['Simulation_Date'],
                'Current_Net_Arb': result['Current_Net_Arb'],
                'Historical_Volatility': result['Historical_Volatility'],
                'Probability_Arb_Open': result['Probability_Arb_Open'],
                'Mean_Net_Arb': result['Mean_Net_Arb'],
                'Std_Net_Arb': result['Std_Net_Arb'],
                'P5_Net_Arb': result['Percentile_5'],
                'P95_Net_Arb': result['Percentile_95'],
                'Risk_Band_Width': result['Risk_Band'],
                'N_Simulations': self.n_simulations,
                'Horizon_Days': self.horizon_days
            })
        
        # Create DataFrame and save
        results_df = pd.DataFrame(export_data)
        results_df.to_csv(output_file, index=False)
        
        print(f"    Results saved with {len(results_df)} arbitrage pairs")
        
        return results_df
    
    def run_historical_performance_analysis(self, df, lookback_days=90):
        """
        Analyze historical performance of Monte Carlo predictions
        
        Args:
            df: Arbitrage data DataFrame
            lookback_days: Number of days to look back for analysis
        """
        print(f"\n📊 Running Historical Performance Analysis...")
        print(f"   Lookback period: {lookback_days} days")
        
        # Get recent dates for analysis
        recent_dates = df['Date'].nlargest(lookback_days).sort_values()
        
        performance_data = []
        
        for i, current_date in enumerate(recent_dates[:-1]):  # Exclude last date (no future data)
            next_date = recent_dates.iloc[i + 1]
            
            print(f"   Analyzing {current_date.strftime('%Y-%m-%d')} -> {next_date.strftime('%Y-%m-%d')}")
            
            # Run simulation for current date
            current_results = self.run_all_simulations(df, current_date)
            
            if not current_results:
                continue
            
            # Get actual outcomes for next date
            for result in current_results:
                arb_name = result['Arb_Name']
                
                # Get actual data for next date
                actual_data = df[(df['Arb_Name'] == arb_name) & 
                               (df['Date'] == next_date)]
                
                if len(actual_data) == 0:
                    continue
                
                actual_data = actual_data.iloc[0]
                actual_net_arb = actual_data['Net_Arb']
                
                # Calculate prediction accuracy metrics
                predicted_prob_open = result['Probability_Arb_Open']
                predicted_mean = result['Mean_Net_Arb']
                predicted_5th = result['Percentile_5']
                predicted_95th = result['Percentile_95']
                
                # Determine if prediction was correct
                actual_was_open = actual_net_arb > 0
                predicted_open = predicted_prob_open > 50
                prediction_correct = actual_was_open == predicted_open
                
                # Check if actual value fell within predicted range
                within_range = (actual_net_arb >= predicted_5th) and (actual_net_arb <= predicted_95th)
                
                # Calculate prediction error
                prediction_error = abs(actual_net_arb - predicted_mean)
                
                performance_data.append({
                    'Date': current_date,
                    'Next_Date': next_date,
                    'Arb_Name': arb_name,
                    'Predicted_Prob_Open': predicted_prob_open,
                    'Predicted_Mean': predicted_mean,
                    'Predicted_5th': predicted_5th,
                    'Predicted_95th': predicted_95th,
                    'Actual_Net_Arb': actual_net_arb,
                    'Actual_Was_Open': actual_was_open,
                    'Predicted_Open': predicted_open,
                    'Prediction_Correct': prediction_correct,
                    'Within_Range': within_range,
                    'Prediction_Error': prediction_error,
                    'Days_Ahead': (next_date - current_date).days
                })
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(performance_data)
        
        if len(performance_df) == 0:
            print("   No historical performance data available")
            return None
        
        # Calculate summary statistics
        self.calculate_performance_metrics(performance_df)
        
        # Store for later use
        self.historical_performance = performance_df
        
        return performance_df
    
    def calculate_performance_metrics(self, performance_df):
        """Calculate and display performance metrics"""
        print(f"\n📈 Historical Performance Metrics:")
        print("=" * 60)
        
        # Overall accuracy
        total_predictions = len(performance_df)
        correct_predictions = performance_df['Prediction_Correct'].sum()
        accuracy = (correct_predictions / total_predictions) * 100
        
        # Range accuracy
        within_range_count = performance_df['Within_Range'].sum()
        range_accuracy = (within_range_count / total_predictions) * 100
        
        # Average prediction error
        mean_error = performance_df['Prediction_Error'].mean()
        
        # By arbitrage pair
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"Range Accuracy: {range_accuracy:.1f}% ({within_range_count}/{total_predictions})")
        print(f"Mean Prediction Error: ${mean_error:.2f}/bbl")
        
        # Performance by arbitrage pair
        print(f"\nPerformance by Arbitrage Pair:")
        for arb_name in performance_df['Arb_Name'].unique():
            arb_data = performance_df[performance_df['Arb_Name'] == arb_name]
            arb_accuracy = (arb_data['Prediction_Correct'].sum() / len(arb_data)) * 100
            arb_range_accuracy = (arb_data['Within_Range'].sum() / len(arb_data)) * 100
            arb_mean_error = arb_data['Prediction_Error'].mean()
            
            print(f"  {arb_name}:")
            print(f"    Accuracy: {arb_accuracy:.1f}%")
            print(f"    Range Accuracy: {arb_range_accuracy:.1f}%")
            print(f"    Mean Error: ${arb_mean_error:.2f}/bbl")
        
        # Performance by prediction confidence
        print(f"\nPerformance by Prediction Confidence:")
        confidence_bins = [0, 25, 50, 75, 100]
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            mask = (performance_df['Predicted_Prob_Open'] >= low) & (performance_df['Predicted_Prob_Open'] < high)
            bin_data = performance_df[mask]
            
            if len(bin_data) > 0:
                bin_accuracy = (bin_data['Prediction_Correct'].sum() / len(bin_data)) * 100
                print(f"  {low}-{high}% confidence: {bin_accuracy:.1f}% accuracy ({len(bin_data)} predictions)")
    
    def create_performance_visualizations(self, output_dir="outputs"):
        """Create visualizations for historical performance analysis"""
        if self.historical_performance is None or len(self.historical_performance) == 0:
            print("   No historical performance data to visualize")
            return
        
        print(f"\n📊 Creating performance visualizations...")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create performance dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction Accuracy Over Time
        ax1 = axes[0, 0]
        performance_df = self.historical_performance
        
        # Group by date and calculate daily accuracy
        daily_accuracy = performance_df.groupby('Date')['Prediction_Correct'].agg(['mean', 'count'])
        daily_accuracy['accuracy_pct'] = daily_accuracy['mean'] * 100
        
        ax1.plot(daily_accuracy.index, daily_accuracy['accuracy_pct'], marker='o', linewidth=2)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Guess')
        ax1.set_title('Prediction Accuracy Over Time')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xlabel('Date')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction Error Distribution
        ax2 = axes[0, 1]
        errors = performance_df['Prediction_Error']
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'Mean Error: ${errors.mean():.2f}')
        ax2.set_title('Prediction Error Distribution')
        ax2.set_xlabel('Absolute Error ($/bbl)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted Values
        ax3 = axes[1, 0]
        actual = performance_df['Actual_Net_Arb']
        predicted = performance_df['Predicted_Mean']
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
        
        # Scatter plot
        scatter = ax3.scatter(predicted, actual, alpha=0.6, 
                            c=performance_df['Prediction_Error'], cmap='viridis')
        ax3.set_title('Actual vs Predicted Net Arbitrage')
        ax3.set_xlabel('Predicted ($/bbl)')
        ax3.set_ylabel('Actual ($/bbl)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Prediction Error ($/bbl)')
        
        # 4. Performance by Arbitrage Pair
        ax4 = axes[1, 1]
        arb_performance = performance_df.groupby('Arb_Name').agg({
            'Prediction_Correct': 'mean',
            'Within_Range': 'mean',
            'Prediction_Error': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(arb_performance))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, arb_performance['Prediction_Correct'] * 100, 
                       width, label='Prediction Accuracy', alpha=0.7)
        bars2 = ax4.bar(x_pos + width/2, arb_performance['Within_Range'] * 100, 
                       width, label='Range Accuracy', alpha=0.7)
        
        ax4.set_title('Performance by Arbitrage Pair')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_xlabel('Arbitrage Pair')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(arb_performance['Arb_Name'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/historical_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Performance visualizations saved to {output_dir}/historical_performance.png")
    
    def save_performance_results(self, output_file):
        """Save historical performance results to CSV"""
        if self.historical_performance is None:
            print("   No historical performance data to save")
            return None
        
        print(f"\n💾 Saving performance results to {output_file}...")
        
        # Save detailed performance data
        self.historical_performance.to_csv(output_file, index=False)
        
        # Create summary statistics
        summary_stats = self.historical_performance.groupby('Arb_Name').agg({
            'Prediction_Correct': ['mean', 'count'],
            'Within_Range': 'mean',
            'Prediction_Error': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats.reset_index(inplace=True)
        
        # Save summary
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_stats.to_csv(summary_file, index=False)
        
        print(f"   Detailed results saved to: {output_file}")
        print(f"   Summary statistics saved to: {summary_file}")
        
        return summary_stats

    def run_complete_simulation(self, input_file, output_file, simulation_date=None, include_historical=True):
        """Run complete Monte Carlo simulation pipeline"""
        print(" Starting Monte Carlo Simulation Pipeline")
        print("=" * 60)
        
        # Load data
        df = self.load_arbitrage_data(input_file)
        
        # Run current simulations
        self.run_all_simulations(df, simulation_date)
        
        # Generate summary
        summary_df = self.generate_summary_statistics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        results_df = self.save_results(output_file)
        
        # Run historical performance analysis if requested
        if include_historical:
            print(f"\n" + "=" * 60)
            print("HISTORICAL PERFORMANCE ANALYSIS")
            print("=" * 60)
            
            performance_df = self.run_historical_performance_analysis(df)
            
            if performance_df is not None:
                # Create performance visualizations
                self.create_performance_visualizations()
                
                # Save performance results
                performance_file = output_file.replace('.csv', '_performance.csv')
                self.save_performance_results(performance_file)
        
        print(f"\n Monte Carlo simulation complete!")
        print(f" Results saved to: {output_file}")
        print(f" Visualizations saved to: outputs/")
        
        if include_historical and performance_df is not None:
            print(f" Historical performance saved to: {performance_file}")
        
        return results_df, summary_df

def main():
    """Main function to run Monte Carlo simulations"""
    # File paths
    input_file = "outputs/arb_results.csv"
    output_file = "outputs/simulation_results.csv"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = MonteCarloSimulator(n_simulations=10000, horizon_days=30)
    
    # Run complete simulation with historical performance analysis
    results_df, summary_df = simulator.run_complete_simulation(
        input_file, output_file, include_historical=True
    )
    
    return results_df, summary_df

if __name__ == "__main__":
    main()
