"""
Sensitivity Analysis for WaterToC Mesa Model
Implements OAT, Sobol, and KS sensitivity analysis methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import latin
import warnings
from typing import Dict, List, Tuple, Callable
import multiprocessing as mp
from functools import partial
import itertools

from src.model import WaterToC

warnings.filterwarnings('ignore')

class WaterToCeSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for WaterToC model
    """
    
    def __init__(self, 
                 n_runs_per_config: int = 10,
                 max_steps: int = 100,
                 n_jobs: int = None):
        """
        Initialize sensitivity analyzer
        
        Args:
            n_runs_per_config: Number of runs per parameter configuration
            max_steps: Maximum steps for each model run
            n_jobs: Number of parallel jobs (None for all cores)
        """
        self.n_runs_per_config = n_runs_per_config
        self.max_steps = max_steps
        self.n_jobs = n_jobs or mp.cpu_count()
        
        # Define parameter space
        self.param_bounds = {
            'initial_humans': [5, 20],
            'initial_ai': [5, 20],
            'C_Payoff': [0.1, 1.0],
            'D_Payoff': [0.5, 2.0],
            'max_water_capacity': [5, 15],
            'water_cell_density': [0.2, 0.5],
            'theta': [1.0, 10.0]
        }
        
        # Define baseline parameters
        self.baseline_params = {
            'initial_humans': 10,
            'initial_ai': 10,
            'C_Payoff': 0.5,
            'D_Payoff': 1.0,
            'max_water_capacity': 10,
            'water_cell_density': 0.3,
            'theta': 3.0
        }
        
        # Key output metrics to analyze
        self.output_metrics = [
            'final_coop_fraction',
            'final_environment_state', 
            'final_total_water',
            'avg_coop_fraction',
            'avg_environment_state',
            'water_depletion_rate',
            'stability_index'
        ]

    def run_single_simulation(self, params: Dict, seed: int = None) -> Dict:
        """
        Run a single model simulation and extract key metrics
        
        Args:
            params: Dictionary of model parameters
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of output metrics
        """
        try:
            # Create model with given parameters
            model = WaterToC(
                height=20,  # Fixed grid size
                width=20,   # Fixed grid size
                initial_humans=int(params['initial_humans']),
                initial_ai=int(params['initial_ai']),
                C_Payoff=params['C_Payoff'],
                D_Payoff=params['D_Payoff'],
                max_water_capacity=params['max_water_capacity'],
                water_cell_density=params['water_cell_density'],
                theta=params['theta'],
                seed=seed
            )
            
            # Run simulation
            for _ in range(self.max_steps):
                model.step()
            
            # Extract data
            data = model.datacollector.get_model_vars_dataframe()
            
            # Calculate output metrics
            outputs = {
                'final_coop_fraction': data['Coop_Fraction'].iloc[-1],
                'final_environment_state': data['Environment_State'].iloc[-1],
                'final_total_water': data['Total_Water'].iloc[-1],
                'avg_coop_fraction': data['Coop_Fraction'].mean(),
                'avg_environment_state': data['Environment_State'].mean(),
                'water_depletion_rate': (data['Total_Water'].iloc[0] - data['Total_Water'].iloc[-1]) / self.max_steps,
                'stability_index': 1.0 / (1.0 + data['Coop_Fraction'].std())  # Higher = more stable
            }
            
            return outputs
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return {metric: np.nan for metric in self.output_metrics}

    def run_batch_simulations(self, param_configs: List[Dict]) -> pd.DataFrame:
        """
        Run multiple simulations with different parameter configurations
        
        Args:
            param_configs: List of parameter dictionaries
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for i, params in enumerate(param_configs):
            print(f"Running configuration {i+1}/{len(param_configs)}")
            
            # Run multiple replications for each configuration
            for run in range(self.n_runs_per_config):
                seed = i * self.n_runs_per_config + run
                outputs = self.run_single_simulation(params, seed=seed)
                
                # Combine parameters and outputs
                result = {**params, **outputs, 'config_id': i, 'run_id': run}
                results.append(result)
        
        return pd.DataFrame(results)

    def oat_sensitivity_analysis(self, n_samples: int = 10) -> Dict:
        """
        One-at-a-Time sensitivity analysis
        
        Args:
            n_samples: Number of samples for each parameter
            
        Returns:
            Dictionary with OAT results
        """
        print("Running OAT Sensitivity Analysis...")
        
        oat_results = {}
        
        for param_name in self.param_bounds.keys():
            print(f"Analyzing parameter: {param_name}")
            
            # Create parameter configurations
            param_configs = []
            param_values = np.linspace(
                self.param_bounds[param_name][0],
                self.param_bounds[param_name][1],
                n_samples
            )
            
            for value in param_values:
                config = self.baseline_params.copy()
                config[param_name] = value
                param_configs.append(config)
            
            # Run simulations
            results_df = self.run_batch_simulations(param_configs)
            
            # Calculate sensitivity metrics
            param_effects = {}
            for metric in self.output_metrics:
                if metric in results_df.columns:
                    # Group by parameter value and calculate mean
                    grouped = results_df.groupby(param_name)[metric].mean()
                    
                    # Calculate sensitivity measures
                    param_effects[metric] = {
                        'range': grouped.max() - grouped.min(),
                        'std': grouped.std(),
                        'correlation': results_df[param_name].corr(results_df[metric]),
                        'values': grouped.to_dict()
                    }
            
            oat_results[param_name] = {
                'effects': param_effects,
                'data': results_df
            }
        
        return oat_results

    def sobol_sensitivity_analysis(self, n_samples: int = 1000) -> Dict:
        """
        Sobol global sensitivity analysis
        
        Args:
            n_samples: Number of samples for Sobol analysis
            
        Returns:
            Dictionary with Sobol indices
        """
        print("Running Sobol Sensitivity Analysis...")
        
        # Define problem for SALib
        problem = {
            'num_vars': len(self.param_bounds),
            'names': list(self.param_bounds.keys()),
            'bounds': list(self.param_bounds.values())
        }
        
        # Generate samples
        param_values = sobol.sample(problem, n_samples)
        
        # Create parameter configurations
        param_configs = []
        for values in param_values:
            config = dict(zip(problem['names'], values))
            param_configs.append(config)
        
        # Run simulations
        results_df = self.run_batch_simulations(param_configs)
        
        # Calculate Sobol indices for each output metric
        sobol_results = {}
        
        for metric in self.output_metrics:
            if metric in results_df.columns and not results_df[metric].isna().all():
                y = results_df[metric].values
                
                try:
                    Si = sobol_analyze.analyze(problem, y, print_to_console=False)
                    sobol_results[metric] = {
                        'S1': Si['S1'],  # First-order indices
                        'ST': Si['ST'],  # Total-order indices
                        'S2': Si['S2'] if 'S2' in Si else None,  # Second-order indices
                        'S1_conf': Si['S1_conf'],  # Confidence intervals
                        'ST_conf': Si['ST_conf']
                    }
                except Exception as e:
                    print(f"Error calculating Sobol indices for {metric}: {e}")
                    sobol_results[metric] = None
        
        return {
            'sobol_indices': sobol_results,
            'problem': problem,
            'data': results_df
        }

    def ks_sensitivity_analysis(self, n_samples: int = 100, n_perturbations: int = 50) -> Dict:
        """
        Kolmogorov-Smirnov sensitivity analysis
        
        Args:
            n_samples: Number of baseline samples
            n_perturbations: Number of perturbations per parameter
            
        Returns:
            Dictionary with KS test results
        """
        print("Running KS Sensitivity Analysis...")
        
        # Generate baseline sample
        baseline_configs = []
        for _ in range(n_samples):
            config = {}
            for param, bounds in self.param_bounds.items():
                config[param] = np.random.uniform(bounds[0], bounds[1])
            baseline_configs.append(config)
        
        baseline_results = self.run_batch_simulations(baseline_configs)
        
        ks_results = {}
        
        for param_name in self.param_bounds.keys():
            print(f"KS analysis for parameter: {param_name}")
            
            # Generate perturbed sample (vary only this parameter)
            perturbed_configs = []
            for _ in range(n_perturbations):
                config = self.baseline_params.copy()
                config[param_name] = np.random.uniform(
                    self.param_bounds[param_name][0],
                    self.param_bounds[param_name][1]
                )
                perturbed_configs.append(config)
            
            perturbed_results = self.run_batch_simulations(perturbed_configs)
            
            # Perform KS tests for each output metric
            param_ks_results = {}
            for metric in self.output_metrics:
                if metric in baseline_results.columns and metric in perturbed_results.columns:
                    baseline_values = baseline_results[metric].dropna()
                    perturbed_values = perturbed_results[metric].dropna()
                    
                    if len(baseline_values) > 0 and len(perturbed_values) > 0:
                        ks_stat, p_value = stats.ks_2samp(baseline_values, perturbed_values)
                        param_ks_results[metric] = {
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            
            ks_results[param_name] = {
                'ks_tests': param_ks_results,
                'baseline_data': baseline_results,
                'perturbed_data': perturbed_results
            }
        
        return ks_results

    def plot_oat_results(self, oat_results: Dict, save_path: str = None):
        """
        Plot OAT sensitivity analysis results
        """
        n_params = len(oat_results)
        n_metrics = len(self.output_metrics)
        
        fig, axes = plt.subplots(n_metrics, n_params, figsize=(4*n_params, 3*n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(self.output_metrics):
            for j, (param_name, results) in enumerate(oat_results.items()):
                ax = axes[i, j]
                
                if metric in results['effects']:
                    data = results['data']
                    # Plot mean and confidence intervals
                    grouped = data.groupby(param_name)[metric].agg(['mean', 'std'])
                    
                    ax.plot(grouped.index, grouped['mean'], 'o-', linewidth=2)
                    ax.fill_between(
                        grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.3
                    )
                    
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sobol_results(self, sobol_results: Dict, save_path: str = None):
        """
        Plot Sobol sensitivity analysis results
        """
        sobol_indices = sobol_results['sobol_indices']
        problem = sobol_results['problem']
        
        # Create heatmap of first-order and total-order indices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # First-order indices
        s1_data = []
        st_data = []
        metrics_with_data = []
        
        for metric in self.output_metrics:
            if metric in sobol_indices and sobol_indices[metric] is not None:
                s1_data.append(sobol_indices[metric]['S1'])
                st_data.append(sobol_indices[metric]['ST'])
                metrics_with_data.append(metric)
        
        if s1_data:
            s1_df = pd.DataFrame(s1_data, 
                               columns=problem['names'], 
                               index=metrics_with_data)
            st_df = pd.DataFrame(st_data, 
                               columns=problem['names'], 
                               index=metrics_with_data)
            
            sns.heatmap(s1_df, annot=True, cmap='YlOrRd', ax=ax1, 
                       cbar_kws={'label': 'First-order Index'})
            ax1.set_title('First-order Sobol Indices')
            
            sns.heatmap(st_df, annot=True, cmap='YlOrRd', ax=ax2,
                       cbar_kws={'label': 'Total-order Index'})
            ax2.set_title('Total-order Sobol Indices')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ks_results(self, ks_results: Dict, save_path: str = None):
        """
        Plot KS sensitivity analysis results
        """
        # Create heatmap of KS statistics
        ks_matrix = []
        param_names = []
        
        for param_name, results in ks_results.items():
            param_names.append(param_name)
            ks_row = []
            for metric in self.output_metrics:
                if metric in results['ks_tests']:
                    ks_row.append(results['ks_tests'][metric]['ks_statistic'])
                else:
                    ks_row.append(0)
            ks_matrix.append(ks_row)
        
        ks_df = pd.DataFrame(ks_matrix, 
                           columns=self.output_metrics, 
                           index=param_names)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(ks_df, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': 'KS Statistic'})
        plt.title('Kolmogorov-Smirnov Sensitivity Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_sensitivity_report(self, oat_results: Dict, sobol_results: Dict, 
                                  ks_results: Dict) -> pd.DataFrame:
        """
        Generate comprehensive sensitivity report
        """
        report_data = []
        
        for param_name in self.param_bounds.keys():
            row = {'Parameter': param_name}
            
            # OAT metrics (average across all output metrics)
            if param_name in oat_results:
                ranges = []
                correlations = []
                for metric in self.output_metrics:
                    if metric in oat_results[param_name]['effects']:
                        ranges.append(abs(oat_results[param_name]['effects'][metric]['range']))
                        correlations.append(abs(oat_results[param_name]['effects'][metric]['correlation']))
                
                row['OAT_Avg_Range'] = np.mean(ranges) if ranges else 0
                row['OAT_Avg_Correlation'] = np.mean(correlations) if correlations else 0
            
            # Sobol metrics (average across all output metrics)
            sobol_indices = sobol_results['sobol_indices']
            s1_values = []
            st_values = []
            
            for metric in self.output_metrics:
                if (metric in sobol_indices and 
                    sobol_indices[metric] is not None):
                    param_idx = sobol_results['problem']['names'].index(param_name)
                    s1_values.append(sobol_indices[metric]['S1'][param_idx])
                    st_values.append(sobol_indices[metric]['ST'][param_idx])
            
            row['Sobol_Avg_S1'] = np.mean(s1_values) if s1_values else 0
            row['Sobol_Avg_ST'] = np.mean(st_values) if st_values else 0
            
            # KS metrics (average across all output metrics)
            if param_name in ks_results:
                ks_stats = []
                for metric in self.output_metrics:
                    if metric in ks_results[param_name]['ks_tests']:
                        ks_stats.append(ks_results[param_name]['ks_tests'][metric]['ks_statistic'])
                
                row['KS_Avg_Statistic'] = np.mean(ks_stats) if ks_stats else 0
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        # Calculate overall sensitivity ranking
        report_df['Overall_Sensitivity'] = (
            report_df['OAT_Avg_Range'] + 
            report_df['Sobol_Avg_ST'] + 
            report_df['KS_Avg_Statistic']
        ) / 3
        
        return report_df.sort_values('Overall_Sensitivity', ascending=False)

# Example usage functions
def run_full_sensitivity_analysis():
    """
    Run complete sensitivity analysis suite
    """
    analyzer = WaterToCeSensitivityAnalyzer(
        n_runs_per_config=5,
        max_steps=50,  # Reduced for faster execution
        n_jobs=4
    )
    
    # Run all three types of analysis
    oat_results = analyzer.oat_sensitivity_analysis(n_samples=8)
    sobol_results = analyzer.sobol_sensitivity_analysis(n_samples=500)
    ks_results = analyzer.ks_sensitivity_analysis(n_samples=50, n_perturbations=30)
    
    # Generate plots
    analyzer.plot_oat_results(oat_results, 'oat_sensitivity.png')
    analyzer.plot_sobol_results(sobol_results, 'sobol_sensitivity.png')
    analyzer.plot_ks_results(ks_results, 'ks_sensitivity.png')
    
    # Generate report
    report = analyzer.generate_sensitivity_report(oat_results, sobol_results, ks_results)
    
    return {
        'oat_results': oat_results,
        'sobol_results': sobol_results,
        'ks_results': ks_results,
        'report': report,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    # Run example analysis
    results = run_full_sensitivity_analysis()
    print("\nSensitivity Analysis Report:")
    print(results['report'])