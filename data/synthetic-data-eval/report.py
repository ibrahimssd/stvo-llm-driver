
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentTracker:
    """Track and aggregate results across multiple experiments"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / 'all_experiments_results.csv'
        self.results = []
        
    def add_experiment(self, experiment_data: Dict):
        """Add a single experiment result"""
        self.results.append(experiment_data)
        
    def save_consolidated_csv(self):
        """Save all results to a single CSV"""
        if not self.results:
            logging.warning("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        logging.info(f"Consolidated results saved to {self.results_file}")
        return df
    
    def load_existing_results(self):
        """Load existing results if file exists"""
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            self.results = df.to_dict('records')
            logging.info(f"Loaded {len(self.results)} existing results")
            return df
        return None



class EvaluationReportGenerator:
    """Generate evaluation reports and visualizations"""
    
    def __init__(self, model_name: str, num_classes: int,
                 paragraph_id_clustering: bool = False,
                 add_paragraph: bool = False,
                 paragraph_id_to_label: Dict[int, str] = None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.paragraph_id_clustering = paragraph_id_clustering
        self.add_paragraph = add_paragraph
        self.paragraph_id_to_label = paragraph_id_to_label or {}
    
    def generate_report(self, output_path: Path, report: Dict):
            """Generate evaluation report with consolidated tracking"""
            logging.info("Generating evaluation report...")
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract configuration
            mode_flag = "paragraph_id_clustering" if self.paragraph_id_clustering else "binary_classification"
            context_flag = "with_paragraph" if self.add_paragraph else "without_paragraph"
            model_name = self.model_name.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize or load experiment tracker
            tracker = ExperimentTracker(output_path)
            tracker.load_existing_results()
            
            # Process each evaluation setting (e.g., zero_shot, fine_tuned, etc.)
            for setting_name, metrics in report.items():
                if not isinstance(metrics, dict):
                    continue
                
                # Create experiment record
                experiment_record = {
                    'timestamp': timestamp,
                    'model_name': model_name,
                    'model_full_path': self.model_name,
                    'mode': mode_flag,
                    'context': context_flag,
                    'setting': setting_name,
                    'num_classes': self.num_classes,
                    
                    # Core metrics
                    'accuracy': metrics.get('eval_accuracy', None),
                    'balanced_accuracy': metrics.get('eval_balanced_accuracy', None),
                    'precision': metrics.get('eval_precision', None),
                    'recall': metrics.get('eval_recall', None),
                    'f1_macro': metrics.get('eval_f1_macro', None),
                    'f1_micro': metrics.get('eval_f1_micro', None),
                    
                    # Additional metrics if available
                    'loss': metrics.get('eval_loss', None),
                    'runtime': metrics.get('eval_runtime', None),
                    'samples_per_second': metrics.get('eval_samples_per_second', None),
                }
                
                # Add to tracker
                tracker.add_experiment(experiment_record)
            
            # Save consolidated CSV
            df = tracker.save_consolidated_csv()
            
            # Also save individual JSON report for detailed inspection
            output_report = {
                'mode': mode_flag,
                'context_flag': context_flag,
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'timestamp': timestamp,
                'paragraph_id_mapping': self.paragraph_id_to_label if self.paragraph_id_clustering else None
            }
            
            for key, value in report.items():
                if isinstance(value, dict):
                    for metric_key, metric_value in value.items():
                        output_report[f"{key}_{metric_key}"] = metric_value
                else:
                    output_report[key] = value
            
            report_path = output_path / f'evaluation_report_{mode_flag}_{model_name}_{context_flag}_{timestamp}.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(output_report, f, indent=2, default=str, ensure_ascii=False)
            
            logging.info(f"Individual JSON report saved to {report_path}")
            
            # Generate visualizations
            if df is not None and len(df) > 0:
                self.generate_comparison_plots(df, output_path)
            
            return df

    def generate_comparison_plots(self, df: pd.DataFrame, output_path: Path):
            """Generate comprehensive comparison plots from consolidated results"""
            
            plot_dir = output_path / 'comparison_plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (14, 8)
            
            # 1. Overall Performance Comparison (All Models, All Settings)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Model Performance Comparison Across All Experiments', fontsize=16, fontweight='bold')
            
            metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro']
            metric_titles = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Macro', 'F1 Micro']
            
            for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[idx // 3, idx % 3]
                
                # Filter valid data
                plot_data = df[df[metric].notna()].copy()
                
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title)
                    continue
                
                # Create comparison plot
                plot_data['model_setting'] = plot_data['model_name'] + '\n' + plot_data['setting']
                
                sns.barplot(
                    data=plot_data,
                    x='model_setting',
                    y=metric,
                    hue='mode',
                    ax=ax,
                    palette='Set2'
                )
                
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Mode', loc='lower right', fontsize=8)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', fontsize=7)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved overall comparison plot")
            
            # 2. Model-by-Model Comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('F1 Score Comparison by Model', fontsize=16, fontweight='bold')
            
            # F1 Macro
            plot_data = df[df['f1_macro'].notna()].copy()
            if len(plot_data) > 0:
                pivot_macro = plot_data.pivot_table(
                    values='f1_macro',
                    index='model_name',
                    columns='setting',
                    aggfunc='mean'
                )
                
                pivot_macro.plot(kind='bar', ax=axes[0])
                axes[0].set_title('F1 Macro Score', fontweight='bold')
                axes[0].set_xlabel('Model')
                axes[0].set_ylabel('F1 Macro')
                axes[0].legend(title='Setting', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].set_ylim(0, 1)
                axes[0].grid(axis='y', alpha=0.3)
            
            # F1 Micro
            plot_data = df[df['f1_micro'].notna()].copy()
            if len(plot_data) > 0:
                pivot_micro = plot_data.pivot_table(
                    values='f1_micro',
                    index='model_name',
                    columns='setting',
                    aggfunc='mean'
                )
                
                pivot_micro.plot(kind='bar', ax=axes[1])
                axes[1].set_title('F1 Micro Score', fontweight='bold')
                axes[1].set_xlabel('Model')
                axes[1].set_ylabel('F1 Micro')
                axes[1].legend(title='Setting', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].set_ylim(0, 1)
                axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved F1 comparison plot")
            
            # 3. Context Impact (With vs Without Paragraph)
            if 'context' in df.columns and len(df['context'].unique()) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                context_comparison = df[df['f1_macro'].notna()].groupby(['model_name', 'context'])['f1_macro'].mean().unstack()
                
                context_comparison.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
                ax.set_title('Impact of Paragraph Context on F1 Macro Score', fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel('F1 Macro Score', fontsize=12)
                ax.legend(title='Context', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'context_impact.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Saved context impact plot")
            
            # 4. Heatmap of Model Performance
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create pivot table for heatmap
            heatmap_data = df.pivot_table(
                values='f1_macro',
                index='model_name',
                columns='setting',
                aggfunc='mean'
            )
            
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0.5,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'F1 Macro Score'}
            )
            
            ax.set_title('Performance Heatmap: F1 Macro by Model and Setting', fontsize=14, fontweight='bold')
            ax.set_xlabel('Evaluation Setting', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved performance heatmap")
            
            # 5. Time Series Plot (if multiple timestamps)
            if len(df['timestamp'].unique()) > 1:
                fig, ax = plt.subplots(figsize=(14, 6))
                
                for model in df['model_name'].unique():
                    model_data = df[df['model_name'] == model].sort_values('timestamp')
                    ax.plot(
                        range(len(model_data)),
                        model_data['f1_macro'],
                        marker='o',
                        label=model,
                        linewidth=2
                    )
                
                ax.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Experiment Number', fontsize=12)
                ax.set_ylabel('F1 Macro Score', fontsize=12)
                ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Saved performance timeline")
            
            # 6. Generate summary statistics table
            summary_stats = df.groupby(['model_name', 'setting'])[['accuracy', 'f1_macro', 'f1_micro']].agg(['mean', 'std']).round(4)
            summary_stats.to_csv(plot_dir / 'summary_statistics.csv')
            
            # 7. Best performing configurations
            best_configs = df.nlargest(10, 'f1_macro')[['model_name', 'mode', 'context', 'setting', 'f1_macro', 'accuracy']]
            best_configs.to_csv(plot_dir / 'top_10_configurations.csv', index=False)
            
            logging.info(f"All comparison plots saved to {plot_dir}")
            
            # Print summary to console
            print("\n" + "="*80)
            print("EXPERIMENT SUMMARY")
            print("="*80)
            print(f"\nTotal experiments: {len(df)}")
            print(f"Models evaluated: {', '.join(df['model_name'].unique())}")
            print(f"Settings tested: {', '.join(df['setting'].unique())}")
            print(f"\nBest F1 Macro: {df['f1_macro'].max():.4f}")
            print(f"Best Configuration:")
            best_row = df.loc[df['f1_macro'].idxmax()]
            print(f"  Model: {best_row['model_name']}")
            print(f"  Setting: {best_row['setting']}")
            print(f"  Mode: {best_row['mode']}")
            print(f"  Context: {best_row['context']}")
            print("="*80 + "\n")


    def plot_consolidated_results(self,csv_path: str, output_dir: str = None):
            """
            Standalone function to plot results from consolidated CSV.
            Can be called separately to regenerate plots.
            
            Usage:
                python plot_results.py --csv all_experiments_results.csv --output plots/
            """
            df = pd.read_csv(csv_path)
            
            if output_dir is None:
                output_dir = Path(csv_path).parent / 'plots'
            
            self.generate_comparison_plots(df, Path(output_dir))
            print(f"✓ Plots generated in {output_dir}")



