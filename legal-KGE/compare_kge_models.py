"""
Script to train and compare multiple KGE models for legal domain
Runs experiments with different models and generates comparison reports
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class KGEComparison:
    """Compare multiple KGE models on the same dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "./kge_comparison", python_file: str = "../legal-KGE/train_legal_kge.py"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.python_file = python_file
    
    def train_model(self, model_type: str, config: dict):
        """Train a single model with given configuration."""
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*80}\n")
        
        # Build command
        cmd = [
            "python", self.python_file,
            "--model_type", model_type,
            "--data_path", self.data_path,
            "--output_dir", str(self.output_dir / model_type),
        ]
        
        # Add config parameters
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Run training
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ {model_type.upper()} training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ {model_type.upper()} training failed:")
            print(e.stderr)
            return False
    
    def load_results(self, model_type: str):
        """Load training results for a model."""
        model_dir = self.output_dir / model_type
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            print(f"Warning: No metadata found for {model_type}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            'model': model_type,
            'train_loss': metadata.get('final_losses', {}).get('train_loss'),
            'val_loss': metadata.get('final_losses', {}).get('val_loss'),
            'num_params': metadata.get('model_config', {}).get('num_entities', 0) * 
                         metadata.get('model_config', {}).get('embedding_dim', 0),
            'embedding_dim': metadata.get('model_config', {}).get('embedding_dim'),
        }
    
    def run_comparison(self, models_config: dict):
        """Run comparison across multiple models."""
        
        for model_type, config in models_config.items():
            success = self.train_model(model_type, config)
            
            if success:
                result = self.load_results(model_type)
                if result:
                    self.results.append(result)
        
        # Generate comparison report
        self.generate_report()
    
    def generate_report(self):
        """Generate comparison report with visualizations."""
        if not self.results:
            print("No results to compare")
            return
        
        df = pd.DataFrame(self.results)
        
        # Save results table
        df.to_csv(self.output_dir / "comparison_results.csv", index=False)
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80 + "\n")
        print(df.to_string(index=False))
        print("\n")
        
        # Generate plots
        self._plot_comparison(df)
        
        # Print recommendations
        self._print_recommendations(df)
    
    def _plot_comparison(self, df: pd.DataFrame):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Validation Loss Comparison
        ax1 = axes[0, 0]
        df_sorted = df.sort_values('val_loss')
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
        ax1.barh(df_sorted['model'], df_sorted['val_loss'], color=colors)
        ax1.set_xlabel('Validation Loss (lower is better)', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Training Loss Comparison
        ax2 = axes[0, 1]
        df_sorted = df.sort_values('train_loss')
        ax2.barh(df_sorted['model'], df_sorted['train_loss'], color='#e74c3c')
        ax2.set_xlabel('Training Loss', fontsize=12)
        ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Overfitting Analysis
        ax3 = axes[1, 0]
        df['overfit_gap'] = df['val_loss'] - df['train_loss']
        df_sorted = df.sort_values('overfit_gap')
        colors = ['#2ecc71' if gap < 0.1 else '#f39c12' if gap < 0.3 else '#e74c3c' 
                 for gap in df_sorted['overfit_gap']]
        ax3.barh(df_sorted['model'], df_sorted['overfit_gap'], color=colors)
        ax3.set_xlabel('Overfitting Gap (Val - Train)', fontsize=12)
        ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax3.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
        ax3.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (<0.3)')
        ax3.legend()
        ax3.invert_yaxis()
        
        # 4. Model Complexity
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['num_params'], df['val_loss'], 
                            s=200, c=range(len(df)), cmap='viridis', alpha=0.6)
        for idx, row in df.iterrows():
            ax4.annotate(row['model'], (row['num_params'], row['val_loss']),
                        fontsize=10, ha='center')
        ax4.set_xlabel('Number of Parameters', fontsize=12)
        ax4.set_ylabel('Validation Loss', fontsize=12)
        ax4.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {self.output_dir / 'model_comparison.png'}")
        plt.close()
    
    def _print_recommendations(self, df: pd.DataFrame):
        """Print model recommendations based on results."""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80 + "\n")
        
        # Best overall
        best_model = df.loc[df['val_loss'].idxmin()]
        print(f"🏆 Best Overall Performance: {best_model['model'].upper()}")
        print(f"   Validation Loss: {best_model['val_loss']:.4f}")
        print()
        
        # Least overfitting
        df['overfit_gap'] = df['val_loss'] - df['train_loss']
        best_generalization = df.loc[df['overfit_gap'].idxmin()]
        print(f"🎯 Best Generalization: {best_generalization['model'].upper()}")
        print(f"   Overfitting Gap: {best_generalization['overfit_gap']:.4f}")
        print()
        
        # Most efficient (best performance per parameter)
        df['efficiency'] = df['val_loss'] * df['num_params']
        most_efficient = df.loc[df['efficiency'].idxmin()]
        print(f"⚡ Most Efficient: {most_efficient['model'].upper()}")
        print(f"   Performance/Complexity Ratio: {most_efficient['efficiency']:.2e}")
        print()
        
        # Final recommendation
        print("📋 Final Recommendation:")
        if best_model['model'] == best_generalization['model']:
            print(f"   → Use {best_model['model'].upper()} - it has both best performance and generalization")
        else:
            print(f"   → Use {best_model['model'].upper()} for best accuracy")
            print(f"   → Use {best_generalization['model'].upper()} if you need better generalization")
        print()


def main():
    """Run model comparison experiment."""
    
    # Configuration for each model
    models_config = {
        'transe': {
            'embedding_dim': 200,
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 0.001,
            'margin': 1.0,
            'l2_reg': 0.01,
            'dropout': 0.1,
            'scheduler': 'plateau',
        },
        'distmult': {
            'embedding_dim': 200,
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 0.001,
            'l2_reg': 0.01,
            'dropout': 0.1,
            'scheduler': 'plateau',
        },
        'complex': {
            'embedding_dim': 200,  # Will be 200 total (real + imag)
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 0.001,
            'l2_reg': 0.001,  # Lower for ComplEx
            'dropout': 0.1,
            'scheduler': 'plateau',
        },
        'rotate': {
            'embedding_dim': 100,  # Will be 200 total (real + imag)
            'epochs': 75,  # More epochs for RotatE
            'batch_size': 256,
            'learning_rate': 0.0001,  # Lower LR
            'margin': 6.0,  # Larger margin
            'l2_reg': 0.0001,
            'dropout': 0.1,
            'scheduler': 'cosine',
        },
    }
    
    
    # Run comparison
    comparison = KGEComparison(
        data_path="../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt",
        output_dir="../legal-KGE/kge_comparison_results",
        python_file="../legal-KGE/train_legal_kge.py",
        
    )
    
    comparison.run_comparison(models_config)


if __name__ == "__main__":
    main()