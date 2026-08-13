import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


def extract_real_data_metrics(csv_path: str, 
                               clm: float, 
                               clu: float, 
                               cls: float, 
                               data_name: str = None) -> pd.DataFrame:
    """
    Extract and compare baseline vs pre-trained performance on real data evaluation.
    
    Args:
        csv_path: Path to CSV file with evaluation results
        clm: CLM lambda value (e.g., 0.4)
        clu: CLU lambda value (e.g., 0.1)
        cls: CLS lambda value (e.g., 0.5)
        data_name: Optional filter by dataset name (e.g., 'austrian', 'german', 'irish')
    
    Returns:
        DataFrame with comparison metrics
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter by dataset if specified
    if data_name:
        df = df[df['data_name'] == data_name]
    
    # Separate baseline and pre-trained models
    baseline = df[df['Model_Name'].str.startswith('baseline_')].copy()
    pretrained = df[~df['Model_Name'].str.startswith('baseline_')].copy()
    
    # Extract base model name for matching
    baseline['base_model'] = baseline['Model_Name'].str.replace('baseline_', '')
    
    # Parse config from pretrained model names
    def parse_pretrained_config(name: str) -> Tuple[str, float, float, float]:
        """Extract base model and config from pretrained model name."""
        # Example: bert-base-uncased-eurlex_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5.pt
        parts = name.split('_m2m100_')[0]
        base = parts
        
        # Extract CLM, CLU, CLS values
        try:
            clm_idx = name.find('clm_nsp_') + len('clm_nsp_')
            clm_end = name.find('_clu_')
            clm_val = float(name[clm_idx:clm_end])
            
            clu_idx = name.find('_clu_') + len('_clu_')
            clu_end = name.find('_cls_')
            clu_val = float(name[clu_idx:clu_end])
            
            cls_idx = name.find('_cls_') + len('_cls_')
            cls_end = name.find('.pt')
            if cls_end == -1:
                cls_end = len(name)
            cls_val = float(name[cls_idx:cls_end])
            
            return base, clm_val, clu_val, cls_val
        except:
            return base, None, None, None
    
    pretrained['parsed'] = pretrained['Model_Name'].apply(parse_pretrained_config)
    pretrained['base_model'] = pretrained['parsed'].apply(lambda x: x[0])
    pretrained['clm_value'] = pretrained['parsed'].apply(lambda x: x[1])
    pretrained['clu_value'] = pretrained['parsed'].apply(lambda x: x[2])
    pretrained['cls_value'] = pretrained['parsed'].apply(lambda x: x[3])
    
    # Filter by specified config
    config_match = pretrained[
        (pretrained['clm_value'] == clm) &
        (pretrained['clu_value'] == clu) &
        (pretrained['cls_value'] == cls)
    ]
    
    # Build comparison results
    results = []
    
    for _, base_row in baseline.iterrows():
        base_model_name = base_row['base_model']
        
        # Find matching pretrained model
        match = config_match[config_match['base_model'] == base_model_name]
        
        if len(match) > 0:
            pt_row = match.iloc[0]
            
            results.append({
                'model': base_model_name,
                'dataset': base_row['data_name'],
                'config': f"({clm:.1f},{clu:.1f},{cls:.1f})",
                
                # Baseline metrics
                'baseline_accuracy': base_row['Accuracy'],
                'baseline_balanced_acc': base_row['Balanced_Accuracy'],
                'baseline_precision': base_row['Precision'],
                'baseline_recall': base_row['Recall'],
                'baseline_F1_Macro': base_row['F1_Macro'],
                'baseline_F1_Macro': base_row['F1_Macro'],
                'baseline_mcc': base_row['MCC'],
                
                # Pre-trained metrics
                'pretrained_accuracy': pt_row['Accuracy'],
                'pretrained_balanced_acc': pt_row['Balanced_Accuracy'],
                'pretrained_precision': pt_row['Precision'],
                'pretrained_recall': pt_row['Recall'],
                'pretrained_F1_Macro': pt_row['F1_Macro'],
                'pretrained_F1_Macro': pt_row['F1_Macro'],
                'pretrained_mcc': pt_row['MCC'],
                
                # Gains (absolute)
                'gain_accuracy': pt_row['Accuracy'] - base_row['Accuracy'],
                'gain_balanced_acc': pt_row['Balanced_Accuracy'] - base_row['Balanced_Accuracy'],
                'gain_precision': pt_row['Precision'] - base_row['Precision'],
                'gain_recall': pt_row['Recall'] - base_row['Recall'],
                'gain_F1_Macro': pt_row['F1_Macro'] - base_row['F1_Macro'],
                'gain_F1_Macro': pt_row['F1_Macro'] - base_row['F1_Macro'],
                'gain_mcc': pt_row['MCC'] - base_row['MCC'],
                
                # Gains (percentage) - handle division by zero
                'gain_F1_Macro_pct': ((pt_row['F1_Macro'] - base_row['F1_Macro']) / base_row['F1_Macro'] * 100) 
                                      if base_row['F1_Macro'] > 0 else 0,
                'gain_accuracy_pct': ((pt_row['Accuracy'] - base_row['Accuracy']) / base_row['Accuracy'] * 100)
                                      if base_row['Accuracy'] > 0 else 0,
                
                # Prediction distribution
                'baseline_pred_yes': base_row['number_of_pred_yes'],
                'baseline_pred_no': base_row['number_of_pred_no'],
                'pretrained_pred_yes': pt_row['number_of_pred_yes'],
                'pretrained_pred_no': pt_row['number_of_pred_no'],
            })
    
    return pd.DataFrame(results)


def table_6_real_data_evaluation(csv_path: str, 
                                   clm: float, 
                                   clu: float, 
                                   cls: float,
                                   data_name: str = None) -> str:
    """
    Generate comparison table for real data evaluation with specific config.
    
    Args:
        csv_path: Path to CSV file with evaluation results
        clm: CLM lambda value
        clu: CLU lambda value  
        cls: CLS lambda value
        data_name: Optional dataset name filter
    
    Returns:
        Formatted table string
    """
    # Extract metrics
    df = extract_real_data_metrics(csv_path, clm, clu, cls, data_name)
    
    if len(df) == 0:
        return f"No data found for config ({clm:.1f}, {clu:.1f}, {cls:.1f})"
    
    output = []
    output.append("="*100)
    output.append(f"TABLE 6: REAL DATA EVALUATION - Config ({clm:.1f}, {clu:.1f}, {cls:.1f})")
    output.append("="*100)
    output.append("")
    
    # Dataset info
    dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else "Unknown"
    output.append(f"Dataset: {dataset_name.upper()}")
    output.append(f"Number of models: {len(df)}")
    output.append("")
    
    # Main comparison table - F1 Macro (primary metric for imbalanced data)
    output.append("F1-MACRO COMPARISON (Primary Metric):")
    output.append("-"*100)
    output.append(f"{'Model':<35} {'Baseline':<12} {'Pre-trained':<12} {'Gain':<15} {'Improvement'}")
    output.append("-"*100)
    
    # Sort by pretrained F1-Macro (descending)
    df_sorted = df.sort_values('pretrained_F1_Macro', ascending=False)
    
    for _, row in df_sorted.iterrows():
        model_short = row['model'][:33]
        gain_str = f"+{row['gain_F1_Macro']:.3f}"
        pct_str = f"({row['gain_F1_Macro_pct']:>6.1f}%)"
        
        output.append(
            f"{model_short:<35} "
            f"{row['baseline_F1_Macro']:<12.3f} "
            f"{row['pretrained_F1_Macro']:<12.3f} "
            f"{gain_str:<15} "
            f"{pct_str}"
        )
    
    # Balanced Accuracy comparison
    output.append("")
    output.append("BALANCED ACCURACY COMPARISON:")
    output.append("-"*100)
    output.append(f"{'Model':<35} {'Baseline':<12} {'Pre-trained':<12} {'Gain'}")
    output.append("-"*100)
    
    for _, row in df_sorted.iterrows():
        model_short = row['model'][:33]
        gain_str = f"+{row['gain_balanced_acc']:.3f}"
        
        output.append(
            f"{model_short:<35} "
            f"{row['baseline_balanced_acc']:<12.3f} "
            f"{row['pretrained_balanced_acc']:<12.3f} "
            f"{gain_str}"
        )
    
    # Precision/Recall comparison
    output.append("")
    output.append("PRECISION & RECALL:")
    output.append("-"*100)
    output.append(f"{'Model':<35} {'Metric':<12} {'Baseline':<12} {'Pre-trained':<12} {'Gain'}")
    output.append("-"*100)
    
    for _, row in df_sorted.iterrows():
        model_short = row['model'][:33]
        
        # Precision
        output.append(
            f"{model_short:<35} "
            f"{'Precision':<12} "
            f"{row['baseline_precision']:<12.3f} "
            f"{row['pretrained_precision']:<12.3f} "
            f"+{row['gain_precision']:.3f}"
        )
        
        # Recall
        output.append(
            f"{'':<35} "
            f"{'Recall':<12} "
            f"{row['baseline_recall']:<12.3f} "
            f"{row['pretrained_recall']:<12.3f} "
            f"+{row['gain_recall']:.3f}"
        )
        output.append("")
    
    # Summary statistics
    output.append("")
    output.append("="*100)
    output.append("SUMMARY STATISTICS")
    output.append("="*100)
    output.append("")
    
    avg_baseline_f1 = df['baseline_F1_Macro'].mean()
    avg_pretrained_f1 = df['pretrained_F1_Macro'].mean()
    avg_gain_f1 = df['gain_F1_Macro'].mean()
    avg_gain_pct = df['gain_F1_Macro_pct'].mean()
    
    output.append(f"Average F1-Macro (Baseline):     {avg_baseline_f1:.3f}")
    output.append(f"Average F1-Macro (Pre-trained):  {avg_pretrained_f1:.3f}")
    output.append(f"Average Gain:                    +{avg_gain_f1:.3f} ({avg_gain_pct:>5.1f}%)")
    output.append("")
    
    avg_baseline_acc = df['baseline_accuracy'].mean()
    avg_pretrained_acc = df['pretrained_accuracy'].mean()
    avg_gain_acc = df['gain_accuracy'].mean()
    
    output.append(f"Average Accuracy (Baseline):     {avg_baseline_acc:.3f}")
    output.append(f"Average Accuracy (Pre-trained):  {avg_pretrained_acc:.3f}")
    output.append(f"Average Gain:                    +{avg_gain_acc:.3f}")
    output.append("")
    
    # Best model
    best_idx = df['pretrained_F1_Macro'].idxmax()
    best_model = df.loc[best_idx]
    
    output.append("BEST MODEL:")
    output.append(f"  Model:           {best_model['model']}")
    output.append(f"  F1-Macro:        {best_model['pretrained_F1_Macro']:.3f} (baseline: {best_model['baseline_F1_Macro']:.3f})")
    output.append(f"  Gain:            +{best_model['gain_F1_Macro']:.3f} ({best_model['gain_F1_Macro_pct']:.1f}%)")
    output.append(f"  Balanced Acc:    {best_model['pretrained_balanced_acc']:.3f}")
    output.append(f"  Precision:       {best_model['pretrained_precision']:.3f}")
    output.append(f"  Recall:          {best_model['pretrained_recall']:.3f}")
    output.append("")
    
    # Positive/Negative cases
    positive_gains = df[df['gain_F1_Macro'] > 0]
    output.append(f"Models with positive gain:       {len(positive_gains)}/{len(df)} ({len(positive_gains)/len(df)*100:.1f}%)")
    
    output.append("")
    return "\n".join(output)


def generate_all_configs_comparison(csv_path: str, 
                                      data_name: str = None,
                                      top_n: int = 10) -> str:
    """
    Compare performance across ALL configs and find the best one for each model.
    
    Args:
        csv_path: Path to CSV file
        data_name: Optional dataset name
        top_n: Number of top configs to show
        
    Returns:
        Formatted comparison string
    """
    df = pd.read_csv(csv_path)
    
    if data_name:
        df = df[df['data_name'] == data_name]
    
    # Separate baseline and pretrained
    baseline = df[df['Model_Name'].str.startswith('baseline_')].copy()
    pretrained = df[~df['Model_Name'].str.startswith('baseline_')].copy()
    
    baseline['base_model'] = baseline['Model_Name'].str.replace('baseline_', '')
    
    # Parse all pretrained configs
    def parse_config_all(name: str) -> Tuple[str, float, float, float]:
        parts = name.split('_m2m100_')[0]
        base = parts
        
        try:
            clm_idx = name.find('clm_nsp_') + len('clm_nsp_')
            clm_end = name.find('_clu_')
            clm_val = float(name[clm_idx:clm_end])
            
            clu_idx = name.find('_clu_') + len('_clu_')
            clu_end = name.find('_cls_')
            clu_val = float(name[clu_idx:clu_end])
            
            cls_idx = name.find('_cls_') + len('_cls_')
            cls_end = name.find('.pt')
            if cls_end == -1:
                cls_end = len(name)
            cls_val = float(name[cls_idx:cls_end])
            
            return base, clm_val, clu_val, cls_val
        except:
            return base, None, None, None
    
    pretrained['parsed'] = pretrained['Model_Name'].apply(parse_config_all)
    pretrained['base_model'] = pretrained['parsed'].apply(lambda x: x[0])
    pretrained['clm'] = pretrained['parsed'].apply(lambda x: x[1])
    pretrained['clu'] = pretrained['parsed'].apply(lambda x: x[2])
    pretrained['cls'] = pretrained['parsed'].apply(lambda x: x[3])
    
    output = []
    output.append("="*100)
    output.append("BEST CONFIGS PER MODEL (Real Data Evaluation) on Dataset: " + (data_name.upper() if data_name else "ALL"))
    output.append("="*100)
    output.append("")
    
    # For each base model, find best config
    results = []
    
    for base_name in baseline['base_model'].unique():
        base_row = baseline[baseline['base_model'] == base_name].iloc[0]
        base_f1 = base_row['F1_Macro']
        
        # Get all pretrained versions of this model
        model_variants = pretrained[pretrained['base_model'] == base_name]
        
        if len(model_variants) > 0:
            # Find best config
            best = model_variants.nlargest(1, 'F1_Macro').iloc[0]
            
            results.append({
                'model': base_name,
                'baseline_f1': base_f1,
                'best_f1': best['F1_Macro'],
                'gain': best['F1_Macro'] - base_f1,
                'best_config': f"({best['clm']:.1f},{best['clu']:.1f},{best['cls']:.1f})"
            })
    
    results_df = pd.DataFrame(results).sort_values('best_f1', ascending=False)
    
    output.append(f"{'Model':<35} {'Baseline':<12} {'Best PT':<12} {'Gain':<15} {'Best Config'}")
    output.append("-"*100)
    
    for _, row in results_df.iterrows():
        output.append(
            f"{row['model']:<35} "
            f"{row['baseline_f1']:<12.3f} "
            f"{row['best_f1']:<12.3f} "
            f"+{row['gain']:<14.3f} "
            f"{row['best_config']}"
        )
    
    output.append("")
    output.append(f"Average gain: +{results_df['gain'].mean():.3f}")
    output.append("")
    
    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    # Example 1: Specific config
    # file_path = "./model-performance-clm_nsp_clu_cls-austrian-TMUX_20260220_133049-stvo-20_epoch_10_segments_transe.csv"
    file_path = "./model-performance-clm_nsp_clu_cls-irish-TMUX_20260220_132851-stvo-20_epoch_10_segments_transe.csv"
    result = table_6_real_data_evaluation(
        csv_path=file_path,
        clm=0.4,
        clu=0.1,
        cls=0.5,
        data_name="irish"
    )
    # print(result)
    
    # Example 2: Extract metrics as DataFrame
    df = extract_real_data_metrics(
        csv_path=file_path,
        clm=0.4,
        clu=0.1,
        cls=0.5
    )
    # print(df)
    
    # Example 3: Find best configs
    best_configs = generate_all_configs_comparison(
        csv_path=file_path,
        data_name="irish"
    )
    print(best_configs)