import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def normalize_model_name(name: str) -> str:
    """
    Normalize model name for matching.
    Removes prefixes and converts underscores to hyphens.
    """
    # Remove common prefixes
    name = name.replace('nlpaueb_', '').replace('google-bert_', '').replace('casehold_', '').replace('dlicari_', '').replace('avichr_', '')
    # Convert underscores to hyphens for consistency
    name = name.replace('_', '-')
    return name.lower()


def parse_config(model_name: str) -> Tuple[str, float, float, float]:
    """
    Extract base model and lambda values from model name.
    
    Returns:
        (base_model, clm, clu, cls) or (base_model, None, None, None) if not pretrained
    """
    if '_clm_nsp_' not in model_name:
        # Clean baseline name for matching
        return normalize_model_name(model_name), None, None, None
    
    base = model_name.split('_m2m100')[0]
    # Normalize base name to match baseline naming
    base = normalize_model_name(base)
    
    config_part = model_name.split('_clm_nsp_')[1]
    parts = config_part.split('_')
    
    try:
        clm = float(parts[0])
        clu = float(parts[2])
        cls = float(parts[4])
        return base, clm, clu, cls
    except (IndexError, ValueError):
        return base, None, None, None


def load_and_process_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and separate baseline vs pretrained models."""
    df = pd.read_csv(csv_path)
    
    # Separate baseline and pretrained
    baseline = df[~df['model_name'].str.contains('_clm_nsp_')].copy()
    pretrained = df[df['model_name'].str.contains('_clm_nsp_')].copy()
    
    # Normalize baseline names for matching
    baseline['clean_name'] = baseline['model_name'].apply(normalize_model_name)
    
    # Parse configurations for pretrained models
    parsed = pretrained['model_name'].apply(parse_config)
    pretrained['base_model'] = parsed.apply(lambda x: x[0])
    pretrained['clm'] = parsed.apply(lambda x: x[1])
    pretrained['clu'] = parsed.apply(lambda x: x[2])
    pretrained['cls'] = parsed.apply(lambda x: x[3])
    
    # print(f"Loaded {len(baseline)} baseline models and {len(pretrained)} pre-trained models from {csv_path}")
    return baseline, pretrained


def table_1_baseline_performance(baseline: pd.DataFrame) -> str:
    """
    TABLE 1: Baseline Model Performance
    """
    output = []
    output.append("="*80)
    output.append("TABLE 1: BASELINE MODEL PERFORMANCE (Zero-shot)")
    output.append("="*80)
    output.append("")
    
    # take only samples with baslines _ pretrained when setting == zero_shot_whole_dataset
    baseline = baseline[baseline['setting'] == 'zero_shot_whole_dataset']
    print(f"Found {len(baseline)} baseline models in zero-shot setting.") 
    
    # Sort by F1-macro descending
    baseline_sorted = baseline.sort_values('f1_macro', ascending=False)
    
    # Header
    output.append(f"{'Model':<40} {'F1-Macro':<10} {'Bal.Acc':<10}")
    output.append("-"*60)
    
    # Rows
    for _, row in baseline_sorted.iterrows():
        model_short = row['model_name'].replace('_', ' ')[:38]
        status = "✓ Working" if row['precision'] > 0 else "✗ Degenerate"
        output.append(
            f"{model_short:<40} "
            f"{row['f1_macro']:<10.3f} "
            f"{row['balanced_accuracy']:<10.3f} "
            # f"{status}"
        )
    
    output.append("")
    return "\n".join(output)


def table_2_best_pretrained(baseline: pd.DataFrame, pretrained: pd.DataFrame) -> str:
    """
    TABLE 2: Best Pre-trained Models with Gains
    """
    output = []
    output.append("="*80)
    output.append("TABLE 2: BEST PRE-TRAINED MODELS (Ranked by F1-Macro)")
    output.append("="*80)
    output.append("")
    
    # take only samples with baslines _ pretrained when setting == zero_shot_whole_dataset
    baseline = baseline[baseline['setting'] == 'zero_shot_whole_dataset']
    pretrained = pretrained[pretrained['setting'] == 'zero_shot_whole_dataset']
    print(f"Comparing {len(baseline)} baseline models with {len(pretrained)} pre-trained models in zero-shot setting.")
    # Get best pretrained model for each base
    results = []
    for _, base_row in baseline.iterrows():
        base_name = base_row['model_name']
        clean_name = base_row['clean_name']
        base_f1 = base_row['f1_macro']
        
        related = pretrained[pretrained['base_model'] == clean_name]
        if len(related) == 0:
            continue
        
        best = related.nlargest(1, 'f1_macro').iloc[0]
        gain = best['f1_macro'] - base_f1
        
        results.append({
            'model': base_name,
            'baseline_f1': base_f1,
            'best_f1': best['f1_macro'],
            'gain': gain,
            'clm': best['clm'],
            'clu': best['clu'],
            'cls': best['cls'],
            'precision': best['precision'],
            'recall': best['recall'],
            'balanced_accuracy': best['balanced_accuracy']
        })
    
    if not results:
        output.append("No pre-trained models found for comparison.")
        output.append("")
        return "\n".join(output)
    
    results_df = pd.DataFrame(results).sort_values('best_f1', ascending=False)
    
    # Header
    output.append(f"{'Model':<35} {'Baseline':<10} {'Best':<10} {'Gain':<12} {'Config'}")
    output.append("-"*90)
    
    # Rows
    for _, row in results_df.iterrows():
        model_short = row['model'].replace('_', ' ')[:33]
        config_str = f"({row['clm']:.1f},{row['clu']:.1f},{row['cls']:.1f})"
        output.append(
            f"{model_short:<35} "
            f"{row['baseline_f1']:<10.3f} "
            f"{row['best_f1']:<10.3f} "
            f"+{row['gain']:.3f} ({row['gain']/row['baseline_f1']*100:>5.1f}%) "
            f"{config_str}"
        )
    
    output.append("")
    return "\n".join(output)


def table_3_detailed_metrics(baseline: pd.DataFrame, pretrained: pd.DataFrame) -> str:
    """
    TABLE 3: Detailed Metrics for Top Models
    """
    output = []
    output.append("="*80)
    output.append("TABLE 3: DETAILED METRICS (Top 4 Pre-trained Models)")
    output.append("="*80)
    output.append("")
    
    # take only samples with baslines _ pretrained when setting == zero_shot_whole_dataset
    baseline = baseline[baseline['setting'] == 'zero_shot_whole_dataset']
    pretrained = pretrained[pretrained['setting'] == 'zero_shot_whole_dataset']
    # Get best model for each base
    results = []
    for _, base_row in baseline.iterrows():
        base_name = base_row['model_name']
        clean_name = base_row['clean_name']
        
        related = pretrained[pretrained['base_model'] == clean_name]
        if len(related) == 0:
            continue
        
        best = related.nlargest(1, 'f1_macro').iloc[0]
        results.append({
            'model': base_name,
            'config': f"{best['clm']:.1f}/{best['clu']:.1f}/{best['cls']:.1f}",
            'f1_micro': best['f1_micro'],
            'balanced_accuracy': best['balanced_accuracy'],
            'precision': best['precision'],
            'recall': best['recall']
        })
    
    if not results:
        output.append("No pre-trained models found for comparison.")
        output.append("")
        return "\n".join(output)
    
    results_df = pd.DataFrame(results).sort_values('f1_micro', ascending=False)
    
    # Unicode box drawing for professional table
    output.append("┌" + "─"*33 + "┬" + "─"*12 + "┬" + "─"*9 + "┬" + "─"*10 + "┬" + "─"*10 + "┬" + "─"*9 + "┐")
    output.append(
        "│ " + f"{'Model':<31}" + " │ " + 
        f"{'Config':<10}" + " │ " +
        f"{'F1-Micro':<7}" + " │ " +
        f"{'Bal. Acc.':<8}" + " │ " +
        f"{'Precision':<8}" + " │ " +
        f"{'Recall':<7}" + " │"
    )
    output.append("├" + "─"*33 + "┼" + "─"*12 + "┼" + "─"*9 + "┼" + "─"*10 + "┼" + "─"*10 + "┼" + "─"*9 + "┤")
    
    for _, row in results_df.iterrows():
        model_short = row['model'].replace('_', ' ')[:30]
        output.append(
            "│ " + f"{model_short:<31}" + " │ " +
            f"{row['config']:<10}" + " │ " +
            f"{row['f1_micro']:<7.3f}" + " │ " +
            f"{row['balanced_accuracy']:<8.3f}" + " │ " +
            f"{row['precision']:<8.3f}" + " │ " +
            f"{row['recall']:<7.3f}" + " │"
        )
    
    output.append("└" + "─"*33 + "┴" + "─"*12 + "┴" + "─"*9 + "┴" + "─"*10 + "┴" + "─"*10 + "┴" + "─"*9 + "┘")
    output.append("")
    return "\n".join(output)


def table_4_optimal_configs(pretrained: pd.DataFrame) -> str:
    """
    TABLE 4: Top Lambda Configurations (Averaged Across Models)
    """
    output = []
    output.append("="*80)
    output.append("TABLE 4: OPTIMAL LAMBDA CONFIGURATIONS (Universal)")
    output.append("="*80)
    output.append("")
    
    # Group by configuration and calculate average F1 gain
    config_performance = []
    
    for (clm, clu, cls), group in pretrained.groupby(['clm', 'clu', 'cls']):
        if len(group) >= 2:  # Need at least 2 models to average
            config_performance.append({
                'clm': clm,
                'clu': clu,
                'cls': cls,
                'avg_f1': group['f1_macro'].mean(),
                'n_models': len(group)
            })
    
    if not config_performance:
        output.append("Insufficient data for configuration analysis.")
        output.append("")
        return "\n".join(output)
    
    config_df = pd.DataFrame(config_performance).sort_values('avg_f1', ascending=False).head(10)
    
    # Header
    output.append(f"{'Rank':<6} {'Config (CLM/CLU/CLS)':<25} {'Avg F1':<10} {'#Models':<10} {'Rating'}")
    output.append("-"*80)
    
    # Rows
    for idx, (_, row) in enumerate(config_df.iterrows(), 1):
        config_str = f"{row['clm']:.1f} / {row['clu']:.1f} / {row['cls']:.1f}"
        
        # Visual rating bars
        rating = int(row['avg_f1'] * 15)
        bar = "█" * rating
        
        output.append(
            f"{idx:<6} "
            f"{config_str:<25} "
            f"{row['avg_f1']:<10.3f} "
            f"{int(row['n_models']):<10} "
            f"{bar}"
        )
    
    output.append("")
    
    # Highlight optimal
    best = config_df.iloc[0]
    output.append("🎯 RECOMMENDED CONFIG:")
    output.append(f"   CLM = {best['clm']:.1f}  |  CLU = {best['clu']:.1f}  |  CLS = {best['cls']:.1f}")
    output.append(f"   Average F1 = {best['avg_f1']:.3f} across {int(best['n_models'])} models")
    output.append("")
    
    return "\n".join(output)


def table_5_clu_contribution(baseline: pd.DataFrame, pretrained: pd.DataFrame) -> str:
    """
    TABLE 5: CLU Task Contribution Analysis
    """
    output = []
    output.append("="*80)
    output.append("TABLE 5: CLU TASK EFFECTIVENESS")
    output.append("="*80)
    output.append("")
    
    output.append(f"{'Base Model':<35} {'No CLU':<12} {'With CLU':<12} {'Contribution'}")
    output.append("-"*80)
    
    for _, base_row in baseline.iterrows():
        base_name = base_row['model_name']
        clean_name = base_row['clean_name']
        
        base_pretrained = pretrained[pretrained['base_model'] == clean_name]
        
        no_clu = base_pretrained[base_pretrained['clu'] == 0.0]
        with_clu = base_pretrained[base_pretrained['clu'] > 0.0]
        
        if len(no_clu) > 0 and len(with_clu) > 0:
            model_short = base_name.replace('_', ' ')[:33]
            contribution = with_clu['f1_macro'].mean() - no_clu['f1_macro'].mean()
            
            status = "✓" if contribution > 0.02 else "~" if contribution > 0 else "✗"
            
            output.append(
                f"{model_short:<35} "
                f"{no_clu['f1_macro'].mean():<12.3f} "
                f"{with_clu['f1_macro'].mean():<12.3f} "
                f"{status} {contribution:+.4f}"
            )
    
    output.append("")
    output.append("Legend: ✓ Beneficial (>0.02)  ~ Marginal (0-0.02)  ✗ Harmful (<0)")
    output.append("")
    return "\n".join(output)


def table_7_finetuning_comparison(baseline: pd.DataFrame, pretrained: pd.DataFrame) -> str:
    """
    TABLE 7: Performance Progression Analysis
    """

    def get_base_group(name):
        name = name.lower()
        # Handle the Google-BERT family first due to naming overlap
        if 'google-bert' in name or name.startswith('google_bert-'): 
            return 'GOOGLE-BERT'
        
        if 'eurlex' in name: return 'EURLEX'
        if 'contracts' in name: return 'CONTRACTS'
        if 'echr' in name: return 'ECHR'
        if 'custom-legalbert' in name or 'casehold' in name: return 'CUSTOM-LEGALBERT'
        if 'italian' in name: return 'ITALIAN'
        if 'legal-bert-base' in name: return 'LEGAL-BERT-BASE'
        if 'legal-bert-small' in name: return 'LEGAL-BERT-SMALL'
        if 'hebert' in name: return 'LEGAL-HEBERT'
        
        return 'OTHER'

    output = []
    output.append("="*105)
    output.append(f"{'TABLE 7: PERFORMANCE PROGRESSION ANALYSIS':^105}")
    output.append("="*105)
    header = f"{'Base Model Group':<18} | {'Baseline ZS':^12} | {'Pre-trained ZS':^16} | {'Fine-tuned':^12} | {'PT Gain':^12} | {'FT Gain'}"
    output.append(header)
    output.append("-" * 105)

    base_groups = ['EURLEX', 'CONTRACTS', 'ECHR', 'CUSTOM-LEGALBERT', 'ITALIAN', 'GOOGLE-BERT', 'LEGAL-BERT-BASE', 'LEGAL-BERT-SMALL', 'LEGAL-HEBERT']
    
    comparison_data = []
    # Combine data for easier access
    all_data = pd.concat([baseline, pretrained])

    for group in base_groups:
        # Filter data for this specific group
        group_df = all_data[all_data['model_name'].apply(get_base_group) == group].copy()
        if group_df.empty:
            continue

        # 1. BASELINE (Original HF model)
        # We look for rows that DO NOT have 'm2m100' or '_clm_' in the name
        mask_base = (group_df['setting'] == 'zero_shot_whole_dataset') & \
                    (~group_df['model_name'].str.contains('m2m100|_clm_'))
        val_base = group_df[mask_base]['f1_macro'].max()

        # 2. PRE-TRAINED ZERO-SHOT (Our M2M variants)
        mask_pt_zs = (group_df['setting'] == 'zero_shot_whole_dataset') & \
                     (group_df['model_name'].str.contains('m2m100|_clm_'))
        val_pt_zs = group_df[mask_pt_zs]['f1_macro'].max()

        # 3. FINE-TUNED (Final tuned versions)
        mask_ft = (group_df['setting'] == 'fine_tuned_test_set') & \
                  (group_df['model_name'].str.contains('m2m100|_clm_'))
        val_ft = group_df[mask_ft]['f1_macro'].max()

        # Only add if we have enough data points to show progression
        if pd.notnull(val_base) and pd.notnull(val_pt_zs):
            # If FT is missing, we use PT_ZS as a placeholder or skip
            v_ft = val_ft if pd.notnull(val_ft) else val_pt_zs
            
            pt_gain = val_pt_zs - val_base
            ft_gain = v_ft - val_pt_zs
            
            comparison_data.append({
                'group': group,
                'base': val_base,
                'pt_zs': val_pt_zs,
                'ft': v_ft,
                'pt_gain': pt_gain,
                'ft_gain': ft_gain
            })

    # Sort by Fine-tuned performance
    comparison_data.sort(key=lambda x: x['ft'], reverse=True)

    for row in comparison_data:
        output.append(
            f"{row['group']:<18} | "
            f"{row['base']:^12.3f} | "
            f"{row['pt_zs']:^16.3f} | "
            f"{row['ft']:^12.3f} | "
            f"{row['pt_gain']:+12.3f} | "
            f"{row['ft_gain']:+9.3f}"
        )

    output.append("-" * 105)
    
    if comparison_data:
        avg_pt = sum(d['pt_gain'] for d in comparison_data) / len(comparison_data)
        avg_ft = sum(d['ft_gain'] for d in comparison_data) / len(comparison_data)

        output.append("\nKEY INSIGHTS:")
        output.append(f"1. PRE-TRAINING IMPACT: Average boost of {avg_pt:+.4f} F1.")
        output.append(f"2. FINE-TUNING IMPACT: Additional average gain of {avg_ft:+.4f} F1.")
    
    output.append("="*105)
    return "\n".join(output)


def table_8_ablation_study(pretrained: pd.DataFrame) -> str:
    """
    TABLE 8: Ablation Study on Task Contributions
    
    For each base model, analyze the impact of each task by comparing:
    - Individual tasks: (CLM only), (CLU only), (CLS only)
    - Pairs of tasks: (CLM+CLU), (CLM+CLS), (CLU+CLS)
    - All three tasks: (CLM+CLU+CLS)
    
    Shows which tasks contribute most to performance.
    """
    output = []
    output.append("="*80)
    output.append("TABLE 8: ABLATION STUDY - TASK CONTRIBUTIONS")
    output.append("="*80)
    output.append("")
    
    if len(pretrained) == 0:
        output.append("No pre-trained models found for ablation study.")
        output.append("")
        return "\n".join(output)
    
    # Define task combinations
    # (clm, clu, cls) tuples where value > 0 means task is present
    task_combinations = {
        'CLM only': lambda row: row['clm'] > 0 and row['clu'] == 0 and row['cls'] == 0,
        'CLU only': lambda row: row['clm'] == 0 and row['clu'] > 0 and row['cls'] == 0,
        'CLS only': lambda row: row['clm'] == 0 and row['clu'] == 0 and row['cls'] > 0,
        'CLM+CLU': lambda row: row['clm'] > 0 and row['clu'] > 0 and row['cls'] == 0,
        'CLM+CLS': lambda row: row['clm'] > 0 and row['clu'] == 0 and row['cls'] > 0,
        'CLU+CLS': lambda row: row['clm'] == 0 and row['clu'] > 0 and row['cls'] > 0,
        'ALL (CLM+CLU+CLS)': lambda row: row['clm'] > 0 and row['clu'] > 0 and row['cls'] > 0
    }
    
    # Process each base model
    # select only zero-shot whole dataset samples for fair comparison
    pretrained = pretrained[pretrained['setting'] == 'zero_shot_whole_dataset']
    base_models = pretrained['base_model'].unique()
    
    
    for base_model in sorted(base_models):
        model_data = pretrained[pretrained['base_model'] == base_model]
        
        if len(model_data) == 0:
            continue
        
        output.append(f"{'='*80}")
        output.append(f"BASE MODEL: {base_model}")
        output.append(f"{'='*80}")
        output.append("")
        
        # Find best config for each task combination
        results = {}
        for combo_name, combo_filter in task_combinations.items():
            matching = model_data[model_data.apply(combo_filter, axis=1)]
            
            if len(matching) > 0:
                best = matching.nlargest(1, 'f1_macro').iloc[0]
                results[combo_name] = {
                    'f1_macro': best['f1_macro'],
                    'clm': best['clm'],
                    'clu': best['clu'],
                    'cls': best['cls'],
                    'precision': best['precision'],
                    'recall': best['recall']
                }
            else:
                results[combo_name] = None
        
        # Display results
        output.append(f"{'Configuration':<20} {'F1-Macro':<12} {'Config (CLM/CLU/CLS)':<25} {'Precision':<12} {'Recall'}")
        output.append("-"*100)
        
        # Sort by F1 score descending
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if v is not None],
            key=lambda x: x[1]['f1_macro'],
            reverse=True
        )
        
        for combo_name, metrics in sorted_results:
            config_str = f"({metrics['clm']:.1f}, {metrics['clu']:.1f}, {metrics['cls']:.1f})"
            output.append(
                f"{combo_name:<20} "
                f"{metrics['f1_macro']:<12.3f} "
                f"{config_str:<25} "
                f"{metrics['precision']:<12.3f} "
                f"{metrics['recall']:.3f}"
            )
        
        # Analysis: Compare with full model
        if 'ALL (CLM+CLU+CLS)' in results and results['ALL (CLM+CLU+CLS)'] is not None:
            output.append("")
            output.append("TASK CONTRIBUTION ANALYSIS:")
            output.append("-"*80)
            
            full_f1 = results['ALL (CLM+CLU+CLS)']['f1_macro']
            
            # Calculate drops when removing each task
            drops = []
            
            # Remove CLM (keep CLU+CLS)
            if 'CLU+CLS' in results and results['CLU+CLS'] is not None:
                drop_clm = full_f1 - results['CLU+CLS']['f1_macro']
                drops.append(('Removing CLM', drop_clm, 'CLU+CLS', results['CLU+CLS']['f1_macro']))
            
            # Remove CLU (keep CLM+CLS)
            if 'CLM+CLS' in results and results['CLM+CLS'] is not None:
                drop_clu = full_f1 - results['CLM+CLS']['f1_macro']
                drops.append(('Removing CLU', drop_clu, 'CLM+CLS', results['CLM+CLS']['f1_macro']))
            
            # Remove CLS (keep CLM+CLU)
            if 'CLM+CLU' in results and results['CLM+CLU'] is not None:
                drop_cls = full_f1 - results['CLM+CLU']['f1_macro']
                drops.append(('Removing CLS', drop_cls, 'CLM+CLU', results['CLM+CLU']['f1_macro']))
            
            # Sort by impact (largest drop = most important task)
            drops.sort(key=lambda x: x[1], reverse=True)
            
            output.append(f"Full model (ALL tasks):     F1 = {full_f1:.3f}")
            output.append("")
            
            if drops:
                output.append("Impact of removing each task (ranked by importance):")
                for rank, (desc, drop, config, f1) in enumerate(drops, 1):
                    sign = '-' if drop > 0 else '+'
                    impact = "🔴 CRITICAL" if abs(drop) > 0.05 else "🟡 MODERATE" if abs(drop) > 0.02 else "🟢 MINOR"
                    output.append(
                        f"{rank}. {desc:<20} → {config:<12} "
                        f"F1={f1:.3f} ({sign}{abs(drop):.3f})  {impact}"
                    )
                
                # Identify most important task
                output.append("")
                most_important = drops[0][0].replace('Removing ', '')
                output.append(f"→ MOST CRITICAL TASK: {most_important} (largest drop: {drops[0][1]:.3f})")
            
            # Individual task performance
            output.append("")
            output.append("Individual task performance (standalone):")
            individual = []
            for task_name in ['CLM only', 'CLU only', 'CLS only']:
                if task_name in results and results[task_name] is not None:
                    f1 = results[task_name]['f1_macro']
                    gain_vs_full = f1 - full_f1
                    individual.append((task_name, f1, gain_vs_full))
            
            individual.sort(key=lambda x: x[1], reverse=True)
            for task_name, f1, gain in individual:
                sign = '+' if gain >= 0 else ''
                output.append(f"  {task_name:<12} F1={f1:.3f} ({sign}{gain:.3f} vs full model)")
        
        output.append("")
    
    # Cross-model summary
    output.append("")
    output.append("="*80)
    output.append("CROSS-MODEL SUMMARY")
    output.append("="*80)
    output.append("")
    
    # Aggregate insights across all models
    all_task_importance = {'CLM': [], 'CLU': [], 'CLS': []}
    
    for base_model in base_models:
        model_data = pretrained[pretrained['base_model'] == base_model]
        
        # Find full model
        full_model = model_data[(model_data['clm'] > 0) & (model_data['clu'] > 0) & (model_data['cls'] > 0)]
        if len(full_model) == 0:
            continue
        
        full_f1 = full_model['f1_macro'].max()
        
        # CLM importance (drop when removed)
        without_clm = model_data[(model_data['clm'] == 0) & (model_data['clu'] > 0) & (model_data['cls'] > 0)]
        if len(without_clm) > 0:
            drop = full_f1 - without_clm['f1_macro'].max()
            all_task_importance['CLM'].append(drop)
        
        # CLU importance
        without_clu = model_data[(model_data['clm'] > 0) & (model_data['clu'] == 0) & (model_data['cls'] > 0)]
        if len(without_clu) > 0:
            drop = full_f1 - without_clu['f1_macro'].max()
            all_task_importance['CLU'].append(drop)
        
        # CLS importance
        without_cls = model_data[(model_data['clm'] > 0) & (model_data['clu'] > 0) & (model_data['cls'] == 0)]
        if len(without_cls) > 0:
            drop = full_f1 - without_cls['f1_macro'].max()
            all_task_importance['CLS'].append(drop)
    
    # Report average importance
    if any(len(v) > 0 for v in all_task_importance.values()):
        output.append("Average task importance across all models:")
        output.append("-"*80)
        
        importance_summary = []
        for task, drops in all_task_importance.items():
            if len(drops) > 0:
                avg_drop = np.mean(drops)
                std_drop = np.std(drops)
                importance_summary.append((task, avg_drop, std_drop, len(drops)))
        
        importance_summary.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (task, avg_drop, std_drop, n) in enumerate(importance_summary, 1):
            sign = '-' if avg_drop > 0 else '+'
            output.append(
                f"{rank}. {task:<10} "
                f"Avg drop: {sign}{abs(avg_drop):.3f} ± {std_drop:.3f}  "
                f"(n={n} models)"
            )
        
        output.append("")
        output.append("KEY FINDING:")
        if importance_summary:
            most_important = importance_summary[0][0]
            least_important = importance_summary[-1][0]
            output.append(f"  Most critical task:     {most_important} (avg drop: {importance_summary[0][1]:.3f})")
            output.append(f"  Least critical task:    {least_important} (avg drop: {importance_summary[-1][1]:.3f})")
    
    output.append("")
    return "\n".join(output)

        
    


def generate_all_tables(csv_path: str) -> str:
    """Generate all analysis tables."""
    baseline, pretrained = load_and_process_data(csv_path)
    
    output = []
    output.append("\n" + "="*80)
    output.append("MULTI-TASK PRE-TRAINING: ANALYSIS TABLES")
    output.append("="*80)
    output.append(f"Dataset: {csv_path}")
    output.append(f"Total Experiments: {len(baseline) + len(pretrained)}")
    output.append(f"Baseline Models: {len(baseline)}")
    output.append(f"Pre-trained Configs: {len(pretrained)}")
    output.append("="*80)
    output.append("")
    
    # output.append(table_1_baseline_performance(baseline))
    output.append(table_2_best_pretrained(baseline, pretrained))
    # output.append(table_3_detailed_metrics(baseline, pretrained))
    
    
    # output.append(table_7_finetuning_comparison(baseline, pretrained))
    # output.append(table_8_ablation_study(pretrained))
    
    # output.append(table_4_optimal_configs(pretrained))
    # output.append(table_5_clu_contribution(baseline, pretrained))

    
    return "\n".join(output)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_tables.py <input_csv>")
        print("Example: python generate_tables.py aggregated_metrics.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        tables = generate_all_tables(csv_path)
        print(tables)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()