import pandas as pd

def generate_performance_table(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # helper to extract core architecture name
    # e.g., 'baseline_bert-base-eurlex' -> 'bert-base-eurlex'
    # e.g., 'bert-base-eurlex_m2m100...' -> 'bert-base-eurlex'
    def get_core_name(name):
        clean = name.replace('baseline_', '')
        return clean.split('_m2m100')[0].replace('.pt', '')

    df['core_model'] = df['Model_Name'].apply(get_core_name)
    
    results = []
    
    # Iterate through unique model architectures
    for model in df['core_model'].unique():
        model_group = df[df['core_model'] == model]
        
        # 1. Identify Baseline
        base_row = model_group[model_group['Model_Name'].str.contains('baseline')]
        if base_row.empty:
            continue
            
        base_f1 = base_row['F1_Macro'].values[0]
        
        # 2. Identify Best StVO Fine-tuned (The .pt checkpoints)
        ft_rows = model_group[model_group['Model_Name'].str.contains('.pt')]
        if ft_rows.empty:
            best_ft_f1 = base_f1 # No improvement if no FT models
        else:
            best_ft_f1 = ft_rows['F1_Macro'].max()
            
        results.append({
            'Base Model': model,
            'Base F1': round(base_f1, 4),
            'StVO Fine-tuned F1': round(best_ft_f1, 4),
            'Improvement (Δ)': round(best_ft_f1 - base_f1, 4)
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Sort by best fine-tuned score
    summary_df = summary_df.sort_values(by='StVO Fine-tuned F1', ascending=False)
    
    return summary_df

if __name__ == "__main__":
    print("Generating model performance comparison table for austrian driving license dataset...")
    # file_path = "../results/model-performance-clm_nsp_clu_cls-austrian-TMUX_20260220_133049-stvo-20_epoch_10_segments_transe.csv"
    file_path = "../results/model-performance-clm_nsp_clu_cls-austrian-TMUX_20260512_110205-stvo-20_epoch_10_segments_transe_final.csv"

    print(generate_performance_table(file_path))
    print("===================================================================")

    print("Generating model performance comparison table for irish driving license dataset...")
    # file_path = "../results/model-performance-clm_nsp_clu_cls-irish-TMUX_20260220_132851-stvo-20_epoch_10_segments_transe.csv"
    file_path = "../results/model-performance-clm_nsp_clu_cls-irish-TMUX_20260512_123705-stvo-20_epoch_10_segments_transe_final.csv"
    print(generate_performance_table(file_path))
    print("===================================================================")
    
    