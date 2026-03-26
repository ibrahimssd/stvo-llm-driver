import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mticker
import re



class Plotter:
    def __init__(self, input_csv_file=None, output_csv_file=None):
        """
        Initializes the Plotter class with the CSV file path.

        Parameters:
        - csv_file: str, path to the CSV file.
        """
        self.input_csv_file = input_csv_file
        self.output_csv_file = output_csv_file
        self.df = None  # DataFrame to hold the data
        self.metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        if input_csv_file:
            self.file_name = self.input_csv_file.split('/')[-1].split('.')[0]

    def read_data(self):
        """
        Reads the CSV file and preprocesses the data.
        """
        # Read the CSV file
        self.df = pd.read_csv(self.input_csv_file)

        # Ensure Temperature is numeric
        self.df['Temperature'] = pd.to_numeric(self.df['Temperature'], errors='coerce')

        # Remove any rows with NaN Temperature
        self.df = self.df.dropna(subset=['Temperature'])

        # Convert Temperature to float
        self.df['Temperature'] = self.df['Temperature'].astype(float)

        # Get unique models
        self.models = self.df['Model_Checkpoint'].unique()

        # get number of questions 
        self.Number_of_Subquestions = self.df['Number_of_Subquestions'].unique()
        self.number_of_true_yes = self.df['number_of_true_yes'].unique()
        self.number_of_true_no = self.df['number_of_true_no'].unique()
        self.number_of_pred_yes = self.df['number_of_pred_yes'].unique()
        self.number_of_pred_no = self.df['number_of_pred_no'].unique()

        
    def plot_tempreture_metrics(self):
        """
        Generates plots for different metrics.
        Each plot shows Temperature vs. Metric Value for all unique models.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please run read_data() first.")

        # Set the style for seaborn
        sns.set(style='whitegrid')

        # For each metric, create a plot
        for metric in self.metrics:
            plt.figure(figsize=(20, 15))
            sns.lineplot(
                data=self.df,
                x='Temperature',
                y=metric,
                hue='Model_Checkpoint',
                style='Model_Checkpoint',
                markers=True,
                dashes=False
            )
            plt.title(f'Temperature vs {metric}', fontsize=16)
            plt.xlabel('Temperature', fontsize=14)
            plt.ylabel(metric, fontsize=14)
            plt.legend(title='Model', fontsize=12, title_fontsize=13)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            # add number of subquestions to the buttom of the plot
            plt.text(0.5, 0.01, f'Number of Questions: {self.Number_of_Subquestions[0]}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.05, f'Number of True Yes: {self.number_of_true_yes[0]}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.13, f'Number of Predicted Yes: {self.number_of_pred_yes[0]}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.09, f'Number of True No: {self.number_of_true_no[0]}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.17, f'Number of Predicted No: {self.number_of_pred_no[0]}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
            
            # Save the plot as an image file
            plt.savefig(f'plots/Temperature_vs_{metric}_{self.file_name}.png')


    def _create_bar_plot(self,data, x, y, xlabel, ylabel, title, palette, output_path, xtick_rotation=0):
        """
        Helper function to create and save a bar plot with data labels.
        """
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=x, y=y, data=data, palette=palette)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if necessary
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')

        # Add data labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points', fontsize=12, weight='bold')

        # Save the plot
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def bar_plot(self):
        """
        Generates bar plots for different metrics using Seaborn.
        """
        # Ensure the output directory exists
        os.makedirs('plots', exist_ok=True)

        # Read CSV file
        df = pd.read_csv(self.input_csv_file)

        # Set seaborn style
        sns.set(style="whitegrid")
        plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titlesize': 16, 'axes.titleweight': 'bold'})

        # Find the best prompt version based on accuracy for each temperature
        best_prompt_by_temp = df.loc[df.groupby('Temperature')['Accuracy'].idxmax()]
        best_prompt = best_prompt_by_temp.loc[best_prompt_by_temp['Accuracy'].idxmax()][['Prompt_Version', 'Temperature', 'Accuracy', 'MAX_New_Tokens']]
        print("Best Prompt:", best_prompt)

        # Plot 1: Temperature vs Model Accuracy for the best prompt based on accuracy
        self._create_bar_plot(
            data=best_prompt_by_temp,
            x='Temperature',
            y='Accuracy',
            xlabel='Temperature',
            ylabel='Accuracy',
            title=f'Temperature vs Model Accuracy for the Best Prompt {best_prompt["Prompt_Version"]}',
            palette='Blues_d',
            output_path=f'plots/best_temperature_by_accuracy_{self.file_name}.png'
        )

        # Find the best temperature based on accuracy for each prompt version
        best_temp_by_prompt = df.loc[df.groupby('Prompt_Version')['Accuracy'].idxmax()]
        best_tempreture = best_temp_by_prompt.loc[best_temp_by_prompt['Accuracy'].idxmax()][['Prompt_Version', 'Temperature', 'Accuracy', 'MAX_New_Tokens']]
        print("Best Temperature:", best_tempreture)

        # Plot 2: Prompt Versions vs Accuracy for the best temperature based on accuracy
        self._create_bar_plot(
            data=best_temp_by_prompt,
            x='Prompt_Version',
            y='Accuracy',
            xlabel='Prompt Version',
            ylabel='Accuracy',
            title=f'Prompt Versions vs Accuracy for the Best Temperature {best_tempreture["Temperature"]}',
            palette='Reds_d',
            output_path=f'plots/best_prompt_version_by_accuracy_{self.file_name}.png',
            xtick_rotation=45
        )

        # Plot 3: MAX_New_Tokens vs Accuracy for the best temperature and prompt version
        best_temp_prompt = df.loc[(df['Temperature'] == best_tempreture['Temperature']) & 
                                (df['Prompt_Version'] == best_tempreture['Prompt_Version'])]
        self._create_bar_plot(
            data=best_temp_prompt,
            x='MAX_New_Tokens',
            y='Accuracy',
            xlabel='MAX_New_Tokens',
            ylabel='Accuracy',
            title=f'MAX_New_Tokens vs Accuracy for Best Temperature {best_tempreture["Temperature"]} and Prompt Version {best_tempreture["Prompt_Version"]}',
            palette='Greens_d',
            output_path=f'plots/best_MAX_New_Tokens_by_accuracy_{self.file_name}.png',
            xtick_rotation=45
        )

            
            
    def select_best_model_config(self):
        """
        Selects the best model configurations based on the highest accuracy for each unique model checkpoint.
        Model_Checkpoint,Number_of_Questions,Number_of_Subquestions,number_of_true_yes,number_of_true_no,number_of_pred_yes,number_of_pred_no,number_of_pred_invalid,Accuracy,Precision,Recall,F1,MCC,Seed,Max_Length,MAX_New_Tokens,Top_P,Top_K,Temperature,Repetition_Penalty,evaluation_mode
        """

        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.input_csv_file)

        # Group by 'Model_Checkpoint' and find the row with the highest accuracy in each group
        best_models = df.loc[df.groupby('Model_Checkpoint')['Accuracy'].idxmax()]

        # order from highest to lowest accuracy
        best_models = best_models.sort_values(by='Accuracy', ascending=False)

        # remove evaluation_mode , top_p, top_k, repetition_penalty , max length , Number_of_pred_yes, Number_of_pred_no, Number_of_pred_invalid
        best_models = best_models.drop(columns=['evaluation_mode', 'Top_P', 'Top_K', 'Repetition_Penalty', 'Max_Length', 'number_of_pred_yes', 'number_of_pred_no', 'number_of_pred_invalid'])

        # Save the best model configurations to a new CSV file
        best_models.to_csv(self.output_csv_file, index=False)
        print("Saved the best model configurations to:", self.output_csv_file)


    def plot_training_progress(self, args, history: dict, save_path: str = "training_progress.png"):
        """
        Plots the training and validation progress from the history dictionary
        in a publication-ready format with adaptive layout based on available data.

        Args:
            args: An object containing training arguments, specifically clm_mlm_lambda and clu_lambda.
            history (dict): A dictionary containing lists of metrics. Expected keys:
                            "steps", "clm_loss_step", "clu_loss_step",
                            "combined_loss_step", "epochs", "train_loss_epoch",
                            "val_loss_epoch", "val_f1_epoch".
            save_path (str): The path to save the generated plot image.
        """
        if not history or not history.get('epochs'):
            print("History is empty or lacks 'epochs' data. Cannot generate plots.")
            return

        if args.modeling_type == 'causal':
            core_task = 'clm'
        elif args.modeling_type == 'masked':
            core_task = 'nsp'

        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'axes.edgecolor': 'black',
            'axes.axisbelow': True
        })

        # Check data availability
        has_val_loss = 'val_loss_epoch' in history and len(history['val_loss_epoch']) > 0
        has_val_f1 = 'val_f1_epoch' in history and len(history['val_f1_epoch']) > 0
        has_train_loss = 'train_loss_epoch' in history and len(history['train_loss_epoch']) > 0
        has_step_data = 'steps' in history and len(history.get('steps', [])) > 0
        clm_loss_step = f'{core_task}_loss_step' in history and len(history[f'{core_task}_loss_step']) > 0
        clu_loss_step = 'clu_loss_step' in history and len(history['clu_loss_step']) > 0
        cls_loss_step = 'cls_loss_step' in history and len(history['cls_loss_step']) > 0    

      

        # Determine layout based on available data
        if has_val_loss and has_val_f1:
            # Full 2x2 layout
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            layout = 'full'
        elif has_val_loss or has_val_f1:
            # 2x2 layout with one validation metric
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            layout = 'partial_val'
        else:
            # 1x2 layout - only training data
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            layout = 'train_only'

        # Calculate lambda values
        clm_mlm_lambda = getattr(args, 'clm_mlm_lambda', 0)
        cls_lambda = getattr(args, 'cls_lambda', 0)
        clu_lambda = 1- (clm_mlm_lambda + cls_lambda)


        # Create title with lambda values
        title = f'Multi-Task Training Progress\n'
        title += f'$\\lambda_{{CLM_MLM}}$={clm_mlm_lambda:.2f}, $\\lambda_{{CLU}}$={clu_lambda:.2f}, $\\lambda_{{CLS}}$={cls_lambda:.2f}'
        
        fig.suptitle(title, fontsize=16, y=0.98 if layout != 'train_only' else 0.95)

        epochs = history['epochs']
        
        # Define color palette for consistency
        colors = {
            'train': '#1f77b4',      # Blue
            'val': '#ff7f0e',        # Orange
            'f1': '#2ca02c',         # Green
            'combined': '#d62728',   # Red
            f'{core_task}': '#9467bd',        # Purple
            'clu': '#8c564b',        # Brown
            'cls': '#7f7f7f',        # Grey
        }

        def setup_axis(ax, title, xlabel, ylabel, scientific=True):
            """Helper function to setup a axis with consistent styling"""
            ax.set_title(title, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if scientific:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Plot 1: Training Loss (and Validation Loss if available)
        if layout == 'train_only':
            ax1 = axs[0]
        else:
            ax1 = axs[0, 0]
        
        if has_train_loss:
            ax1.plot(epochs, history['train_loss_epoch'], 
                    marker='o', linestyle='-', color=colors['train'], 
                    markersize=4, label='Training Loss', linewidth=2)
        
        if has_val_loss:
            ax1.plot(epochs, history['val_loss_epoch'], 
                    marker='s', linestyle='-', color=colors['val'], 
                    markersize=4, label='Validation Loss', linewidth=2)
        
        title_text = 'Training Loss' if not has_val_loss else 'Training vs. Validation Loss'
        setup_axis(ax1, f'{title_text} per Epoch', 'Epoch', 'Loss')
        ax1.set_xticks(epochs)
        ax1.legend(frameon=True, fancybox=True, shadow=True, loc='best')

        # Plot 2: Validation F1 Score or Step-wise Combined Loss
        if layout == 'train_only':
            ax2 = axs[1]
            # Show step-wise combined loss for training-only layout
            if has_step_data and 'combined_loss_step' in history:
                steps = history['steps']
                window_size = min(50, max(1, len(steps) // 10))
                combined_loss_data = pd.DataFrame({'Combined Loss': history['combined_loss_step']})
                clm_loss_step = pd.DataFrame({f'{core_task} Loss': history[f'{core_task}_loss_step']}) if f'{core_task}_loss_step' in history else pd.DataFrame()
                clu_loss_step = pd.DataFrame({'CLU Loss': history['clu_loss_step']}) if 'clu_loss_step' in history else pd.DataFrame()
                cls_loss_step = pd.DataFrame({'CLS Loss': history['cls_loss_step']}) if 'cls_loss_step' in history else pd.DataFrame()

                smoothed_combined_loss = combined_loss_data.rolling(window=window_size, min_periods=1).mean()
                ax2.plot(steps, smoothed_combined_loss['Combined Loss'], 
                        color=colors['combined'], linewidth=2, label='Combined Loss (Smoothed)')
                if not clm_loss_step.empty:
                    smoothed_clm_loss = clm_loss_step.rolling(window=window_size, min_periods=1).mean()
                    ax2.plot(steps, smoothed_clm_loss[f'{core_task} Loss'], 
                            color=colors[core_task], linewidth=2, label=f'{core_task} Loss (Smoothed)')
                if not clu_loss_step.empty:
                    smoothed_clu_loss = clu_loss_step.rolling(window=window_size, min_periods=1).mean()
                    ax2.plot(steps, smoothed_clu_loss['CLU Loss'], 
                            color=colors['clu'], linewidth=2, label='CLU Loss (Smoothed)')
                if not cls_loss_step.empty:
                    smoothed_cls_loss = cls_loss_step.rolling(window=window_size, min_periods=1).mean()
                    ax2.plot(steps, smoothed_cls_loss['CLS Loss'], 
                            color=colors['cls'], linewidth=2, label='CLS Loss (Smoothed)')
                setup_axis(ax2, 'Smoothed Combined Training Loss per Step', 'Training Steps', 'Loss')

                ax2.legend(frameon=True, fancybox=True, shadow=True)
            
            else:
                ax2.text(0.5, 0.5, 'No step-wise data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                setup_axis(ax2, 'Combined Training Loss per Step', 'Training Steps', 'Loss')
        else:
            ax2 = axs[0, 1]
            if has_val_f1:
                ax2.plot(epochs, history['val_f1_epoch'], 
                        marker='o', linestyle='-', color=colors['f1'], 
                        markersize=4, label='Validation F1 Score', linewidth=2)
                setup_axis(ax2, 'Validation F1 Score per Epoch', 'Epoch', 'F1 Score', scientific=False)
                ax2.set_xticks(epochs)
                ax2.set_ylim(0, 1.05)
                ax2.legend(frameon=True, fancybox=True, shadow=True)
            else:
                ax2.text(0.5, 0.5, 'No validation F1 data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                setup_axis(ax2, 'Validation F1 Score per Epoch', 'Epoch', 'F1 Score', scientific=False)

        # Plot 3 & 4: Step-wise losses (only for 2x2 layouts)
        if layout != 'train_only':
            # Plot 3: Smoothed Combined Training Loss per Step
            ax3 = axs[1, 0]
            if has_step_data and 'combined_loss_step' in history:
                steps = history['steps']
                window_size = min(50, max(1, len(steps) // 10))
                combined_loss_data = pd.DataFrame({'Combined Loss': history['combined_loss_step']})
                smoothed_combined_loss = combined_loss_data.rolling(window=window_size, min_periods=1).mean()
                
                ax3.plot(steps, smoothed_combined_loss['Combined Loss'], 
                        color=colors['combined'], linewidth=2, label=f'Combined Loss (MA-{window_size})')
                setup_axis(ax3, 'Smoothed Combined Training Loss', 'Training Steps', 'Loss')
                ax3.legend(frameon=True, fancybox=True, shadow=True)
            else:
                ax3.text(0.5, 0.5, 'No step-wise data available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                setup_axis(ax3, 'Combined Training Loss per Step', 'Training Steps', 'Loss')

            # Plot 4: Smoothed Individual Task Training Losses per Step
            ax4 = axs[1, 1]
            if has_step_data:
                steps = history['steps']
                window_size = min(50, max(1, len(steps) // 10))
                
                # Plot available individual losses
                individual_losses = {
                    f'{core_task} Loss': (f'{core_task}_loss_step', colors[core_task]),
                    'CLU Loss': ('clu_loss_step', colors['clu']),
                    'CLS Loss': ('cls_loss_step', colors['cls']),
                }
                
                legend_elements = []
                for loss_name, (key, color) in individual_losses.items():
                    if key in history and len(history[key]) > 0:
                        loss_data = pd.DataFrame({loss_name: history[key]})
                        smoothed_loss = loss_data.rolling(window=window_size, min_periods=1).mean()
                        ax4.plot(steps, smoothed_loss[loss_name], 
                                color=color, linewidth=2, alpha=0.8, 
                                label=f'{loss_name} (MA-{window_size})')
                        legend_elements.append(loss_name)
                
                if legend_elements:
                    setup_axis(ax4, 'Smoothed Individual Task Losses', 'Training Steps', 'Loss')
                    ax4.legend(frameon=True, fancybox=True, shadow=True, loc='best')
                else:
                    ax4.text(0.5, 0.5, 'No individual loss data available', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    setup_axis(ax4, 'Individual Task Losses per Step', 'Training Steps', 'Loss')
            else:
                ax4.text(0.5, 0.5, 'No step-wise data available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                setup_axis(ax4, 'Individual Task Losses per Step', 'Training Steps', 'Loss')

        # Add data availability annotation
        data_status = []
        if has_train_loss: data_status.append("Training Loss")
        if has_val_loss: data_status.append("Validation Loss")
        if has_val_f1: data_status.append("Validation F1")
        if has_step_data: data_status.append("Step-wise Data")
        
        status_text = f"Available data: {', '.join(data_status)}"
        fig.text(0.02, 0.02, status_text, fontsize=9, style='italic', alpha=0.7)

        # Adjust layout
        if layout == 'train_only':
            plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        else:
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the plot
        try:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            
            # Save in multiple formats for publication
            base_path = os.path.splitext(save_path)[0]
            
            # High-resolution PNG for papers
            plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            
            
            
            print(f"Training progress plots saved:")
            print(f"  - PNG (300 DPI): {base_path}.png")
           
        except Exception as e:
            print(f"Failed to save plot to {save_path}: {e}")
        finally:
            plt.close(fig)


    def plot_finetuning_performance(self,file_path: str, model_family: str, dataset_name: str, output_filename: str):
        """
        Reads model performance data from a CSV and creates a publication-quality
        bar chart showing the impact of fine-tuning variations.
        """
        df = pd.read_csv(file_path)
        df_filtered = df[df['data_name'] == dataset_name]
        model_df = df_filtered[df_filtered['Model_Name'].str.contains(model_family)].copy()

        baseline_row = model_df[model_df['Model_Name'] == f'baseline_{model_family}']
        if baseline_row.empty:
            print(f"Error: Baseline for '{model_family}' not found for dataset '{dataset_name}'.")
            return
        baseline_accuracy = baseline_row['Accuracy'].iloc[0]

        ft_df = model_df[model_df['Model_Name'] != f'baseline_{model_family}'].copy()

        def extract_weights(model_name):
            match = re.search(r'_mlm_([0-9\.]+)_clu_([0-9\.]+)_cls_([0-9\.]+)', model_name)
            if match:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
            return None, None, None

        ft_df[['mlm', 'clu', 'cls']] = ft_df['Model_Name'].apply(lambda x: pd.Series(extract_weights(x)))
        ft_df['config_label'] = ft_df.apply(
            lambda row: f"MLM {row['mlm']}\nCLU {row['clu']}\nCLS {row['cls']}", axis=1
        )
        ft_df = ft_df.sort_values(by=['mlm', 'cls']).reset_index()

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(18, 9))

        barplot = sns.barplot(
            x='config_label', 
            y='Accuracy', 
            hue='mlm', 
            data=ft_df,
            palette='viridis',
            dodge=False
        )

        plt.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2, 
                    label=f'Baseline Accuracy ({baseline_accuracy:.2%})')

        for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.2%}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=10)

        plt.title(f'Fine-Tuning Performance of "{model_family}" on {dataset_name.title()} Dataset', fontsize=20, pad=20)
        plt.xlabel('Fine-Tuning Objective Weights', fontsize=16, labelpad=15)
        plt.ylabel('Accuracy', fontsize=16, labelpad=15)
        
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=12)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, title='MLM Weight', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=14)

        plt.ylim(bottom=baseline_accuracy * 0.9)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved successfully to '{output_filename}'")

    def plot_finetuning_heatmap(self,file_path: str, model_family: str, dataset_name: str, output_filename: str):
        """
        Reads model performance data and creates a publication-quality heatmap
        to visualize the impact of fine-tuning objective weights (CLU vs. CLS).
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'.")
            return

        # --- 1. Data Filtering and Preparation ---
        df_filtered = df[(df['data_name'] == dataset_name) & (df['Model_Name'].str.contains(model_family))]
        
        baseline_row = df_filtered[df_filtered['Model_Name'] == f'baseline_{model_family}']
        if baseline_row.empty:
            print(f"Error: Baseline for '{model_family}' not found.")
            return
        baseline_accuracy = baseline_row['Accuracy'].iloc[0]

        ft_df = df_filtered[df_filtered['Model_Name'] != f'baseline_{model_family}'].copy()

        # --- 2. Extract Hyperparameter Weights ---
        def extract_weights(model_name):
            # This pattern handles both clm and mlm prefixes
            match = re.search(r'_(?:clm|mlm)_([0-9\.]+)_clu_([0-9\.]+)_cls_([0-9\.]+)', model_name)
            if match:
                # We only need clu and cls for the heatmap axes
                return float(match.group(2)), float(match.group(3))
            return None, None

        ft_df[['clu', 'cls']] = ft_df['Model_Name'].apply(lambda x: pd.Series(extract_weights(x)))
        ft_df.dropna(subset=['clu', 'cls'], inplace=True)

        # --- 3. Create Pivot Table for Heatmap ---
        performance_pivot = ft_df.pivot_table(index='clu', columns='cls', values='Accuracy')

        # Find the position of the best performance to highlight it
        max_acc = ft_df['Accuracy'].max()
        best_config = ft_df.loc[ft_df['Accuracy'].idxmax()]
        
        # --- 4. Generate the Heatmap ---
        sns.set_theme(style="white")
        plt.figure(figsize=(12, 8))

        heatmap = sns.heatmap(
            performance_pivot,
            annot=True,          # Show accuracy values in cells
            fmt=".3f",           # Format to 3 decimal places
            cmap='viridis',      # A visually appealing color map
            linewidths=.5,
            annot_kws={"size": 12, "weight": "bold"}
        )

        # --- 5. Enhancements and Annotations ---
        # Highlight the best-performing cell
        for i, idx in enumerate(performance_pivot.index):
            for j, col in enumerate(performance_pivot.columns):
                if performance_pivot.loc[idx, col] == max_acc:
                    heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
                    break

        # Set title and labels with clear, informative text
        plt.title(
            f'Accuracy Landscape for {model_family.replace("-", " ").title()}\nBaseline Accuracy: {baseline_accuracy:.3f}\n MLM= 1 - (CLU + CLS)',
            fontsize=18, weight='bold', pad=20
        )
        plt.xlabel('Classification Weight (CLS)', fontsize=14, weight='bold', labelpad=15)
        plt.ylabel('Clustering Weight (CLU)', fontsize=14, weight='bold', labelpad=15)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        # Adjust the color bar
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Model Accuracy', rotation=270, labelpad=20, fontsize=14, weight='bold')

        plt.tight_layout()
        
        # create plots directory if not exists
        os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)
        # --- 6. Save the Plot ---
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved successfully to '{output_filename}'")



            
if __name__ == "__main__":
    
    

    # Example usage
    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no-temp-prompts-tokens_german.csv'
    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no-temp-prompts_german.csv'

    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no_austrian.csv'
    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no_irish.csv'
    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no_german.csv'
    # output_file = f'tables/best_model_config_{input_file.split("/")[-1]}'
    # plotter = Plotter(input_file, output_file)
    # plotter.select_best_model_config()

    # # Example usage
    # # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no-temp-prompts_german.csv'
    # # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no-temp-prompts-tokens_german.csv'
    # input_file = 'results/baseline-model-performance-full-precision-batch-non-positive-no-temp-prompts-tokens-no_sysprompt_german.csv'
    # plotter = Plotter(input_file)
    # plotter.bar_plot()


    # visualize model variation example
    # The data provided by the user
    # --- Main execution ---
    # csv_file = './results/model-performance-clm_mlm_clu_cls-german-2572-final.csv'
    csv_file = './results/final-1/model-performance-clm_mlm_clu_cls-austrian-2719-final-stvo-part-1-final.csv'
    plotter = Plotter()
    
    df = pd.read_csv(csv_file)
    # baslines 
    families  = df[df['Model_Name'].str.contains('baseline')]
    for family in families['Model_Name'].unique():
        print(family)
        plotter.plot_finetuning_heatmap(
            file_path=csv_file,
            model_family=family.replace('baseline_',''),
            dataset_name='austrian',
            output_filename=f'family_plots/finetuning_heatmap_{family.replace("baseline_","")}_austrian.png'
        )
    # plotter.plot_finetuning_heatmap(
    #         file_path=csv_file,
    #         model_family='bert-base-uncased',
    #         dataset_name='austrian',
    #         output_filename='finetuning_heatmap_austrian.png'
    #     )
    
    
    