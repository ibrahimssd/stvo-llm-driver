#!/usr/bin/env python3
"""
Clustering Visualization in Embedding Space
============================================

Visualizes Q&A embeddings using t-SNE, UMAP, and PCA to demonstrate
LLM's ability to separate correct vs incorrect answers and different
legal concepts. Higher separation indicates better data quality.

Author: For Synthetic Legal Q&A Dataset Paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional packages (install if needed)
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/transformers not available. Install with: pip install torch transformers")

# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class EmbeddingExtractor:
    """Extract embeddings from Q&A text using pre-trained models"""
    
    def __init__(self, model_name: str = "dlicari/Italian-Legal-BERT", model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize embedding extractor
        
        Args:
            model_name: HuggingFace model ID
            device: 'cuda' or 'cpu'
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and transformers required. Install with: pip install torch transformers")
        
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_path if model_path else model_name)
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded on device: {device}")
    
    def get_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Get [CLS] token embedding for text
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            Embedding vector (numpy array)
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding.squeeze().cpu().numpy()
    
    def extract_embeddings_batch(self, texts: List[str], 
                                batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for multiple texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings (num_texts, embedding_dim)
        """
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                emb = self.get_embedding(text)
                batch_embeddings.append(emb)
            
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % (batch_size * 5) == 0:
                print(f"Processed {min(i + batch_size, total)}/{total} texts")
        
        return np.array(embeddings)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_qa_dataset(jsonl_path: str, max_samples: Optional[int] = None) -> Tuple[List, List, List, List]:
    """
    Load Q&A dataset from JSONL file
    
    Args:
        jsonl_path: Path to JSONL file
        max_samples: Maximum samples to load (None = all)
    
    Returns:
        Tuple of (questions, answers, labels)
    """
    questions = []
    answers = []
    paragraphs = []
    labels = []
    
    print(f"Loading dataset from: {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                item = json.loads(line)
                questions.append(item['question'])
                answers.append(item['answer'])
                paragraphs.append(item.get('paragraph', 'N/A'))
                labels.append('Correct' if item['label'] == "yes" else 'Incorrect')
            
            except json.JSONDecodeError:
                print(f"Error parsing line {idx}")
                continue
    
    print(f"Loaded {len(questions)} samples")
    return questions, answers, labels , paragraphs


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

class DimensionalityReducer:
    """Reduce embeddings to 2D for visualization"""
    
    @staticmethod
    def apply_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Apply PCA dimensionality reduction"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required. Install with: pip install scikit-learn")
        
        print(f"Applying PCA (original shape: {embeddings.shape})")
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        
        # Calculate explained variance
        explained_var = sum(pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.2%}")
        
        return reduced
    
    @staticmethod
    def apply_tsne(embeddings: np.ndarray, n_components: int = 2, 
                   perplexity: int = 30, random_state: int = 42) -> np.ndarray:
        """Apply t-SNE dimensionality reduction"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required. Install with: pip install scikit-learn")
        
        print(f"Applying t-SNE (original shape: {embeddings.shape})")
        print(f"Parameters: perplexity={perplexity}, random_state={random_state}")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )
        reduced = tsne.fit_transform(embeddings)
        print("t-SNE completed")
        
        return reduced
    
    @staticmethod
    def apply_umap(embeddings: np.ndarray, n_components: int = 2,
                   n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """Apply UMAP dimensionality reduction"""
        if not UMAP_AVAILABLE:
            raise RuntimeError("UMAP required. Install with: pip install umap-learn")
        
        print(f"Applying UMAP (original shape: {embeddings.shape})")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=True
        )
        reduced = reducer.fit_transform(embeddings)
        print("UMAP completed")
        
        return reduced


# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def calculate_cluster_separation(embeddings_2d: np.ndarray, labels: List[str]) -> Dict:
    """
    Calculate metrics for cluster separation quality
    
    Args:
        embeddings_2d: 2D embeddings
        labels: List of labels for each sample
    
    Returns:
        Dictionary with separation metrics
    """
    metrics = {}
    
    # Get indices for each label
    correct_indices = np.where(np.array(labels) == 'Correct')[0]
    incorrect_indices = np.where(np.array(labels) == 'Incorrect')[0]
    
    # Calculate centroids
    correct_centroid = embeddings_2d[correct_indices].mean(axis=0)
    incorrect_centroid = embeddings_2d[incorrect_indices].mean(axis=0)
    
    # Calculate centroid distance
    centroid_distance = np.linalg.norm(correct_centroid - incorrect_centroid)
    metrics['centroid_distance'] = centroid_distance
    
    # Calculate intra-cluster distances (compactness)
    correct_distances = np.linalg.norm(
        embeddings_2d[correct_indices] - correct_centroid, axis=1
    )
    incorrect_distances = np.linalg.norm(
        embeddings_2d[incorrect_indices] - incorrect_centroid, axis=1
    )
    
    metrics['correct_compactness'] = correct_distances.mean()
    metrics['incorrect_compactness'] = incorrect_distances.mean()
    metrics['avg_compactness'] = (correct_distances.mean() + incorrect_distances.mean()) / 2
    
    # Calculate separation quality (higher is better)
    # Separation ratio: centroid_distance / avg_compactness
    separation_ratio = centroid_distance / metrics['avg_compactness']
    metrics['separation_ratio'] = separation_ratio
    
    # Calculate silhouette-like metric
    correct_to_incorrect = np.linalg.norm(
        embeddings_2d[correct_indices] - incorrect_centroid[np.newaxis, :], axis=1
    ).mean()
    incorrect_to_correct = np.linalg.norm(
        embeddings_2d[incorrect_indices] - correct_centroid[np.newaxis, :], axis=1
    ).mean()
    
    metrics['quality_score'] = (
        (correct_to_incorrect - metrics['correct_compactness']) + 
        (incorrect_to_correct - metrics['incorrect_compactness'])
    ) / 2
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_embeddings_by_label(embeddings_2d: np.ndarray, labels: List[str],
                            title: str = "Embedding Space Visualization",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot embeddings colored by label (correct vs incorrect)
    
    Args:
        embeddings_2d: 2D embeddings
        labels: List of labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate by label
    correct_mask = np.array(labels) == 'Correct'
    incorrect_mask = np.array(labels) == 'Incorrect'
    
    # Plot
    ax.scatter(
        embeddings_2d[correct_mask, 0],
        embeddings_2d[correct_mask, 1],
        c='#2ecc71', s=30, alpha=0.6, label='Correct (Yes)',
        edgecolors='darkgreen', linewidth=0.5
    )
    ax.scatter(
        embeddings_2d[incorrect_mask, 0],
        embeddings_2d[incorrect_mask, 1],
        c='#e74c3c', s=30, alpha=0.6, label='Incorrect (No)',
        edgecolors='darkred', linewidth=0.5
    )
    
    # Calculate and plot centroids
    correct_centroid = embeddings_2d[correct_mask].mean(axis=0)
    incorrect_centroid = embeddings_2d[incorrect_mask].mean(axis=0)
    
    ax.scatter(
        [correct_centroid[0]], [correct_centroid[1]],
        c='darkgreen', s=300, marker='*', edgecolors='black', linewidth=2,
        label='Correct Centroid', zorder=5
    )
    ax.scatter(
        [incorrect_centroid[0]], [incorrect_centroid[1]],
        c='darkred', s=300, marker='*', edgecolors='black', linewidth=2,
        label='Incorrect Centroid', zorder=5
    )
    
    # Draw line between centroids
    ax.plot(
        [correct_centroid[0], incorrect_centroid[0]],
        [correct_centroid[1], incorrect_centroid[1]],
        'k--', linewidth=2, alpha=0.5, label='Centroid Distance'
    )
    
    # Formatting
    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_embeddings_by_section(embeddings_2d: np.ndarray, sections: List[str],
                              title: str = "Embedding Space by Legal Section",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (14, 10),
                              top_n_sections: int = 10) -> plt.Figure:
    """
    Plot embeddings colored by legal section (shows concept clustering)
    
    Args:
        embeddings_2d: 2D embeddings
        sections: List of section IDs
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_n_sections: Number of most frequent sections to highlight
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top sections
    unique_sections, counts = np.unique(sections, return_counts=True)
    top_sections = unique_sections[np.argsort(-counts)][:top_n_sections]
    
    # Color palette
    colors = sns.color_palette("husl", len(top_sections))
    
    # Plot other sections in gray
    other_mask = np.array([s not in top_sections for s in sections])
    ax.scatter(
        embeddings_2d[other_mask, 0],
        embeddings_2d[other_mask, 1],
        c='lightgray', s=20, alpha=0.3, label='Other sections'
    )
    
    # Plot top sections with colors
    for idx, section in enumerate(top_sections):
        mask = np.array(sections) == section
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]], s=50, alpha=0.7, label=section,
            edgecolors='black', linewidth=0.5
        )
    
    # Formatting
    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, 
             title=f'Top {top_n_sections} Sections', title_fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_separation_metrics(metrics: Dict, method: str = "t-SNE",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cluster separation quality metrics
    
    Args:
        metrics: Dictionary of metrics from calculate_cluster_separation
        method: Name of dimensionality reduction method
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Cluster Separation Quality Metrics ({method})', 
                fontsize=14, fontweight='bold')
    
    # 1. Centroid Distance
    ax = axes[0, 0]
    ax.bar(['Centroid\nDistance'], [metrics['centroid_distance']], 
          color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Distance', fontweight='bold')
    ax.set_title('Centroid Separation (Higher = Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([metrics['centroid_distance']]):
        ax.text(i, v + 0.1, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. Compactness
    ax = axes[0, 1]
    compactness = [metrics['correct_compactness'], metrics['incorrect_compactness']]
    labels = ['Correct', 'Incorrect']
    bars = ax.bar(labels, compactness, color=['#2ecc71', '#e74c3c'], 
                 alpha=0.7, edgecolor='black')
    ax.set_ylabel('Avg Distance to Centroid', fontweight='bold')
    ax.set_title('Cluster Compactness (Lower = Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{height:.3f}', ha='center', fontweight='bold')
    
    # 3. Separation Ratio
    ax = axes[1, 0]
    separation_ratio = metrics['separation_ratio']
    color = '#27ae60' if separation_ratio > 1.5 else '#f39c12' if separation_ratio > 1.0 else '#e74c3c'
    ax.bar(['Separation\nRatio'], [separation_ratio], color=color, 
          alpha=0.7, edgecolor='black')
    ax.set_ylabel('Ratio', fontweight='bold')
    ax.set_title('Separation Ratio (Higher = Better)\nInterpretation: >1.5 Excellent, >1.0 Good', 
                fontweight='bold', fontsize=10)
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Good threshold')
    ax.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='Excellent threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    for i, v in enumerate([separation_ratio]):
        ax.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. Quality Score
    ax = axes[1, 1]
    quality = metrics['quality_score']
    color = '#27ae60' if quality > 0.5 else '#f39c12' if quality > 0.3 else '#e74c3c'
    ax.bar(['Quality\nScore'], [quality], color=color, 
          alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Overall Quality Score (Higher = Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([quality]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_comparison(results: Dict[str, Tuple[np.ndarray, Dict]],
                   labels: List[str],
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple dimensionality reduction methods
    
    Args:
        results: Dict of {method_name: (embeddings_2d, metrics)}
        labels: List of labels
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    fig.suptitle('Comparison of Dimensionality Reduction Methods', 
                fontsize=14, fontweight='bold')
    
    for idx, (method, (embeddings_2d, metrics)) in enumerate(results.items()):
        ax = axes[idx]
        
        # Separate by label
        correct_mask = np.array(labels) == 'Correct'
        incorrect_mask = np.array(labels) == 'Incorrect'
        
        # Plot
        ax.scatter(
            embeddings_2d[correct_mask, 0],
            embeddings_2d[correct_mask, 1],
            c='#2ecc71', s=20, alpha=0.5, label='Correct'
        )
        ax.scatter(
            embeddings_2d[incorrect_mask, 0],
            embeddings_2d[incorrect_mask, 1],
            c='#e74c3c', s=20, alpha=0.5, label='Incorrect'
        )
        
        # Plot centroids
        correct_centroid = embeddings_2d[correct_mask].mean(axis=0)
        incorrect_centroid = embeddings_2d[incorrect_mask].mean(axis=0)
        
        ax.scatter(*correct_centroid, c='darkgreen', s=200, marker='*', 
                  edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter(*incorrect_centroid, c='darkred', s=200, marker='*', 
                  edgecolors='black', linewidth=1.5, zorder=5)
        
        ax.set_title(f'{method}\nSep. Ratio: {metrics["separation_ratio"]:.3f}',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Dim 1', fontweight='bold')
        ax.set_ylabel('Dim 2', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def visualize_embeddings_pipeline(
    jsonl_path: str,
    output_dir: str = "./clustering_visualization",
    model_name: str = "dlicari/Italian-Legal-BERT",
    model_path: Optional[str] = None,
    max_samples: Optional[int] = 1000,
    methods: List[str] = ["tsne", "umap", "pca"],
    use_qa_concatenation: bool = True,
    compute_metrics: bool = True
) -> Dict:
    """
    Complete pipeline for embedding visualization and clustering analysis
    
    Args:
        jsonl_path: Path to JSONL dataset
        output_dir: Directory to save outputs
        model_name: HuggingFace model for embeddings
        model_path: Local path to model (overrides model_name if provided)
        max_samples: Maximum samples to process
        methods: Dimensionality reduction methods to use
        use_qa_concatenation: Concatenate Q&A or use answers only
        compute_metrics: Calculate clustering metrics
    
    Returns:
        Dictionary with results and metrics
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    questions, answers, labels , paragraphs = load_qa_dataset(jsonl_path, max_samples)
    
    # Extract embeddings
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING EMBEDDINGS")
    print("="*70)
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. Install with: pip install torch transformers")
        return {}
    
    extractor = EmbeddingExtractor(model_name=model_name, model_path=model_path)
    
    if use_qa_concatenation:
        print("Concatenating questions and answers")
        texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]
    else:
        print("Using answers only")
        texts = answers
    
    embeddings = extractor.extract_embeddings_batch(texts, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    results = {}
    reducer = DimensionalityReducer()
    
    # Apply dimensionality reduction methods
    print("\n" + "="*70)
    print("STEP 3: DIMENSIONALITY REDUCTION")
    print("="*70)
    
    embeddings_2d_results = {}
    
    if "pca" in methods and SKLEARN_AVAILABLE:
        print("\n--- PCA ---")
        embeddings_2d_pca = reducer.apply_pca(embeddings, n_components=2)
        embeddings_2d_results['PCA'] = embeddings_2d_pca
    
    if "tsne" in methods and SKLEARN_AVAILABLE:
        print("\n--- t-SNE ---")
        embeddings_2d_tsne = reducer.apply_tsne(embeddings, n_components=2, 
                                               perplexity=min(30, len(embeddings)-1))
        embeddings_2d_results['t-SNE'] = embeddings_2d_tsne
    
    if "umap" in methods and UMAP_AVAILABLE:
        print("\n--- UMAP ---")
        embeddings_2d_umap = reducer.apply_umap(embeddings, n_components=2)
        embeddings_2d_results['UMAP'] = embeddings_2d_umap
    
    # Calculate metrics
    print("\n" + "="*70)
    print("STEP 4: CALCULATING METRICS")
    print("="*70)
    
    metrics_results = {}
    for method, embeddings_2d in embeddings_2d_results.items():
        print(f"\nCalculating metrics for {method}")
        metrics = calculate_cluster_separation(embeddings_2d, labels)
        metrics_results[method] = metrics
        
        print(f"  Centroid Distance: {metrics['centroid_distance']:.4f}")
        print(f"  Separation Ratio: {metrics['separation_ratio']:.4f}")
        print(f"  Quality Score: {metrics['quality_score']:.4f}")
        print(f"  Correct Compactness: {metrics['correct_compactness']:.4f}")
        print(f"  Incorrect Compactness: {metrics['incorrect_compactness']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*70)
    
    for method, embeddings_2d in embeddings_2d_results.items():
        print(f"\nGenerating visualizations for {method}")
        
        # Plot by label
        save_path = f"{output_dir}/{method}_by_label.png"
        plot_embeddings_by_label(
            embeddings_2d, labels,
            title=f"{method}: Correct vs Incorrect Answers",
            save_path=save_path
        )
        
        # Plot by section
        save_path = f"{output_dir}/{method}_by_section.png"
        plot_embeddings_by_section(
            embeddings_2d, paragraphs,
            title=f"{method}: Clustering by Legal Concept",
            save_path=save_path
        )
        
        # Plot metrics
        save_path = f"{output_dir}/{method}_metrics.png"
        plot_separation_metrics(metrics_results[method], method=method, 
                               save_path=save_path)
    
    # Comparison plot
    if len(embeddings_2d_results) > 1:
        print(f"\nGenerating comparison plot for {len(embeddings_2d_results)} methods")
        comparison_data = {
            method: (embeddings_2d, metrics_results[method])
            for method, embeddings_2d in embeddings_2d_results.items()
        }
        save_path = f"{output_dir}/method_comparison.png"
        plot_comparison(comparison_data, labels, save_path=save_path)
    
    # Generate report
    print("\n" + "="*70)
    print("STEP 6: GENERATING REPORT")
    print("="*70)
    
    report = generate_report(labels, metrics_results, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")
    
    return {
        'embeddings': embeddings,
        'embeddings_2d': embeddings_2d_results,
        'metrics': metrics_results,
        'labels': labels,
        'report': report
    }


def generate_report(labels: List[str], metrics_results: Dict,
                   output_dir: str) -> str:
    """Generate summary report"""
    report = f"""
{'='*70}
CLUSTERING VISUALIZATION REPORT
{'='*70}

Dataset Statistics:
  Total Samples: {len(labels)}
  Correct Answers: {sum(1 for l in labels if l == 'Correct')} ({sum(1 for l in labels if l == 'Correct')/len(labels)*100:.1f}%)
  Incorrect Answers: {sum(1 for l in labels if l == 'Incorrect')} ({sum(1 for l in labels if l == 'Incorrect')/len(labels)*100:.1f}%)

Clustering Quality Assessment:
"""
    
    for method, metrics in metrics_results.items():
        sep_ratio = metrics['separation_ratio']
        quality = "Excellent" if sep_ratio > 1.5 else "Good" if sep_ratio > 1.0 else "Fair"
        
        report += f"""
{method}:
  Separation Ratio: {sep_ratio:.4f} ({quality})
  Quality Score: {metrics['quality_score']:.4f}
  Centroid Distance: {metrics['centroid_distance']:.4f}
  Avg Compactness: {metrics['avg_compactness']:.4f}
"""
    
    report += """
Interpretation:
  - Separation Ratio > 1.5: Excellent cluster separation (strong data quality)
  - Separation Ratio > 1.0: Good cluster separation
  - Separation Ratio < 1.0: Poor cluster separation (weak data quality)
  
  The ability of the LLM to separate correct from incorrect answers
  demonstrates the quality and clarity of the synthetic dataset.
  Higher separation indicates that the model can reliably distinguish
  between correct and incorrect answers in embedding space.
"""
    
    report_path = f"{output_dir}/report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")
    
    return report


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize embedding clusters for synthetic legal Q&A dataset"
    )
    
    parser.add_argument(
        '--input', '-i',
        default='./qa_combined_clean_final.jsonl',
        help='Path to JSONL dataset file'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./clustering_visualization',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--model_path', '-m',
        default='/fast_storage/siddig/driving-license/LLM_evaluator/fine_tuned_model/best_model_nlpaueb_legal-bert-small-uncased/',
        help='HuggingFace model name or path for embeddings'
    )

    parser.add_argument(
        '--model',
        default='nlpaueb/legal-bert-small-uncased',
        help='HuggingFace model name or path for embeddings'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples to process'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['tsne', 'umap', 'pca'],
        help='Dimensionality reduction methods'
    )
    
    parser.add_argument(
        '--concatenate',
        action='store_true',
        default=True,
        help='Concatenate Q&A instead of using answers only'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device: cuda or cpu'
    )
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    
    # Run pipeline
    results = visualize_embeddings_pipeline(
        jsonl_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        model_path=args.model_path,
        max_samples=args.max_samples,
        methods=args.methods,
        use_qa_concatenation=args.concatenate
    )