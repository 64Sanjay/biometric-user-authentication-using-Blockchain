# src/evaluation/visualization.py

"""
Visualization Module for Performance Metrics
Creates publication-quality plots:
- ROC Curve (Receiver Operating Characteristic)
- DET Curve (Detection Error Trade-off)
- Score Distribution
- Confusion Matrix
- Performance Comparison Bar Charts
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import interpolate
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class PerformanceVisualizer:
    """
    Creates publication-quality visualizations for biometric system performance
    """
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(Config.LOG_DIR, 'visualizations')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-quality plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_roc_curve(self, genuine_scores, imposter_scores, title="ROC Curve"):
        """
        Plot Receiver Operating Characteristic (ROC) Curve
        
        ROC shows TAR (True Acceptance Rate) vs FAR (False Acceptance Rate)
        As shown in Figure 7 of the paper.
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
            title: Plot title
        
        Returns:
            Path to saved figure
        """
        # Calculate ROC points
        thresholds = np.linspace(0, 1, 1000)
        
        tar_values = []  # True Acceptance Rate (Sensitivity)
        far_values = []  # False Acceptance Rate (1 - Specificity)
        
        for threshold in thresholds:
            # TAR: proportion of genuine users accepted
            tar = np.mean([s >= threshold for s in genuine_scores])
            # FAR: proportion of imposters accepted
            far = np.mean([s >= threshold for s in imposter_scores])
            
            tar_values.append(tar)
            far_values.append(far)
        
        tar_values = np.array(tar_values)
        far_values = np.array(far_values)
        
        # Calculate AUC (Area Under Curve)
        # Sort by FAR for proper integration
        sorted_indices = np.argsort(far_values)
        far_sorted = far_values[sorted_indices]
        tar_sorted = tar_values[sorted_indices]
        auc = np.trapz(tar_sorted, far_sorted)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(far_values, tar_values, 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {auc:.4f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Mark EER point
        eer_idx = np.argmin(np.abs(tar_values - (1 - far_values)))
        eer_far = far_values[eer_idx]
        eer_tar = tar_values[eer_idx]
        ax.plot(eer_far, eer_tar, 'go', markersize=10, label=f'EER Point ({eer_far:.4f})')
        
        # Labels and title
        ax.set_xlabel('False Acceptance Rate (FAR)')
        ax.set_ylabel('True Acceptance Rate (TAR)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = f'AUC: {auc:.4f}\nEER: {eer_far:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.65, 0.15, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Save
        output_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"ROC Curve saved to: {output_path}")
        
        return output_path, auc
    
    def plot_det_curve(self, genuine_scores, imposter_scores, title="DET Curve"):
        """
        Plot Detection Error Trade-off (DET) Curve
        
        DET shows FRR (False Rejection Rate) vs FAR (False Acceptance Rate)
        As shown in Figure 6 of the paper.
        Uses logarithmic scale for better visualization.
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
            title: Plot title
        
        Returns:
            Path to saved figure
        """
        # Calculate DET points
        thresholds = np.linspace(0, 1, 1000)
        
        frr_values = []  # False Rejection Rate (FNMR)
        far_values = []  # False Acceptance Rate (FMR)
        
        for threshold in thresholds:
            # FRR: proportion of genuine users rejected
            frr = np.mean([s < threshold for s in genuine_scores])
            # FAR: proportion of imposters accepted
            far = np.mean([s >= threshold for s in imposter_scores])
            
            frr_values.append(frr)
            far_values.append(far)
        
        frr_values = np.array(frr_values)
        far_values = np.array(far_values)
        
        # Find EER
        eer_idx = np.argmin(np.abs(frr_values - far_values))
        eer = (frr_values[eer_idx] + far_values[eer_idx]) / 2
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot DET curve
        # Add small epsilon to avoid log(0)
        eps = 1e-4
        far_plot = np.clip(far_values, eps, 1)
        frr_plot = np.clip(frr_values, eps, 1)
        
        ax.loglog(far_plot, frr_plot, 'b-', linewidth=2, label='DET Curve')
        
        # Plot EER line (where FAR = FRR)
        ax.loglog([eps, 1], [eps, 1], 'r--', linewidth=1, label='EER Line (FAR=FRR)')
        
        # Mark EER point
        ax.plot(far_values[eer_idx], frr_values[eer_idx], 'go', markersize=10,
                label=f'EER = {eer:.4f}')
        
        # Labels and title
        ax.set_xlabel('False Acceptance Rate (FAR)')
        ax.set_ylabel('False Rejection Rate (FRR)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        
        # Set axis limits
        ax.set_xlim([0.001, 1])
        ax.set_ylim([0.001, 1])
        
        # Save
        output_path = os.path.join(self.output_dir, 'det_curve.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"DET Curve saved to: {output_path}")
        
        return output_path, eer
    
    def plot_score_distribution(self, genuine_scores, imposter_scores, 
                                 threshold=0.9, title="Score Distribution"):
        """
        Plot score distributions for genuine and imposter users
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
            threshold: Decision threshold
            title: Plot title
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        
        ax.hist(genuine_scores, bins=bins, alpha=0.7, color='green', 
                edgecolor='darkgreen', label=f'Genuine (n={len(genuine_scores)})', density=True)
        ax.hist(imposter_scores, bins=bins, alpha=0.7, color='red', 
                edgecolor='darkred', label=f'Imposter (n={len(imposter_scores)})', density=True)
        
        # Plot threshold line
        ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold}')
        
        # Calculate overlap region
        genuine_below = np.mean([s < threshold for s in genuine_scores])
        imposter_above = np.mean([s >= threshold for s in imposter_scores])
        
        # Labels
        ax.set_xlabel('Matching Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (f'Genuine Mean: {np.mean(genuine_scores):.4f}\n'
                     f'Imposter Mean: {np.mean(imposter_scores):.4f}\n'
                     f'FRR: {genuine_below:.4f}\n'
                     f'FAR: {imposter_above:.4f}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save
        output_path = os.path.join(self.output_dir, 'score_distribution.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Score Distribution saved to: {output_path}")
        
        return output_path
    
    def plot_confusion_matrix(self, genuine_scores, imposter_scores, threshold=0.9):
        """
        Plot confusion matrix
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
            threshold: Decision threshold
        
        Returns:
            Path to saved figure
        """
        # Calculate confusion matrix values
        tp = sum(1 for s in genuine_scores if s >= threshold)  # True Positive
        fn = sum(1 for s in genuine_scores if s < threshold)   # False Negative
        fp = sum(1 for s in imposter_scores if s >= threshold) # False Positive
        tn = sum(1 for s in imposter_scores if s < threshold)  # True Negative
        
        confusion_matrix = np.array([[tp, fn], [fp, tn]])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
        
        # Labels
        labels = ['Genuine', 'Imposter']
        predictions = ['Accept', 'Reject']
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(predictions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (Threshold = {threshold})')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=16)
        
        # Add metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = (f'Accuracy: {accuracy:.4f}\n'
                       f'Precision: {precision:.4f}\n'
                       f'Recall: {recall:.4f}\n'
                       f'F1-Score: {f1:.4f}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Confusion Matrix saved to: {output_path}")
        
        return output_path
    
    def plot_metrics_comparison(self, our_metrics, paper_metrics=None):
        """
        Plot bar chart comparing our metrics with paper's results
        
        Args:
            our_metrics: Dictionary of our metrics
            paper_metrics: Dictionary of paper's metrics (optional)
        
        Returns:
            Path to saved figure
        """
        if paper_metrics is None:
            paper_metrics = {
                'TAR': 0.96,
                'FAR': 0.08,
                'TRR': 0.92,
                'FRR': 0.08,
                'EER': 0.08,
                'Accuracy': 0.998
            }
        
        metrics = list(our_metrics.keys())
        our_values = [our_metrics.get(m, 0) for m in metrics]
        paper_values = [paper_metrics.get(m, 0) for m in metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, our_values, width, label='Our Implementation', 
                       color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, paper_values, width, label='Paper Results', 
                       color='orange', edgecolor='black')
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        add_labels(bars1)
        add_labels(bars2)
        
        # Labels
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Comparison: Our Implementation vs Paper')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Metrics Comparison saved to: {output_path}")
        
        return output_path
    
    def plot_far_frr_vs_threshold(self, genuine_scores, imposter_scores):
        """
        Plot FAR and FRR vs Threshold
        
        Shows how FAR and FRR change with different thresholds.
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
        
        Returns:
            Path to saved figure
        """
        thresholds = np.linspace(0, 1, 100)
        
        far_values = []
        frr_values = []
        
        for t in thresholds:
            far = np.mean([s >= t for s in imposter_scores])
            frr = np.mean([s < t for s in genuine_scores])
            far_values.append(far)
            frr_values.append(frr)
        
        far_values = np.array(far_values)
        frr_values = np.array(frr_values)
        
        # Find EER
        eer_idx = np.argmin(np.abs(far_values - frr_values))
        eer_threshold = thresholds[eer_idx]
        eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, far_values, 'r-', linewidth=2, label='FAR')
        ax.plot(thresholds, frr_values, 'b-', linewidth=2, label='FRR')
        
        # Mark EER point
        ax.axvline(x=eer_threshold, color='green', linestyle='--', 
                   label=f'EER Threshold = {eer_threshold:.4f}')
        ax.plot(eer_threshold, eer, 'go', markersize=10, label=f'EER = {eer:.4f}')
        
        # Labels
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Error Rate')
        ax.set_title('FAR and FRR vs Threshold')
        ax.legend(loc='center right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = os.path.join(self.output_dir, 'far_frr_threshold.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"FAR/FRR vs Threshold saved to: {output_path}")
        
        return output_path, eer_threshold, eer
    
    def generate_all_plots(self, genuine_scores, imposter_scores, metrics, threshold=0.9):
        """
        Generate all visualization plots
        
        Args:
            genuine_scores: List of scores for genuine users
            imposter_scores: List of scores for imposters
            metrics: Dictionary of performance metrics
            threshold: Decision threshold
        
        Returns:
            Dictionary of output paths
        """
        print("\n" + "=" * 60)
        print("GENERATING ALL VISUALIZATIONS")
        print("=" * 60)
        
        outputs = {}
        
        # 1. ROC Curve
        print("\n[1/6] Generating ROC Curve...")
        outputs['roc'], auc = self.plot_roc_curve(genuine_scores, imposter_scores)
        
        # 2. DET Curve
        print("[2/6] Generating DET Curve...")
        outputs['det'], eer = self.plot_det_curve(genuine_scores, imposter_scores)
        
        # 3. Score Distribution
        print("[3/6] Generating Score Distribution...")
        outputs['distribution'] = self.plot_score_distribution(
            genuine_scores, imposter_scores, threshold
        )
        
        # 4. Confusion Matrix
        print("[4/6] Generating Confusion Matrix...")
        outputs['confusion'] = self.plot_confusion_matrix(
            genuine_scores, imposter_scores, threshold
        )
        
        # 5. Metrics Comparison
        print("[5/6] Generating Metrics Comparison...")
        outputs['comparison'] = self.plot_metrics_comparison(metrics)
        
        # 6. FAR/FRR vs Threshold
        print("[6/6] Generating FAR/FRR vs Threshold...")
        outputs['threshold'], eer_t, eer_v = self.plot_far_frr_vs_threshold(
            genuine_scores, imposter_scores
        )
        
        print("\n" + "=" * 60)
        print("ALL VISUALIZATIONS GENERATED")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nFiles created:")
        for name, path in outputs.items():
            print(f"  - {os.path.basename(path)}")
        
        return outputs


def main():
    """Generate visualizations from saved evaluation results"""
    
    # Load saved evaluation results
    results_path = os.path.join(Config.LOG_DIR, 'improved_evaluation.json')
    
    if not os.path.exists(results_path):
        print("No evaluation results found. Running evaluation first...")
        
        # Run evaluation
        from src.evaluation.improved_evaluation import ImprovedEvaluator
        evaluator = ImprovedEvaluator(threshold=0.9)
        metrics = evaluator.run_evaluation(num_users=50, num_imposter_tests=100)
        genuine_scores = evaluator.genuine_scores
        imposter_scores = evaluator.imposter_scores
    else:
        # Load from file
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        metrics = results.get('metrics', {})
        
        # We need to re-run to get scores (they're not saved in JSON)
        print("Re-running evaluation to get score distributions...")
        from src.evaluation.improved_evaluation import ImprovedEvaluator
        evaluator = ImprovedEvaluator(threshold=0.9)
        metrics = evaluator.run_evaluation(num_users=50, num_imposter_tests=100)
        genuine_scores = evaluator.genuine_scores
        imposter_scores = evaluator.imposter_scores
    
    # Generate visualizations
    visualizer = PerformanceVisualizer()
    outputs = visualizer.generate_all_plots(
        genuine_scores, 
        imposter_scores, 
        metrics, 
        threshold=0.9
    )
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()