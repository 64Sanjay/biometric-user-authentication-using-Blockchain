# src/evaluation/full_evaluation.py

"""
Full Dataset Evaluation
Evaluates the system with all 500 users from CASIA-FaceV5
and corresponding hand images from 11k Hands dataset.
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.evaluation.improved_evaluation import ImprovedEvaluator
from src.evaluation.visualization import PerformanceVisualizer


class FullDatasetEvaluator:
    """
    Full dataset evaluation with all 500 users
    """
    
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.output_dir = os.path.join(Config.LOG_DIR, 'full_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def count_available_users(self):
        """Count available users in the dataset"""
        face_dir = Config.FACE_DATA_DIR
        users = [f for f in os.listdir(face_dir) 
                 if os.path.isdir(os.path.join(face_dir, f))]
        return len(users)
    
    def run_full_evaluation(self, num_users=None, num_imposter_tests=500):
        """
        Run evaluation with full dataset
        
        Args:
            num_users: Number of users (None = all available)
            num_imposter_tests: Number of imposter tests
        
        Returns:
            Dictionary with all results
        """
        print("=" * 70)
        print("FULL DATASET EVALUATION")
        print("=" * 70)
        
        # Count available users
        available_users = self.count_available_users()
        print(f"\nAvailable users in dataset: {available_users}")
        
        if num_users is None:
            num_users = available_users
        
        print(f"Users to evaluate: {num_users}")
        print(f"Imposter tests: {num_imposter_tests}")
        print(f"Threshold: {self.threshold}")
        
        start_time = time.time()
        
        # Run evaluation
        evaluator = ImprovedEvaluator(threshold=self.threshold)
        metrics = evaluator.run_evaluation(
            num_users=num_users,
            num_imposter_tests=num_imposter_tests
        )
        
        elapsed_time = time.time() - start_time
        
        # Get scores for visualization
        genuine_scores = evaluator.genuine_scores
        imposter_scores = evaluator.imposter_scores
        
        # Generate visualizations
        print("\n" + "-" * 70)
        print("Generating visualizations...")
        
        viz_output_dir = os.path.join(self.output_dir, 'visualizations')
        visualizer = PerformanceVisualizer(output_dir=viz_output_dir)
        viz_outputs = visualizer.generate_all_plots(
            genuine_scores,
            imposter_scores,
            metrics,
            threshold=self.threshold
        )
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_users': num_users,
                'num_imposter_tests': num_imposter_tests,
                'threshold': self.threshold,
                'genuine_tests': len(genuine_scores),
                'imposter_tests_actual': len(imposter_scores)
            },
            'metrics': {k: float(v) for k, v in metrics.items()},
            'paper_comparison': {
                'TAR': {'ours': metrics.get('TAR', 0), 'paper': 0.96},
                'FAR': {'ours': metrics.get('FAR', 0), 'paper': 0.08},
                'TRR': {'ours': metrics.get('TRR', 0), 'paper': 0.92},
                'FRR': {'ours': metrics.get('FRR', 0), 'paper': 0.08},
                'EER': {'ours': metrics.get('EER', 0), 'paper': 0.08},
                'Accuracy': {'ours': metrics.get('Accuracy', 0), 'paper': 0.998}
            },
            'score_statistics': {
                'genuine': {
                    'count': len(genuine_scores),
                    'mean': float(np.mean(genuine_scores)),
                    'std': float(np.std(genuine_scores)),
                    'min': float(np.min(genuine_scores)),
                    'max': float(np.max(genuine_scores))
                },
                'imposter': {
                    'count': len(imposter_scores),
                    'mean': float(np.mean(imposter_scores)),
                    'std': float(np.std(imposter_scores)),
                    'min': float(np.min(imposter_scores)),
                    'max': float(np.max(imposter_scores))
                }
            },
            'elapsed_time_seconds': elapsed_time,
            'visualization_files': list(viz_outputs.values())
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, 'full_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save scores for future use
        scores_path = os.path.join(self.output_dir, 'scores.npz')
        np.savez(scores_path, 
                 genuine_scores=np.array(genuine_scores),
                 imposter_scores=np.array(imposter_scores))
        
        # Print summary
        print("\n" + "=" * 70)
        print("FULL EVALUATION COMPLETE")
        print("=" * 70)
        
        print(f"\nTime elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        print(f"\n{'Metric':<25} {'Ours':<12} {'Paper':<12} {'Status':<10}")
        print("-" * 60)
        
        for metric in ['TAR', 'FAR', 'TRR', 'FRR', 'EER', 'Accuracy']:
            ours = metrics.get(metric, 0)
            paper = results['paper_comparison'][metric]['paper']
            
            if metric in ['FAR', 'FRR', 'EER']:
                status = '✓' if ours <= paper * 1.2 else '✗'
            else:
                status = '✓' if ours >= paper * 0.9 else '✗'
            
            print(f"{metric:<25} {ours:<12.4f} {paper:<12.4f} {status:<10}")
        
        print(f"\nResults saved to: {results_path}")
        print(f"Scores saved to: {scores_path}")
        print(f"Visualizations saved to: {viz_output_dir}")
        
        return results


def main():
    """Run full dataset evaluation"""
    
    print("\n" + "=" * 70)
    print("STARTING FULL DATASET EVALUATION")
    print("This may take several minutes depending on your GPU...")
    print("=" * 70)
    
    # Create evaluator
    evaluator = FullDatasetEvaluator(threshold=0.9)
    
    # Run with all 500 users
    # Adjust num_imposter_tests based on your needs
    results = evaluator.run_full_evaluation(
        num_users=100,  # Start with 100, increase to 500 if you want
        num_imposter_tests=200
    )
    
    return results


if __name__ == "__main__":
    main()