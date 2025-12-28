# src/evaluation/performance_test.py

"""
Performance Evaluation Module
Calculates metrics as described in Section 5 of the paper:
- True Acceptance Rate (TAR)
- False Acceptance Rate (FAR)
- True Rejection Rate (TRR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER)
"""

import os
import sys
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.preprocessing.face_preprocessing import FacePreprocessor
from src.preprocessing.hand_preprocessing import HandPreprocessor
from src.feature_extraction.face_features import FaceFeatureExtractor
from src.feature_extraction.hand_features import HandFeatureExtractor
from src.fuzzy_vault.vault_encoder import ImprovedFuzzyVaultEncoder
from src.fuzzy_vault.vault_decoder import ImprovedFuzzyVaultDecoder


class PerformanceEvaluator:
    """
    Evaluates system performance using metrics from the paper
    """
    
    def __init__(self):
        print("Initializing Performance Evaluator...")
        
        self.face_preprocessor = FacePreprocessor()
        self.hand_preprocessor = HandPreprocessor()
        self.face_extractor = FaceFeatureExtractor()
        self.hand_extractor = HandFeatureExtractor()
        self.vault_encoder = ImprovedFuzzyVaultEncoder()
        self.vault_decoder = ImprovedFuzzyVaultDecoder()
        
        # Results storage
        self.genuine_scores = []
        self.imposter_scores = []
        
        print("Evaluator initialized.")
    
    def get_user_data(self, num_users=10):
        """
        Get face and hand images for multiple users
        Maps hand images to users (since hand dataset doesn't have user folders)
        """
        users = []
        
        # Get face user folders
        face_dir = Config.FACE_DATA_DIR
        face_folders = sorted([
            f for f in os.listdir(face_dir)
            if os.path.isdir(os.path.join(face_dir, f))
        ])[:num_users]
        
        # Get hand images
        hand_dir = Config.HAND_DATA_DIR
        hand_images = sorted([
            f for f in os.listdir(hand_dir)
            if f.lower().endswith(Config.HAND_IMAGE_FORMATS)
        ])
        
        # Assign hand images to users (multiple images per user)
        hands_per_user = 5  # Assign 5 hand images per user
        
        for idx, face_folder in enumerate(face_folders):
            face_user_dir = os.path.join(face_dir, face_folder)
            face_images = sorted([
                os.path.join(face_user_dir, f)
                for f in os.listdir(face_user_dir)
                if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
            ])
            
            # Assign hand images for this user
            hand_start = idx * hands_per_user
            hand_end = hand_start + hands_per_user
            user_hand_images = [
                os.path.join(hand_dir, hand_images[i % len(hand_images)])
                for i in range(hand_start, hand_end)
            ]
            
            users.append({
                'user_id': face_folder,
                'face_images': face_images,
                'hand_images': user_hand_images
            })
        
        return users
    
    def extract_features(self, face_path, hand_path):
        """Extract features from face and hand images"""
        face_features = self.face_extractor.extract_features(face_path)
        hand_features = self.hand_extractor.extract_features(hand_path)
        return face_features, hand_features
    
    def run_genuine_tests(self, users, num_tests_per_user=3):
        """
        Test genuine users (same person, different images)
        """
        print("\n" + "=" * 60)
        print("GENUINE USER TESTS")
        print("=" * 60)
        
        genuine_results = []
        
        for user in users:
            user_id = user['user_id']
            face_images = user['face_images']
            hand_images = user['hand_images']
            
            if len(face_images) < 2 or len(hand_images) < 2:
                continue
            
            # Enroll with first images
            face_enroll = face_images[0]
            hand_enroll = hand_images[0]
            
            try:
                face_feat_enroll, hand_feat_enroll = self.extract_features(
                    face_enroll, hand_enroll
                )
                
                # Create vault
                vault, token, _, _ = self.vault_encoder.encode(
                    hand_feat_enroll, face_feat_enroll, user_id
                )
                
                # Test with different images of same user
                for test_idx in range(1, min(num_tests_per_user + 1, len(face_images))):
                    face_verify = face_images[test_idx]
                    hand_verify = hand_images[min(test_idx, len(hand_images) - 1)]
                    
                    face_feat_verify, hand_feat_verify = self.extract_features(
                        face_verify, hand_verify
                    )
                    
                    # Verify
                    is_genuine, score, message = self.vault_decoder.verify(
                        vault.tolist(), token, hand_feat_verify, face_feat_verify
                    )
                    
                    genuine_results.append({
                        'user_id': user_id,
                        'is_genuine': is_genuine,
                        'score': score,
                        'expected': True
                    })
                    
                    self.genuine_scores.append(score)
                    
            except Exception as e:
                print(f"  Error testing user {user_id}: {e}")
        
        # Calculate statistics
        if genuine_results:
            accepted = sum(1 for r in genuine_results if r['is_genuine'])
            total = len(genuine_results)
            tar = accepted / total
            frr = 1 - tar
            
            print(f"\nGenuine Tests: {total}")
            print(f"Accepted: {accepted}")
            print(f"TAR (True Acceptance Rate): {tar:.4f}")
            print(f"FRR (False Rejection Rate): {frr:.4f}")
            
            if self.genuine_scores:
                print(f"Score Mean: {np.mean(self.genuine_scores):.4f}")
                print(f"Score Std: {np.std(self.genuine_scores):.4f}")
        
        return genuine_results
    
    def run_imposter_tests(self, users, num_tests=20):
        """
        Test imposters (different person trying to authenticate)
        Uses DIFFERENT hand images for imposters
        """
        print("\n" + "=" * 60)
        print("IMPOSTER TESTS")
        print("=" * 60)
        
        imposter_results = []
        
        if len(users) < 2:
            print("Need at least 2 users for imposter tests")
            return imposter_results
        
        # For each enrolled user, test with other users' biometrics
        tests_done = 0
        
        for enroll_idx, enroll_user in enumerate(users):
            if tests_done >= num_tests:
                break
            
            user_id = enroll_user['user_id']
            face_enroll = enroll_user['face_images'][0]
            hand_enroll = enroll_user['hand_images'][0]
            
            try:
                # Enroll
                face_feat_enroll, hand_feat_enroll = self.extract_features(
                    face_enroll, hand_enroll
                )
                
                vault, token, _, _ = self.vault_encoder.encode(
                    hand_feat_enroll, face_feat_enroll, user_id
                )
                
                # Test with OTHER users' biometrics (both face AND hand)
                for imposter_idx, imposter_user in enumerate(users):
                    if imposter_idx == enroll_idx:
                        continue
                    
                    if tests_done >= num_tests:
                        break
                    
                    # Use imposter's face AND hand
                    face_imposter = imposter_user['face_images'][0]
                    hand_imposter = imposter_user['hand_images'][0]
                    
                    face_feat_imposter, hand_feat_imposter = self.extract_features(
                        face_imposter, hand_imposter
                    )
                    
                    # Verify (should reject - different hand will fail decryption)
                    is_genuine, score, message = self.vault_decoder.verify(
                        vault.tolist(), token, hand_feat_imposter, face_feat_imposter
                    )
                    
                    imposter_results.append({
                        'enrolled_user': user_id,
                        'imposter_user': imposter_user['user_id'],
                        'is_genuine': is_genuine,
                        'score': score,
                        'expected': False
                    })
                    
                    self.imposter_scores.append(score)
                    tests_done += 1
                    
            except Exception as e:
                print(f"  Error testing: {e}")
        
        # Calculate statistics
        if imposter_results:
            rejected = sum(1 for r in imposter_results if not r['is_genuine'])
            total = len(imposter_results)
            trr = rejected / total
            far = 1 - trr
            
            print(f"\nImposter Tests: {total}")
            print(f"Rejected: {rejected}")
            print(f"TRR (True Rejection Rate): {trr:.4f}")
            print(f"FAR (False Acceptance Rate): {far:.4f}")
            
            if self.imposter_scores:
                print(f"Score Mean: {np.mean(self.imposter_scores):.4f}")
                print(f"Score Std: {np.std(self.imposter_scores):.4f}")
        
        return imposter_results
    
    def calculate_eer(self):
        """
        Calculate Equal Error Rate (EER)
        EER is where FAR = FRR
        """
        if not self.genuine_scores or not self.imposter_scores:
            return None
        
        # Try different thresholds
        thresholds = np.linspace(0, 1, 100)
        
        best_eer = 1.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            # FAR: proportion of imposters accepted
            far = np.mean([s >= threshold for s in self.imposter_scores])
            
            # FRR: proportion of genuine rejected
            frr = np.mean([s < threshold for s in self.genuine_scores])
            
            # EER is where FAR ≈ FRR
            diff = abs(far - frr)
            if diff < abs(best_eer - 0):
                eer = (far + frr) / 2
                if eer < best_eer:
                    best_eer = eer
                    best_threshold = threshold
        
        return best_eer, best_threshold
    
    def run_full_evaluation(self, num_users=10, genuine_tests_per_user=3, imposter_tests=20):
        """
        Run complete performance evaluation
        """
        print("\n" + "=" * 70)
        print("PERFORMANCE EVALUATION")
        print("Based on Section 5 of the paper")
        print("=" * 70)
        
        # Get user data
        print("\nLoading user data...")
        users = self.get_user_data(num_users)
        print(f"Loaded {len(users)} users")
        
        # Run tests
        genuine_results = self.run_genuine_tests(users, genuine_tests_per_user)
        imposter_results = self.run_imposter_tests(users, imposter_tests)
        
        # Calculate metrics
        print("\n" + "=" * 60)
        print("FINAL METRICS")
        print("=" * 60)
        
        metrics = {}
        
        # TAR and FRR
        if genuine_results:
            accepted = sum(1 for r in genuine_results if r['is_genuine'])
            total = len(genuine_results)
            metrics['TAR'] = accepted / total
            metrics['FRR'] = 1 - metrics['TAR']
        
        # TRR and FAR
        if imposter_results:
            rejected = sum(1 for r in imposter_results if not r['is_genuine'])
            total = len(imposter_results)
            metrics['TRR'] = rejected / total
            metrics['FAR'] = 1 - metrics['TRR']
        
        # EER
        eer_result = self.calculate_eer()
        if eer_result:
            metrics['EER'], metrics['EER_threshold'] = eer_result
        
        # Accuracy
        if genuine_results and imposter_results:
            correct = (sum(1 for r in genuine_results if r['is_genuine']) + 
                      sum(1 for r in imposter_results if not r['is_genuine']))
            total = len(genuine_results) + len(imposter_results)
            metrics['Accuracy'] = correct / total
        
        # Print results
        print(f"\n{'Metric':<30} {'Value':<15} {'Paper Target':<15}")
        print("-" * 60)
        print(f"{'TAR (True Acceptance Rate)':<30} {metrics.get('TAR', 'N/A'):<15.4f} {'0.96':<15}")
        print(f"{'FAR (False Acceptance Rate)':<30} {metrics.get('FAR', 'N/A'):<15.4f} {'0.08':<15}")
        print(f"{'TRR (True Rejection Rate)':<30} {metrics.get('TRR', 'N/A'):<15.4f} {'0.92':<15}")
        print(f"{'FRR (False Rejection Rate)':<30} {metrics.get('FRR', 'N/A'):<15.4f} {'0.08':<15}")
        print(f"{'EER (Equal Error Rate)':<30} {metrics.get('EER', 'N/A'):<15.4f} {'0.08':<15}")
        print(f"{'Accuracy':<30} {metrics.get('Accuracy', 'N/A'):<15.4f} {'0.998':<15}")
        
        print("\n" + "=" * 70)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_users': num_users,
            'metrics': metrics,
            'genuine_scores': self.genuine_scores,
            'imposter_scores': self.imposter_scores
        }
        
        results_path = os.path.join(Config.LOG_DIR, 'performance_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Results saved to: {results_path}")
        
        return metrics


def main():
    """Run performance evaluation"""
    
    # Create logs directory
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    evaluator = PerformanceEvaluator()
    
    # Run evaluation with 10 users
    metrics = evaluator.run_full_evaluation(
        num_users=10,
        genuine_tests_per_user=3,
        imposter_tests=20
    )
    
    return metrics


if __name__ == "__main__":
    main()