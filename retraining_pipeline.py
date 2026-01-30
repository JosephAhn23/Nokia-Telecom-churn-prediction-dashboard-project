"""
MLOps Retraining Pipeline with Model Drift Detection
Automated model retraining pipeline for production MLOps.
Detects model drift and triggers retraining when needed.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

class ModelDriftDetector:
    """Detects model drift by comparing production data to training data"""
    
    def __init__(self, reference_data_path='telecom_churn_data.csv'):
        """Initialize with reference (training) data"""
        self.reference_data = pd.read_csv(reference_data_path)
        self.reference_stats = self._calculate_statistics(self.reference_data)
        
    def _calculate_statistics(self, data: pd.DataFrame) -> dict:
        """Calculate statistical properties of the dataset"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            if col != 'churn':  # Exclude target
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median()
                }
        
        # Categorical distributions
        categorical_cols = ['plan_type', 'contract_type', 'location', 'gender']
        for col in categorical_cols:
            if col in data.columns:
                stats[col] = {
                    'distribution': data[col].value_counts(normalize=True).to_dict()
                }
        
        # Target distribution
        stats['churn_rate'] = data['churn'].mean()
        
        return stats
    
    def detect_drift(self, production_data: pd.DataFrame, threshold=0.15) -> dict:
        """
        Detect data drift between reference and production data
        
        Args:
            production_data: New production data
            threshold: Drift threshold (0-1)
        
        Returns:
            Dictionary with drift detection results
        """
        production_stats = self._calculate_statistics(production_data)
        drift_detected = False
        drift_details = []
        
        # Check numeric feature drift (using Kolmogorov-Smirnov test approximation)
        numeric_cols = [col for col in self.reference_stats.keys() 
                       if isinstance(self.reference_stats[col], dict) 
                       and 'mean' in self.reference_stats[col]]
        
        for col in numeric_cols:
            if col in production_stats:
                ref_mean = self.reference_stats[col]['mean']
                prod_mean = production_stats[col]['mean']
                ref_std = self.reference_stats[col]['std']
                
                # Calculate drift score (normalized difference)
                if ref_std > 0:
                    drift_score = abs(prod_mean - ref_mean) / ref_std
                    
                    if drift_score > threshold:
                        drift_detected = True
                        drift_details.append({
                            'feature': col,
                            'drift_score': drift_score,
                            'reference_mean': ref_mean,
                            'production_mean': prod_mean,
                            'severity': 'High' if drift_score > threshold * 2 else 'Medium'
                        })
        
        # Check target distribution drift
        ref_churn = self.reference_stats['churn_rate']
        prod_churn = production_stats['churn_rate']
        churn_drift = abs(prod_churn - ref_churn)
        
        if churn_drift > threshold:
            drift_detected = True
            drift_details.append({
                'feature': 'churn_rate',
                'drift_score': churn_drift,
                'reference_mean': ref_churn,
                'production_mean': prod_churn,
                'severity': 'High'
            })
        
        # Check categorical distribution drift
        categorical_cols = ['plan_type', 'contract_type', 'location']
        for col in categorical_cols:
            if col in self.reference_stats and col in production_stats:
                ref_dist = self.reference_stats[col]['distribution']
                prod_dist = production_stats[col]['distribution']
                
                # Calculate total variation distance
                all_keys = set(ref_dist.keys()) | set(prod_dist.keys())
                tv_distance = sum(abs(ref_dist.get(k, 0) - prod_dist.get(k, 0)) 
                                 for k in all_keys) / 2
                
                if tv_distance > threshold:
                    drift_detected = True
                    drift_details.append({
                        'feature': col,
                        'drift_score': tv_distance,
                        'reference_distribution': ref_dist,
                        'production_distribution': prod_dist,
                        'severity': 'Medium'
                    })
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_model_performance_drift(self, model, X_test, y_test, 
                                     reference_accuracy: float, 
                                     threshold=0.05) -> dict:
        """
        Check if model performance has degraded
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            reference_accuracy: Reference accuracy from training
            threshold: Performance degradation threshold
        
        Returns:
            Performance drift detection results
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        current_accuracy = accuracy_score(y_test, y_pred)
        current_auc = roc_auc_score(y_test, y_pred_proba)
        
        accuracy_drop = reference_accuracy - current_accuracy
        
        performance_drift = {
            'reference_accuracy': reference_accuracy,
            'current_accuracy': current_accuracy,
            'accuracy_drop': accuracy_drop,
            'current_auc': current_auc,
            'performance_drift_detected': accuracy_drop > threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return performance_drift

class RetrainingPipeline:
    """Automated retraining pipeline with drift detection"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.drift_detector = ModelDriftDetector()
        
    def load_reference_model_metrics(self) -> dict:
        """Load reference model metrics"""
        try:
            metrics_df = pd.read_csv(self.model_dir / 'model_metrics.csv', index_col=0)
            return {
                'xgboost_accuracy': metrics_df.loc['XGBoost', 'accuracy'],
                'xgboost_roc_auc': metrics_df.loc['XGBoost', 'roc_auc']
            }
        except:
            return {
                'xgboost_accuracy': 0.92,  # Default reference
                'xgboost_roc_auc': 0.94
            }
    
    def prepare_data(self, df):
        """Prepare data for training (same as train_models.py)"""
        df = df.copy()
        
        categorical_cols = ['gender', 'location', 'plan_type', 'contract_type']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        feature_cols = [
            'age', 'gender', 'location', 'tenure_months', 'plan_type',
            'contract_type', 'monthly_charges', 'total_charges',
            'call_duration_minutes', 'data_usage_gb', 'sms_count',
            'roaming_usage', 'network_quality_score', 'dropped_calls',
            'payment_history_score', 'customer_service_calls',
            'tech_support_usage', 'complaints_count'
        ]
        
        X = df[feature_cols]
        y = df['churn']
        
        return X, y, label_encoders, feature_cols
    
    def train_new_model(self, training_data: pd.DataFrame) -> dict:
        """Train a new model on updated data"""
        print("Training new model on updated data...")
        
        X, y, label_encoders, feature_cols = self.prepare_data(training_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost (best performing model)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save model
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.model_dir / f'xgboost_{model_version}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save label encoders
        with open(self.model_dir / f'label_encoders_{model_version}.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Save feature names
        with open(self.model_dir / f'feature_names_{model_version}.json', 'w') as f:
            json.dump(feature_cols, f)
        
        return {
            'model_version': model_version,
            'model_path': str(model_path),
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'training_samples': len(training_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_models(self, old_metrics: dict, new_metrics: dict) -> dict:
        """Compare old and new model performance"""
        accuracy_improvement = new_metrics['accuracy'] - old_metrics['xgboost_accuracy']
        auc_improvement = new_metrics['roc_auc'] - old_metrics['xgboost_roc_auc']
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'auc_improvement': auc_improvement,
            'should_deploy': accuracy_improvement >= -0.02,  # Allow small degradation
            'improvement_significant': accuracy_improvement > 0.01
        }
    
    def run_retraining_check(self, production_data_path: str, 
                            retrain_threshold=0.15) -> dict:
        """
        Main retraining pipeline:
        1. Load production data
        2. Detect drift
        3. Check model performance
        4. Retrain if needed
        5. Compare models
        6. Recommend deployment
        """
        print("="*70)
        print("MLOps Retraining Pipeline")
        print("="*70)
        
        # Load production data
        production_data = pd.read_csv(production_data_path)
        print(f"\nLoaded {len(production_data)} production records")
        
        # Step 1: Detect data drift
        print("\nStep 1: Detecting data drift...")
        drift_results = self.drift_detector.detect_drift(
            production_data, threshold=retrain_threshold
        )
        
        if drift_results['drift_detected']:
            print("WARNING: Data drift detected!")
            for detail in drift_results['drift_details']:
                print(f"  - {detail['feature']}: drift_score={detail['drift_score']:.3f} ({detail['severity']})")
        else:
            print("No significant data drift detected")
        
        # Step 2: Load reference metrics
        reference_metrics = self.load_reference_model_metrics()
        print(f"\nReference model accuracy: {reference_metrics['xgboost_accuracy']:.4f}")
        
        # Step 3: Check if retraining is needed
        retrain_needed = (
            drift_results['drift_detected'] or
            len(drift_results['drift_details']) > 3  # Multiple features drifted
        )
        
        if not retrain_needed:
            print("\nNo retraining needed at this time")
            return {
                'retrain_needed': False,
                'reason': 'No significant drift detected',
                'drift_results': drift_results
            }
        
        # Step 4: Retrain model
        print("\nStep 2: Retraining model on updated data...")
        new_model_metrics = self.train_new_model(production_data)
        
        print(f"New model accuracy: {new_model_metrics['accuracy']:.4f}")
        print(f"New model ROC-AUC: {new_model_metrics['roc_auc']:.4f}")
        
        # Step 5: Compare models
        print("\nStep 3: Comparing models...")
        comparison = self.compare_models(reference_metrics, new_model_metrics)
        
        print(f"Accuracy change: {comparison['accuracy_improvement']:+.4f}")
        print(f"ROC-AUC change: {comparison['auc_improvement']:+.4f}")
        
        # Step 6: Deployment recommendation
        if comparison['should_deploy']:
            print("\nRECOMMENDATION: Deploy new model")
            if comparison['improvement_significant']:
                print("   Significant improvement detected - deploy immediately")
            else:
                print("   Model performance maintained - safe to deploy")
        else:
            print("\nWARNING: RECOMMENDATION: Keep current model")
            print("   New model shows degradation - investigate before deploying")
        
        return {
            'retrain_needed': True,
            'drift_results': drift_results,
            'new_model_metrics': new_model_metrics,
            'comparison': comparison,
            'deployment_recommendation': 'deploy' if comparison['should_deploy'] else 'keep_current',
            'model_version': new_model_metrics['model_version']
        }

# CI/CD Integration Example
def trigger_ci_cd_deployment(model_version: str):
    """
    Simulate CI/CD pipeline trigger for model deployment
    In production, this would:
    1. Build new Docker image with model version
    2. Run tests
    3. Deploy to staging
    4. Run validation
    5. Deploy to production (blue-green or canary)
    """
    print(f"\nTriggering CI/CD deployment for model {model_version}")
    print("  → Building Docker image...")
    print("  → Running tests...")
    print("  → Deploying to staging...")
    print("  → Validation passed")
    print("  → Deploying to production...")
    print("  Deployment complete")

# Example usage
if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    
    # Simulate production data (in reality, this would come from production database)
    # For demo, we'll use the same data but could add some drift
    print("Generating production data sample...")
    from data_generator import generate_telecom_data
    
    # Generate new data with slight distribution shift (simulating drift)
    production_data = generate_telecom_data(n_samples=5000, random_seed=999)
    production_data.to_csv('production_data_sample.csv', index=False)
    
    # Run retraining check
    results = pipeline.run_retraining_check('production_data_sample.csv')
    
    # If deployment recommended, trigger CI/CD
    if results.get('deployment_recommendation') == 'deploy':
        trigger_ci_cd_deployment(results['model_version'])

