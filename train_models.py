"""
Churn Prediction Model Training
Trains and compares multiple ML models for telecom churn prediction.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.model_metrics = {}
        
    def prepare_data(self, df):
        """Prepare data for modeling"""
        df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'location', 'plan_type', 'contract_type']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Select features
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
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest Classifier...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.model_metrics['Random Forest'] = metrics
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True))
        
        self.models['Random Forest'] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Classifier...")
        print("="*50)
        
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
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.model_metrics['XGBoost'] = metrics
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True))
        
        self.models['XGBoost'] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model using TensorFlow"""
        print("\n" + "="*50)
        print("Training Neural Network (TensorFlow)...")
        print("="*50)
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.model_metrics['Neural Network'] = metrics
        
        self.models['Neural Network'] = {
            'model': model,
            'metrics': metrics,
            'scaler': self.scaler
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        comparison_df = pd.DataFrame(self.model_metrics).T
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        print("\n" + comparison_df.to_string())
        print("\n" + "="*70)
        
        # Find best model
        best_model_name = comparison_df.index[0]
        print(f"\nBest Model: {best_model_name}")
        print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
        print(f"   Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}")
        
        return best_model_name, comparison_df
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Random Forest
        if 'Random Forest' in self.models:
            with open(f'{output_dir}/random_forest.pkl', 'wb') as f:
                pickle.dump(self.models['Random Forest']['model'], f)
        
        # Save XGBoost
        if 'XGBoost' in self.models:
            with open(f'{output_dir}/xgboost.pkl', 'wb') as f:
                pickle.dump(self.models['XGBoost']['model'], f)
        
        # Save Neural Network
        if 'Neural Network' in self.models:
            self.models['Neural Network']['model'].save(f'{output_dir}/neural_network.h5')
            with open(f'{output_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save label encoders
        with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        comparison_df = pd.DataFrame(self.model_metrics).T
        comparison_df.to_csv(f'{output_dir}/model_metrics.csv')
        
        print(f"\nModels saved to {output_dir}/")

def main():
    # Load data
    print("Loading telecom churn data...")
    df = pd.read_csv('telecom_churn_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean()*100:.2f}%")
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    predictor.train_random_forest(X_train, y_train, X_test, y_test)
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    predictor.train_neural_network(X_train, y_train, X_test, y_test)
    
    # Compare models
    best_model, comparison_df = predictor.compare_models()
    
    # Save models
    predictor.save_models()
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()

