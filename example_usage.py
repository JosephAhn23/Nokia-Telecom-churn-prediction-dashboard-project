"""
Example Usage: Programmatic Churn Prediction
Demonstrates how to use trained models for batch predictions
"""

import pandas as pd
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model_and_components(model_name='xgboost'):
    """Load a trained model and required components"""
    models = {}
    
    # Load the model
    if model_name == 'xgboost':
        with open('models/xgboost.pkl', 'rb') as f:
            models['model'] = pickle.load(f)
    elif model_name == 'random_forest':
        with open('models/random_forest.pkl', 'rb') as f:
            models['model'] = pickle.load(f)
    else:
        raise ValueError("Model must be 'xgboost' or 'random_forest'")
    
    # Load label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        models['label_encoders'] = pickle.load(f)
    
    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        models['feature_names'] = json.load(f)
    
    return models

def predict_churn_batch(customer_dataframe, model_name='xgboost'):
    """
    Predict churn for a batch of customers
    
    Args:
        customer_dataframe: DataFrame with customer features
        model_name: 'xgboost' or 'random_forest'
    
    Returns:
        DataFrame with churn predictions and probabilities
    """
    # Load model
    models = load_model_and_components(model_name)
    model = models['model']
    label_encoders = models['label_encoders']
    feature_names = models['feature_names']
    
    # Prepare data
    df = customer_dataframe.copy()
    
    # Encode categorical variables
    categorical_cols = ['gender', 'location', 'plan_type', 'contract_type']
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
    
    # Select features
    X = df[feature_names]
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    result_df = customer_dataframe.copy()
    result_df['churn_prediction'] = predictions
    result_df['churn_probability'] = probabilities
    result_df['risk_level'] = result_df['churn_probability'].apply(
        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low')
    )
    
    return result_df

def identify_at_risk_customers(customer_dataframe, threshold=0.5, model_name='xgboost'):
    """
    Identify customers at risk of churning
    
    Args:
        customer_dataframe: DataFrame with customer features
        threshold: Probability threshold for churn risk (default: 0.5)
        model_name: Model to use for prediction
    
    Returns:
        DataFrame with only at-risk customers, sorted by risk
    """
    predictions = predict_churn_batch(customer_dataframe, model_name)
    at_risk = predictions[predictions['churn_probability'] >= threshold]
    at_risk = at_risk.sort_values('churn_probability', ascending=False)
    
    return at_risk

def calculate_retention_roi(at_risk_customers, avg_customer_value=50, 
                            retention_cost_per_customer=20):
    """
    Calculate ROI of retention efforts
    
    Args:
        at_risk_customers: DataFrame of at-risk customers
        avg_customer_value: Average monthly revenue per customer
        retention_cost_per_customer: Cost of retention intervention
    
    Returns:
        Dictionary with ROI metrics
    """
    num_at_risk = len(at_risk_customers)
    total_retention_cost = num_at_risk * retention_cost_per_customer
    
    # Estimate prevented churns (assuming 50% success rate)
    success_rate = 0.5
    prevented_churns = int(num_at_risk * success_rate)
    
    # Calculate savings
    monthly_revenue_saved = prevented_churns * avg_customer_value
    annual_revenue_saved = monthly_revenue_saved * 12
    acquisition_cost_saved = prevented_churns * (avg_customer_value * 7)  # 7x monthly revenue
    total_savings = annual_revenue_saved + acquisition_cost_saved
    
    # ROI
    roi = ((total_savings - total_retention_cost) / total_retention_cost) * 100
    
    return {
        'at_risk_customers': num_at_risk,
        'retention_cost': total_retention_cost,
        'prevented_churns': prevented_churns,
        'monthly_revenue_saved': monthly_revenue_saved,
        'annual_revenue_saved': annual_revenue_saved,
        'total_savings': total_savings,
        'roi_percentage': roi
    }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Example: Batch Churn Prediction")
    print("="*70)
    
    # Load sample data
    try:
        df = pd.read_csv('telecom_churn_data.csv')
        print(f"\nLoaded {len(df)} customer records")
        
        # Take a sample for demonstration
        sample_size = 100
        sample_df = df.head(sample_size).copy()
        
        # Make predictions
        print(f"\nPredicting churn for {sample_size} customers...")
        predictions = predict_churn_batch(sample_df, model_name='xgboost')
        
        # Display results
        print("\n" + "="*70)
        print("Prediction Summary")
        print("="*70)
        print(f"Total customers analyzed: {len(predictions)}")
        print(f"High risk (prob > 0.7): {len(predictions[predictions['risk_level'] == 'High'])}")
        print(f"Medium risk (0.4 < prob <= 0.7): {len(predictions[predictions['risk_level'] == 'Medium'])}")
        print(f"Low risk (prob <= 0.4): {len(predictions[predictions['risk_level'] == 'Low'])}")
        
        # Identify at-risk customers
        print("\n" + "="*70)
        print("At-Risk Customers (Top 10)")
        print("="*70)
        at_risk = identify_at_risk_customers(sample_df, threshold=0.5)
        print(at_risk[['customer_id', 'churn_probability', 'risk_level', 
                      'network_quality_score', 'complaints_count', 'contract_type']].head(10))
        
        # Calculate ROI
        print("\n" + "="*70)
        print("Retention ROI Analysis")
        print("="*70)
        roi_metrics = calculate_retention_roi(at_risk)
        for key, value in roi_metrics.items():
            if isinstance(value, float):
                print(f"{key}: ${value:,.2f}" if 'roi' not in key else f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:,}")
        
    except FileNotFoundError:
        print("\nError: telecom_churn_data.csv not found")
        print("Please run: python data_generator.py")
    except Exception as e:
        print(f"\nError: {e}")

