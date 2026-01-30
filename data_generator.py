"""
Telecom Churn Data Generator
Generates realistic synthetic telecom customer data for churn prediction modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_telecom_data(n_samples=10000, random_seed=42):
    """
    Generate realistic telecom customer data with churn labels.
    
    Features:
    - Demographics: age, gender, location
    - Usage: call_duration, data_usage, sms_count, roaming_usage
    - Service: plan_type, tenure_months, contract_type
    - Quality: network_quality_score, dropped_calls, complaints_count
    - Financial: monthly_charges, payment_history_score, total_charges
    - Support: customer_service_calls, tech_support_usage
    
    Target:
    - churn: 0 (no churn) or 1 (churned)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.normal(45, 15)
        age = max(18, min(80, int(age)))
        gender = np.random.choice(['Male', 'Female'])
        location = np.random.choice(['Urban', 'Suburban', 'Rural'])
        
        # Service details
        plan_type = np.random.choice(['Basic', 'Standard', 'Premium', 'Enterprise'], 
                                     p=[0.3, 0.4, 0.25, 0.05])
        contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                        p=[0.5, 0.3, 0.2])
        tenure_months = np.random.exponential(24)
        tenure_months = min(int(tenure_months), 72)  # Cap at 6 years
        
        # Usage patterns (correlated with plan type)
        plan_multiplier = {'Basic': 0.7, 'Standard': 1.0, 'Premium': 1.5, 'Enterprise': 2.0}[plan_type]
        
        call_duration = max(0, np.random.normal(300 * plan_multiplier, 150))
        data_usage_gb = max(0, np.random.gamma(5 * plan_multiplier, 2))
        sms_count = max(0, np.random.poisson(50 * plan_multiplier))
        roaming_usage = max(0, np.random.exponential(2 * plan_multiplier))
        
        # Network quality (affects churn significantly)
        network_quality_score = np.random.beta(2, 1) * 10  # Skewed toward higher scores
        dropped_calls = max(0, int(np.random.poisson(5 - network_quality_score * 0.3)))
        
        # Financial
        base_charge = {'Basic': 30, 'Standard': 50, 'Premium': 80, 'Enterprise': 150}[plan_type]
        monthly_charges = base_charge + np.random.normal(0, 10)
        monthly_charges = max(20, monthly_charges)
        
        # Payment history (0-10, higher is better)
        payment_history_score = np.random.beta(3, 1) * 10
        
        # Support interactions (high support calls = higher churn risk)
        customer_service_calls = np.random.poisson(1.5)
        tech_support_usage = np.random.choice([0, 1], p=[0.7, 0.3])
        complaints_count = max(0, int(np.random.poisson(0.5)))
        
        # Calculate total charges based on tenure
        total_charges = monthly_charges * tenure_months
        
        # Churn probability calculation (realistic factors)
        churn_prob = 0.0
        
        # High churn factors
        if contract_type == 'Month-to-Month':
            churn_prob += 0.15
        if network_quality_score < 5:
            churn_prob += 0.25
        if complaints_count >= 3:
            churn_prob += 0.30
        if customer_service_calls >= 4:
            churn_prob += 0.20
        if payment_history_score < 5:
            churn_prob += 0.15
        if tenure_months < 6:
            churn_prob += 0.10
        
        # Low churn factors (retention)
        if contract_type == 'Two Year':
            churn_prob -= 0.15
        if tenure_months > 24:
            churn_prob -= 0.10
        if network_quality_score > 8:
            churn_prob -= 0.10
        if plan_type == 'Enterprise':
            churn_prob -= 0.10
        
        # Add some randomness
        churn_prob += np.random.normal(0, 0.1)
        churn_prob = max(0, min(1, churn_prob))
        
        churn = 1 if churn_prob > 0.5 else 0
        
        # Adjust for realistic churn rate (~25-30%)
        if np.random.random() < 0.28:  # Force some churn cases
            if churn == 0 and churn_prob > 0.3:
                churn = 1
        else:
            if churn == 1 and churn_prob < 0.6:
                churn = 0
        
        data.append({
            'customer_id': f'CUST_{i+1:06d}',
            'age': age,
            'gender': gender,
            'location': location,
            'tenure_months': tenure_months,
            'plan_type': plan_type,
            'contract_type': contract_type,
            'monthly_charges': round(monthly_charges, 2),
            'total_charges': round(total_charges, 2),
            'call_duration_minutes': round(call_duration, 2),
            'data_usage_gb': round(data_usage_gb, 2),
            'sms_count': sms_count,
            'roaming_usage': round(roaming_usage, 2),
            'network_quality_score': round(network_quality_score, 2),
            'dropped_calls': dropped_calls,
            'payment_history_score': round(payment_history_score, 2),
            'customer_service_calls': customer_service_calls,
            'tech_support_usage': tech_support_usage,
            'complaints_count': complaints_count,
            'churn': churn
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating telecom churn dataset...")
    df = generate_telecom_data(n_samples=10000)
    
    # Save to CSV
    df.to_csv('telecom_churn_data.csv', index=False)
    print(f"Dataset saved: {len(df)} samples")
    print(f"Churn rate: {df['churn'].mean()*100:.2f}%")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nFeature statistics:")
    print(df.describe())

