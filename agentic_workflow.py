"""
Agentic AI Workflow Simulation
Simulates Nokia's agentic AI systems for automated telecom customer retention workflows.
This demonstrates autonomous decision-making with transparency and explainability.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle

class TelecomAgent:
    """
    Agentic AI agent for telecom customer retention workflows.
    Simulates autonomous decision-making with reasoning transparency.
    """
    
    def __init__(self, model_path='models/xgboost.pkl'):
        """Initialize the agent with a trained churn prediction model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        except:
            self.model_loaded = False
            print("Warning: Model not loaded. Using simulation mode.")
        
        # Agent configuration
        self.config = {
            'high_risk_threshold': 0.7,
            'medium_risk_threshold': 0.4,
            'max_intervention_cost': 50,  # Maximum cost per customer for retention
            'confidence_threshold': 0.8,  # Minimum confidence for autonomous action
        }
        
        # Decision history for learning
        self.decision_history = []
        
    def analyze_customer(self, customer_data: Dict) -> Dict:
        """
        Main agentic workflow: Analyze customer and make retention decision
        
        This simulates Nokia's agentic AI system that:
        1. Detects churn risk
        2. Analyzes root causes
        3. Suggests personalized retention offer
        4. Estimates business impact
        5. Recommends next best action
        6. Logs decision with reasoning (transparency)
        """
        # Step 1: Predict churn risk
        risk_score, confidence = self._predict_churn_risk(customer_data)
        
        # Step 2: Analyze root causes
        root_causes = self._identify_root_causes(customer_data, risk_score)
        
        # Step 3: Generate personalized recommendation
        recommendation = self._generate_recommendation(
            customer_data, risk_score, root_causes
        )
        
        # Step 4: Calculate business impact
        business_impact = self._calculate_business_impact(
            customer_data, risk_score, recommendation
        )
        
        # Step 5: Determine next best action
        next_action = self._determine_next_action(risk_score, confidence)
        
        # Step 6: Create decision log (transparency)
        decision_log = {
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_data.get('customer_id', 'UNKNOWN'),
            'risk_score': risk_score,
            'confidence': confidence,
            'risk_level': self._classify_risk(risk_score),
            'root_causes': root_causes,
            'recommendation': recommendation,
            'business_impact': business_impact,
            'next_action': next_action,
            'reasoning': self._generate_reasoning(customer_data, risk_score, root_causes)
        }
        
        # Store decision for learning
        self.decision_history.append(decision_log)
        
        return decision_log
    
    def _predict_churn_risk(self, customer_data: Dict) -> tuple:
        """Predict churn risk using ML model"""
        if not self.model_loaded:
            # Simulation mode
            risk_score = np.random.beta(2, 3)  # Skewed toward lower risk
            confidence = 0.85
            return risk_score, confidence
        
        try:
            # Prepare features (simplified - would need full preprocessing)
            # In production, this would use the same preprocessing as training
            risk_score = 0.65  # Placeholder - would use actual model prediction
            confidence = 0.88
            return risk_score, confidence
        except Exception as e:
            # Fallback to rule-based
            risk_score = self._rule_based_risk(customer_data)
            confidence = 0.75
            return risk_score, confidence
    
    def _rule_based_risk(self, customer_data: Dict) -> float:
        """Rule-based risk calculation as fallback"""
        risk = 0.0
        
        # Network quality impact
        network_quality = customer_data.get('network_quality_score', 7.0)
        if network_quality < 5:
            risk += 0.3
        elif network_quality < 7:
            risk += 0.15
        
        # Complaints impact
        complaints = customer_data.get('complaints_count', 0)
        if complaints >= 3:
            risk += 0.25
        elif complaints >= 1:
            risk += 0.10
        
        # Contract type impact
        contract = customer_data.get('contract_type', 'Month-to-Month')
        if contract == 'Month-to-Month':
            risk += 0.15
        elif contract == 'Two Year':
            risk -= 0.10
        
        # Service calls impact
        service_calls = customer_data.get('customer_service_calls', 0)
        if service_calls >= 4:
            risk += 0.20
        
        return min(1.0, max(0.0, risk))
    
    def _identify_root_causes(self, customer_data: Dict, risk_score: float) -> List[Dict]:
        """Identify root causes of churn risk"""
        causes = []
        
        # Network quality issues
        network_quality = customer_data.get('network_quality_score', 7.0)
        if network_quality < 6:
            causes.append({
                'factor': 'Network Quality',
                'severity': 'High' if network_quality < 5 else 'Medium',
                'current_value': network_quality,
                'impact': 'Network quality below acceptable threshold',
                'priority': 1
            })
        
        # Complaint issues
        complaints = customer_data.get('complaints_count', 0)
        if complaints >= 2:
            causes.append({
                'factor': 'Customer Complaints',
                'severity': 'High' if complaints >= 3 else 'Medium',
                'current_value': complaints,
                'impact': f'{complaints} unresolved complaints indicate dissatisfaction',
                'priority': 2
            })
        
        # Contract instability
        contract = customer_data.get('contract_type', 'Month-to-Month')
        if contract == 'Month-to-Month':
            causes.append({
                'factor': 'Contract Type',
                'severity': 'Medium',
                'current_value': contract,
                'impact': 'Month-to-month contract indicates low commitment',
                'priority': 3
            })
        
        # Payment issues
        payment_score = customer_data.get('payment_history_score', 8.0)
        if payment_score < 6:
            causes.append({
                'factor': 'Payment History',
                'severity': 'High' if payment_score < 4 else 'Medium',
                'current_value': payment_score,
                'impact': 'Poor payment history suggests financial stress',
                'priority': 2
            })
        
        # High service calls
        service_calls = customer_data.get('customer_service_calls', 0)
        if service_calls >= 3:
            causes.append({
                'factor': 'Support Interactions',
                'severity': 'Medium',
                'current_value': service_calls,
                'impact': 'Frequent support calls indicate ongoing issues',
                'priority': 3
            })
        
        # Sort by priority
        causes.sort(key=lambda x: x['priority'])
        return causes
    
    def _generate_recommendation(self, customer_data: Dict, risk_score: float, 
                                root_causes: List[Dict]) -> Dict:
        """Generate personalized retention recommendation"""
        if risk_score >= self.config['high_risk_threshold']:
            # High risk - aggressive retention
            if 'Network Quality' in [c['factor'] for c in root_causes]:
                recommendation = {
                    'action': 'Network Infrastructure Upgrade',
                    'offer': 'Free network booster device + 20% discount on premium plan',
                    'cost': 35,
                    'urgency': 'Immediate (within 24 hours)',
                    'channel': 'Phone call from retention specialist'
                }
            elif 'Customer Complaints' in [c['factor'] for c in root_causes]:
                recommendation = {
                    'action': 'Dedicated Account Manager',
                    'offer': 'Personal account manager + 25% discount for 6 months',
                    'cost': 45,
                    'urgency': 'Immediate (within 24 hours)',
                    'channel': 'Phone call + email'
                }
            else:
                recommendation = {
                    'action': 'Retention Discount',
                    'offer': '30% discount on current plan for 12 months',
                    'cost': customer_data.get('monthly_charges', 50) * 0.3,
                    'urgency': 'Immediate (within 24 hours)',
                    'channel': 'Automated email + SMS'
                }
        
        elif risk_score >= self.config['medium_risk_threshold']:
            # Medium risk - proactive engagement
            if customer_data.get('contract_type') == 'Month-to-Month':
                recommendation = {
                    'action': 'Contract Upgrade Incentive',
                    'offer': '15% discount for switching to annual contract',
                    'cost': 20,
                    'urgency': 'Within 7 days',
                    'channel': 'Personalized email'
                }
            else:
                recommendation = {
                    'action': 'Loyalty Reward',
                    'offer': 'Free month of premium features',
                    'cost': 15,
                    'urgency': 'Within 14 days',
                    'channel': 'In-app notification + email'
                }
        else:
            # Low risk - maintenance
            recommendation = {
                'action': 'Proactive Check-in',
                'offer': 'Satisfaction survey + small loyalty reward',
                'cost': 5,
                'urgency': 'Within 30 days',
                'channel': 'Email'
            }
        
        return recommendation
    
    def _calculate_business_impact(self, customer_data: Dict, risk_score: float,
                                   recommendation: Dict) -> Dict:
        """Calculate business impact of retention action"""
        monthly_revenue = customer_data.get('monthly_charges', 50)
        acquisition_cost = monthly_revenue * 7  # Industry standard: 7x monthly revenue
        
        # Estimate retention success probability
        if risk_score >= 0.7:
            success_probability = 0.5  # 50% chance of retention with intervention
        elif risk_score >= 0.4:
            success_probability = 0.7  # 70% chance
        else:
            success_probability = 0.9  # 90% chance
        
        # Calculate expected value
        intervention_cost = recommendation.get('cost', 0)
        annual_revenue_preserved = monthly_revenue * 12
        total_value_if_retained = annual_revenue_preserved + acquisition_cost
        
        expected_savings = (total_value_if_retained * success_probability) - intervention_cost
        roi = (expected_savings / intervention_cost * 100) if intervention_cost > 0 else float('inf')
        
        return {
            'monthly_revenue_at_risk': monthly_revenue,
            'annual_revenue_at_risk': annual_revenue_preserved,
            'acquisition_cost_if_lost': acquisition_cost,
            'total_customer_lifetime_value': total_value_if_retained,
            'intervention_cost': intervention_cost,
            'retention_success_probability': success_probability,
            'expected_savings': expected_savings,
            'roi_percentage': roi,
            'estimated_savings': f"${expected_savings:,.2f}"
        }
    
    def _determine_next_action(self, risk_score: float, confidence: float) -> Dict:
        """Determine next best action based on risk and confidence"""
        if risk_score >= self.config['high_risk_threshold']:
            if confidence >= self.config['confidence_threshold']:
                return {
                    'action': 'Autonomous Intervention',
                    'description': 'AI agent automatically triggers retention workflow',
                    'timeline': 'Immediate',
                    'requires_human_approval': False
                }
            else:
                return {
                    'action': 'Escalate to Human Agent',
                    'description': 'High risk but low confidence - human review required',
                    'timeline': 'Within 2 hours',
                    'requires_human_approval': True
                }
        elif risk_score >= self.config['medium_risk_threshold']:
            return {
                'action': 'Scheduled Intervention',
                'description': 'Add to proactive retention campaign queue',
                'timeline': 'Within 7 days',
                'requires_human_approval': False
            }
        else:
            return {
                'action': 'Monitor',
                'description': 'Low risk - continue monitoring for changes',
                'timeline': 'Ongoing',
                'requires_human_approval': False
            }
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level"""
        if risk_score >= self.config['high_risk_threshold']:
            return 'High'
        elif risk_score >= self.config['medium_risk_threshold']:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_reasoning(self, customer_data: Dict, risk_score: float,
                           root_causes: List[Dict]) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = [
            f"Customer has a {risk_score*100:.1f}% churn probability."
        ]
        
        if root_causes:
            top_cause = root_causes[0]
            reasoning_parts.append(
                f"Primary concern: {top_cause['factor']} ({top_cause['impact']})."
            )
        
        if risk_score >= 0.7:
            reasoning_parts.append(
                "Immediate intervention recommended to prevent churn."
            )
        elif risk_score >= 0.4:
            reasoning_parts.append(
                "Proactive engagement recommended to improve retention."
            )
        else:
            reasoning_parts.append(
                "Customer shows low churn risk. Continue standard service."
            )
        
        return " ".join(reasoning_parts)
    
    def batch_analyze(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze multiple customers in batch (agentic workflow at scale)"""
        results = []
        
        for _, customer in customers_df.iterrows():
            decision = self.analyze_customer(customer.to_dict())
            results.append(decision)
        
        return pd.DataFrame(results)
    
    def get_agent_statistics(self) -> Dict:
        """Get statistics about agent decisions (for monitoring)"""
        if not self.decision_history:
            return {'message': 'No decisions made yet'}
        
        df = pd.DataFrame(self.decision_history)
        
        return {
            'total_decisions': len(df),
            'high_risk_customers': len(df[df['risk_level'] == 'High']),
            'medium_risk_customers': len(df[df['risk_level'] == 'Medium']),
            'low_risk_customers': len(df[df['risk_level'] == 'Low']),
            'total_expected_savings': df['business_impact'].apply(
                lambda x: x.get('expected_savings', 0) if isinstance(x, dict) else 0
            ).sum(),
            'autonomous_actions': len(df[df['next_action'].apply(
                lambda x: x.get('requires_human_approval', True) == False
            )]),
            'human_escalations': len(df[df['next_action'].apply(
                lambda x: x.get('requires_human_approval', False) == True
            )])
        }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Agentic AI Workflow Simulation")
    print("="*70)
    
    # Initialize agent
    agent = TelecomAgent()
    
    # Example customer
    customer = {
        'customer_id': 'CUST_001234',
        'age': 42,
        'gender': 'Male',
        'location': 'Urban',
        'tenure_months': 18,
        'plan_type': 'Standard',
        'contract_type': 'Month-to-Month',
        'monthly_charges': 65.0,
        'total_charges': 1170.0,
        'call_duration_minutes': 320,
        'data_usage_gb': 12.5,
        'network_quality_score': 4.2,  # Low - high risk factor
        'dropped_calls': 8,
        'payment_history_score': 7.5,
        'customer_service_calls': 5,  # High - risk factor
        'tech_support_usage': 1,
        'complaints_count': 3  # High - major risk factor
    }
    
    # Analyze customer
    print("\nAnalyzing customer...")
    decision = agent.analyze_customer(customer)
    
    print("\n" + "="*70)
    print("AGENTIC AI DECISION")
    print("="*70)
    print(f"\nCustomer ID: {decision['customer_id']}")
    print(f"Risk Score: {decision['risk_score']:.2%}")
    print(f"Risk Level: {decision['risk_level']}")
    print(f"Confidence: {decision['confidence']:.2%}")
    
    print("\n" + "-"*70)
    print("ROOT CAUSES:")
    print("-"*70)
    for cause in decision['root_causes']:
        print(f"  • {cause['factor']}: {cause['impact']} (Severity: {cause['severity']})")
    
    print("\n" + "-"*70)
    print("RECOMMENDATION:")
    print("-"*70)
    rec = decision['recommendation']
    print(f"  Action: {rec['action']}")
    print(f"  Offer: {rec['offer']}")
    print(f"  Cost: ${rec['cost']}")
    print(f"  Urgency: {rec['urgency']}")
    print(f"  Channel: {rec['channel']}")
    
    print("\n" + "-"*70)
    print("BUSINESS IMPACT:")
    print("-"*70)
    impact = decision['business_impact']
    print(f"  Expected Savings: {impact['estimated_savings']}")
    print(f"  ROI: {impact['roi_percentage']:.1f}%")
    print(f"  Retention Success Probability: {impact['retention_success_probability']:.1%}")
    
    print("\n" + "-"*70)
    print("NEXT ACTION:")
    print("-"*70)
    action = decision['next_action']
    print(f"  {action['action']}")
    print(f"  {action['description']}")
    print(f"  Timeline: {action['timeline']}")
    print(f"  Requires Human Approval: {action['requires_human_approval']}")
    
    print("\n" + "-"*70)
    print("REASONING:")
    print("-"*70)
    print(f"  {decision['reasoning']}")
    
    print("\n" + "="*70)
    print("Agentic AI workflow completed with full transparency")
    print("="*70)

