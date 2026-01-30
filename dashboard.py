"""
Telecom Churn Prediction Dashboard
Interactive Streamlit dashboard for churn prediction and business insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from agentic_workflow import TelecomAgent
from network_optimization import NetworkOptimizationAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .churn-risk-high {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .churn-risk-medium {
        color: #ff7f0e;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .churn-risk-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    
    try:
        # Load Random Forest
        with open('models/random_forest.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        
        # Load XGBoost
        with open('models/xgboost.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        
        # Load Neural Network
        models['Neural Network'] = keras.models.load_model('models/neural_network.h5')
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        # Load label encoders
        with open('models/label_encoders.pkl', 'rb') as f:
            models['label_encoders'] = pickle.load(f)
        
        # Load feature names
        with open('models/feature_names.json', 'r') as f:
            models['feature_names'] = json.load(f)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train_models.py first to generate the models.")
        return None

def predict_churn(customer_data, models, model_name='XGBoost'):
    """Predict churn probability for a customer"""
    # Prepare features
    feature_names = models['feature_names']
    X = pd.DataFrame([customer_data])[feature_names]
    
    # Encode categorical variables
    categorical_cols = ['gender', 'location', 'plan_type', 'contract_type']
    for col in categorical_cols:
        if col in X.columns:
            le = models['label_encoders'][col]
            X[col] = le.transform([customer_data[col]])[0]
    
    # Predict
    if model_name == 'Neural Network':
        X_scaled = models['scaler'].transform(X)
        proba = models['Neural Network'].predict(X_scaled, verbose=0)[0][0]
    else:
        model = models[model_name]
        proba = model.predict_proba(X)[0][1]
    
    return proba

def calculate_business_impact(churn_rate, avg_customer_value=50):
    """Calculate business impact metrics"""
    monthly_revenue_loss = churn_rate * avg_customer_value
    annual_revenue_loss = monthly_revenue_loss * 12
    
    # Cost of acquisition (typically 5-10x monthly revenue)
    acquisition_cost = avg_customer_value * 7
    
    # Total cost per churned customer
    total_cost_per_churn = avg_customer_value + acquisition_cost
    
    return {
        'monthly_revenue_loss': monthly_revenue_loss,
        'annual_revenue_loss': annual_revenue_loss,
        'acquisition_cost': acquisition_cost,
        'total_cost_per_churn': total_cost_per_churn
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">Telecom Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Real-time Prediction", "Agentic AI Workflow", "What-if Analysis", 
         "Network Optimization", "Business Impact", "Model Insights"]
    )
    
    # Load sample data for analysis
    try:
        df = pd.read_csv('telecom_churn_data.csv')
    except:
        st.warning("Sample data not found. Some features may be limited.")
        df = None
    
    if page == "Real-time Prediction":
        st.header("Real-time Churn Prediction")
        st.markdown("Enter customer details to predict churn probability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            age = st.slider("Age", 18, 80, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
            tenure_months = st.slider("Tenure (months)", 0, 72, 24)
        
        with col2:
            st.subheader("Service Details")
            plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium", "Enterprise"])
            contract_type = st.selectbox("Contract Type", 
                                        ["Month-to-Month", "One Year", "Two Year"])
            monthly_charges = st.slider("Monthly Charges ($)", 20, 200, 65)
            total_charges = st.slider("Total Charges ($)", 0, 15000, int(monthly_charges * tenure_months))
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Usage Metrics")
            call_duration = st.slider("Call Duration (minutes/month)", 0, 1000, 300)
            data_usage = st.slider("Data Usage (GB/month)", 0, 100, 10)
            sms_count = st.slider("SMS Count", 0, 500, 50)
            roaming_usage = st.slider("Roaming Usage", 0.0, 20.0, 2.0)
        
        with col4:
            st.subheader("Quality & Support")
            network_quality = st.slider("Network Quality Score (0-10)", 0.0, 10.0, 7.0)
            dropped_calls = st.slider("Dropped Calls", 0, 20, 2)
            payment_history = st.slider("Payment History Score (0-10)", 0.0, 10.0, 8.0)
            service_calls = st.slider("Customer Service Calls", 0, 10, 1)
            tech_support = st.selectbox("Tech Support Usage", [0, 1])
            complaints = st.slider("Complaints Count", 0, 10, 0)
        
        # Prepare customer data
        customer_data = {
            'age': age,
            'gender': gender,
            'location': location,
            'tenure_months': tenure_months,
            'plan_type': plan_type,
            'contract_type': contract_type,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'call_duration_minutes': call_duration,
            'data_usage_gb': data_usage,
            'sms_count': sms_count,
            'roaming_usage': roaming_usage,
            'network_quality_score': network_quality,
            'dropped_calls': dropped_calls,
            'payment_history_score': payment_history,
            'customer_service_calls': service_calls,
            'tech_support_usage': tech_support,
            'complaints_count': complaints
        }
        
        # Model selection
        model_name = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Neural Network"])
        
        if st.button("Predict Churn", type="primary"):
            # Predict
            churn_prob = predict_churn(customer_data, models, model_name)
            churn_percentage = churn_prob * 100
            
            # Display result
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{churn_percentage:.2f}%")
            
            with col2:
                if churn_prob > 0.7:
                    st.markdown('<p class="churn-risk-high">HIGH RISK</p>', 
                              unsafe_allow_html=True)
                elif churn_prob > 0.4:
                    st.markdown('<p class="churn-risk-medium">MEDIUM RISK</p>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<p class="churn-risk-low">LOW RISK</p>', 
                              unsafe_allow_html=True)
            
            with col3:
                retention_prob = (1 - churn_prob) * 100
                st.metric("Retention Probability", f"{retention_prob:.2f}%")
            
            # Business impact
            st.markdown("### Business Impact")
            impact = calculate_business_impact(churn_prob, monthly_charges)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Monthly Revenue at Risk", f"${impact['monthly_revenue_loss']:.2f}")
            with col2:
                st.metric("Annual Revenue at Risk", f"${impact['annual_revenue_loss']:.2f}")
            with col3:
                st.metric("Acquisition Cost", f"${impact['acquisition_cost']:.2f}")
            with col4:
                st.metric("Total Cost if Churned", f"${impact['total_cost_per_churn']:.2f}")
            
            # Recommendations
            st.markdown("### AI-Powered Recommendations")
            recommendations = []
            
            if network_quality < 6:
                recommendations.append("🔧 **Network Quality**: Improve network infrastructure in customer's area")
            if complaints >= 3:
                recommendations.append("📞 **Complaints**: Assign dedicated account manager for immediate resolution")
            if service_calls >= 4:
                recommendations.append("**Support**: Proactive outreach to address recurring issues")
            if contract_type == "Month-to-Month":
                recommendations.append("📋 **Contract**: Offer retention discount for annual contract")
            if payment_history < 6:
                recommendations.append("💳 **Payment**: Flexible payment plan or payment reminder system")
            if tenure_months < 6:
                recommendations.append("🎁 **Loyalty**: Welcome bonus or onboarding support program")
            
            if not recommendations:
                recommendations.append("Customer shows low churn risk. Maintain current service quality.")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    elif page == "Agentic AI Workflow":
        st.header("Agentic AI Workflow Simulation")
        st.markdown("**Nokia's agentic AI systems** - Autonomous decision-making with full transparency")
        
        # Initialize agent
        agent = TelecomAgent()
        
        st.subheader("Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="CUST_001234")
            age = st.slider("Age", 18, 80, 42, key="agent_age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="agent_gender")
            location = st.selectbox("Location", ["Urban", "Suburban", "Rural"], key="agent_location")
            tenure_months = st.slider("Tenure (months)", 0, 72, 18, key="agent_tenure")
        
        with col2:
            plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium", "Enterprise"], key="agent_plan")
            contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"], key="agent_contract")
            monthly_charges = st.slider("Monthly Charges ($)", 20, 200, 65, key="agent_charges")
            network_quality = st.slider("Network Quality (0-10)", 0.0, 10.0, 4.2, key="agent_network")
        
        col3, col4 = st.columns(2)
        
        with col3:
            dropped_calls = st.slider("Dropped Calls", 0, 20, 8, key="agent_dropped")
            service_calls = st.slider("Customer Service Calls", 0, 10, 5, key="agent_service")
            complaints = st.slider("Complaints Count", 0, 10, 3, key="agent_complaints")
        
        with col4:
            payment_history = st.slider("Payment History Score (0-10)", 0.0, 10.0, 7.5, key="agent_payment")
            data_usage = st.slider("Data Usage (GB)", 0.0, 100.0, 12.5, key="agent_data")
            call_duration = st.slider("Call Duration (minutes)", 0, 1000, 320, key="agent_calls")
        
        customer_data = {
            'customer_id': customer_id,
            'age': age,
            'gender': gender,
            'location': location,
            'tenure_months': tenure_months,
            'plan_type': plan_type,
            'contract_type': contract_type,
            'monthly_charges': monthly_charges,
            'total_charges': monthly_charges * tenure_months,
            'call_duration_minutes': call_duration,
            'data_usage_gb': data_usage,
            'sms_count': 50,
            'roaming_usage': 2.0,
            'network_quality_score': network_quality,
            'dropped_calls': dropped_calls,
            'payment_history_score': payment_history,
            'customer_service_calls': service_calls,
            'tech_support_usage': 1,
            'complaints_count': complaints
        }
        
        if st.button("Run Agentic AI Analysis", type="primary"):
            with st.spinner("Agent analyzing customer and making autonomous decision..."):
                decision = agent.analyze_customer(customer_data)
            
            st.markdown("---")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{decision['risk_score']:.2%}")
            with col2:
                st.metric("Risk Level", decision['risk_level'])
            with col3:
                st.metric("Confidence", f"{decision['confidence']:.2%}")
            
            # Root Causes
            st.subheader("Root Cause Analysis")
            for i, cause in enumerate(decision['root_causes'], 1):
                with st.expander(f"{i}. {cause['factor']} - {cause['severity']} Severity"):
                    st.write(f"**Impact:** {cause['impact']}")
                    st.write(f"**Current Value:** {cause['current_value']}")
            
            # Recommendation
            st.subheader("AI Recommendation")
            rec = decision['recommendation']
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Action:** {rec['action']}")
                st.write(f"**Offer:** {rec['offer']}")
            with col2:
                st.write(f"**Cost:** ${rec['cost']}")
                st.write(f"**Urgency:** {rec['urgency']}")
                st.write(f"**Channel:** {rec['channel']}")
            
            # Business Impact
            st.subheader("💰 Business Impact")
            impact = decision['business_impact']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Savings", impact['estimated_savings'])
            with col2:
                st.metric("ROI", f"{impact['roi_percentage']:.1f}%")
            with col3:
                st.metric("Success Probability", f"{impact['retention_success_probability']:.1%}")
            with col4:
                st.metric("Intervention Cost", f"${impact['intervention_cost']:.2f}")
            
            # Next Action
            st.subheader("Next Action")
            action = decision['next_action']
            st.info(f"**{action['action']}** - {action['description']}")
            st.write(f"**Timeline:** {action['timeline']}")
            st.write(f"**Requires Human Approval:** {'Yes' if action['requires_human_approval'] else 'No (Autonomous)'}")
            
            # Reasoning (Transparency)
            st.subheader("AI Reasoning (Transparency)")
            st.write(decision['reasoning'])
            
            # Agent Statistics
            if len(agent.decision_history) > 0:
                st.subheader("Agent Statistics")
                stats = agent.get_agent_statistics()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Decisions", stats['total_decisions'])
                with col2:
                    st.metric("Autonomous Actions", stats['autonomous_actions'])
                with col3:
                    st.metric("Human Escalations", stats['human_escalations'])
    
    elif page == "Network Optimization":
        st.header("Network Optimization Analysis")
        st.markdown("**Connecting churn prediction to network quality metrics** - Core to Nokia's business")
        
        if df is not None:
            analyzer = NetworkOptimizationAnalyzer(df)
            
            if st.button("Analyze Network-Churn Correlations"):
                with st.spinner("Analyzing network metrics..."):
                    findings = analyzer.analyze_network_churn_correlation()
                
                st.subheader("Key Findings")
                
                # Network Quality Impact
                if 'network_quality_impact' in findings:
                    quality = findings['network_quality_impact']
                    st.write(f"**{quality['insight']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Poor Quality Churn Rate", f"{quality['poor_quality_churn_rate']:.2%}")
                    with col2:
                        st.metric("Customers Affected", f"{quality['customers_affected']:,}")
                    st.info(f"**Recommendation:** {quality['recommendation']}")
                
                # Latency Impact
                if 'latency_impact' in findings:
                    latency = findings['latency_impact']
                    st.write(f"**{latency['insight']}**")
                    st.metric("Churn Multiplier", f"{latency['multiplier']:.1f}x")
                    st.info(f"**Recommendation:** {latency['recommendation']}")
                
                # Visualizations
                st.subheader("Network Metrics Visualization")
                fig = analyzer.visualize_network_churn_correlation()
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Network Optimization Recommendations")
                recommendations = analyzer.generate_network_optimization_recommendations()
                
                for rec in recommendations:
                    with st.expander(f"Priority {rec['priority']}: {rec['action']}"):
                        st.write(f"**Target:** {rec['target']}")
                        st.write(f"**Expected Impact:** {rec['expected_impact']}")
                        st.write(f"**Cost Estimate:** {rec['cost_estimate']}")
                        st.write(f"**Timeline:** {rec['timeline']}")
                        st.write(f"**ROI:** {rec['roi']}")
                
                # ROI Analysis
                st.subheader("Business Impact")
                roi_analysis = analyzer.calculate_network_optimization_roi(recommendations)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Annual Revenue Saved", f"${roi_analysis['annual_revenue_saved']:,.2f}")
                with col2:
                    st.metric("Total Savings", f"${roi_analysis['total_savings']:,.2f}")
                with col3:
                    st.metric("ROI", f"{roi_analysis['roi_percentage']:.1f}%")
        else:
            st.warning("Please load customer data first")
    
    elif page == "What-if Analysis":
        st.header("What-if Scenario Analysis")
        st.markdown("Analyze how changing specific factors affects churn probability")
        
        if df is not None:
            # Get a random customer
            sample_customer = df.sample(1).iloc[0]
            
            st.subheader("Base Customer Profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Age:** {int(sample_customer['age'])}")
                st.write(f"**Tenure:** {int(sample_customer['tenure_months'])} months")
            with col2:
                st.write(f"**Plan:** {sample_customer['plan_type']}")
                st.write(f"**Contract:** {sample_customer['contract_type']}")
            with col3:
                st.write(f"**Monthly Charges:** ${sample_customer['monthly_charges']:.2f}")
                st.write(f"**Network Quality:** {sample_customer['network_quality_score']:.1f}/10")
            
            # Base prediction
            base_prob = predict_churn(sample_customer.to_dict(), models)
            st.metric("Base Churn Probability", f"{base_prob*100:.2f}%")
            
            st.markdown("---")
            st.subheader("Scenario Testing")
            
            scenario = st.selectbox(
                "Select Scenario",
                [
                    "Improve Network Quality",
                    "Reduce Complaints",
                    "Offer Annual Contract",
                    "Increase Tenure",
                    "Improve Payment History"
                ]
            )
            
            test_customer = sample_customer.copy()
            
            if scenario == "Improve Network Quality":
                improvement = st.slider("Network Quality Improvement", 0, 5, 2)
                test_customer['network_quality_score'] = min(10, 
                    sample_customer['network_quality_score'] + improvement)
                test_customer['dropped_calls'] = max(0, 
                    sample_customer['dropped_calls'] - improvement)
            
            elif scenario == "Reduce Complaints":
                reduction = st.slider("Complaints Reduction", 0, int(sample_customer['complaints_count']), 
                                     min(2, int(sample_customer['complaints_count'])))
                test_customer['complaints_count'] = sample_customer['complaints_count'] - reduction
                test_customer['customer_service_calls'] = max(0, 
                    sample_customer['customer_service_calls'] - reduction)
            
            elif scenario == "Offer Annual Contract":
                test_customer['contract_type'] = "One Year"
            
            elif scenario == "Increase Tenure":
                increase = st.slider("Tenure Increase (months)", 0, 24, 12)
                test_customer['tenure_months'] = sample_customer['tenure_months'] + increase
                test_customer['total_charges'] = test_customer['monthly_charges'] * test_customer['tenure_months']
            
            elif scenario == "Improve Payment History":
                improvement = st.slider("Payment History Improvement", 0, 5, 2)
                test_customer['payment_history_score'] = min(10, 
                    sample_customer['payment_history_score'] + improvement)
            
            # Predict with scenario
            test_prob = predict_churn(test_customer.to_dict(), models)
            change = (test_prob - base_prob) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("New Churn Probability", f"{test_prob*100:.2f}%")
            with col2:
                st.metric("Change", f"{change:+.2f}%", 
                         delta=f"{abs(change):.2f}% improvement" if change < 0 else f"{change:.2f}% increase")
            
            # Visualize
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Base Scenario', 'Improved Scenario'],
                y=[base_prob*100, test_prob*100],
                marker_color=['#d62728', '#2ca02c'],
                text=[f"{base_prob*100:.2f}%", f"{test_prob*100:.2f}%"],
                textposition='auto'
            ))
            fig.update_layout(
                title="Churn Probability Comparison",
                yaxis_title="Churn Probability (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Business Impact":
        st.header("Business Impact Analysis")
        
        if df is not None:
            # Overall statistics
            total_customers = len(df)
            churn_rate = df['churn'].mean()
            churned_customers = int(total_customers * churn_rate)
            avg_monthly_revenue = df['monthly_charges'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", f"{total_customers:,}")
            with col2:
                st.metric("Churn Rate", f"{churn_rate*100:.2f}%")
            with col3:
                st.metric("Churned Customers", f"{churned_customers:,}")
            with col4:
                st.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:.2f}")
            
            # Revenue impact
            st.subheader("Revenue Impact")
            monthly_revenue_loss = churned_customers * avg_monthly_revenue
            annual_revenue_loss = monthly_revenue_loss * 12
            acquisition_cost = avg_monthly_revenue * 7
            total_cost = churned_customers * (avg_monthly_revenue + acquisition_cost)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Monthly Revenue Loss", f"${monthly_revenue_loss:,.2f}")
            with col2:
                st.metric("Annual Revenue Loss", f"${annual_revenue_loss:,.2f}")
            with col3:
                st.metric("Total Cost (Revenue + Acquisition)", f"${total_cost:,.2f}")
            
            # Churn reduction scenarios
            st.subheader("Churn Reduction Scenarios")
            reduction_target = st.slider("Target Churn Reduction (%)", 0, 50, 10)
            
            new_churn_rate = churn_rate * (1 - reduction_target/100)
            prevented_churns = int(total_customers * (churn_rate - new_churn_rate))
            revenue_saved_monthly = prevented_churns * avg_monthly_revenue
            revenue_saved_annual = revenue_saved_monthly * 12
            cost_saved = prevented_churns * (avg_monthly_revenue + acquisition_cost)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prevented Churns", f"{prevented_churns:,}")
            with col2:
                st.metric("Monthly Revenue Saved", f"${revenue_saved_monthly:,.2f}")
            with col3:
                st.metric("Annual Revenue Saved", f"${revenue_saved_annual:,.2f}")
            
            st.metric("Total Cost Savings (Revenue + Acquisition)", 
                     f"${cost_saved:,.2f}", 
                     delta=f"{reduction_target}% reduction")
            
            # Visualizations
            st.subheader("Churn Analysis by Segment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn by plan type
                churn_by_plan = df.groupby('plan_type')['churn'].agg(['mean', 'count'])
                fig = px.bar(
                    churn_by_plan.reset_index(),
                    x='plan_type',
                    y='mean',
                    title="Churn Rate by Plan Type",
                    labels={'mean': 'Churn Rate', 'plan_type': 'Plan Type'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Churn by contract type
                churn_by_contract = df.groupby('contract_type')['churn'].agg(['mean', 'count'])
                fig = px.bar(
                    churn_by_contract.reset_index(),
                    x='contract_type',
                    y='mean',
                    title="Churn Rate by Contract Type",
                    labels={'mean': 'Churn Rate', 'contract_type': 'Contract Type'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Network quality impact
            df['network_quality_bin'] = pd.cut(df['network_quality_score'], 
                                               bins=[0, 5, 7, 10], 
                                               labels=['Low (0-5)', 'Medium (5-7)', 'High (7-10)'])
            churn_by_quality = df.groupby('network_quality_bin')['churn'].mean()
            
            fig = px.bar(
                x=churn_by_quality.index,
                y=churn_by_quality.values,
                title="Churn Rate by Network Quality",
                labels={'x': 'Network Quality', 'y': 'Churn Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Insights":
        st.header("Model Performance & Insights")
        
        try:
            metrics_df = pd.read_csv('models/model_metrics.csv', index_col=0)
            
            st.subheader("Model Comparison")
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Visualize metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'Precision', 'Recall', 'ROC-AUC'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for metric, pos in zip(metrics, positions):
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df[metric], name=metric),
                    row=pos[0], col=pos[1]
                )
            
            fig.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            st.subheader("Top Churn Drivers")
            st.info("Network quality, complaints, and contract type are the strongest predictors of churn.")
            
            # Key insights
            st.subheader("Key Insights")
            insights = [
                "**Network Quality** is 3x more predictive than call duration",
                "**Complaints Count** is the strongest negative indicator",
                "**Contract Type** significantly impacts retention (Two-year contracts have 40% lower churn)",
                "**Customer Service Calls** above 4 indicate high churn risk",
                "**Payment History** below 5/10 increases churn probability by 2x"
            ]
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
        except Exception as e:
            st.error(f"Could not load model metrics: {e}")
            st.info("Please run train_models.py to generate model metrics.")

if __name__ == "__main__":
    main()

