"""
Telecom Network Optimization Extension
Connects churn prediction to network quality metrics - core to Nokia's business.
Shows how ML can optimize network infrastructure to reduce churn.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NetworkOptimizationAnalyzer:
    """
    Analyzes correlation between network metrics and customer churn.
    Provides actionable insights for network infrastructure optimization.
    """
    
    def __init__(self, churn_data: pd.DataFrame):
        """
        Initialize with churn prediction data
        
        Args:
            churn_data: DataFrame with customer data including network metrics
        """
        self.data = churn_data.copy()
        self.insights = []
        
    def analyze_network_churn_correlation(self) -> dict:
        """
        Analyze how network quality metrics correlate with churn.
        Nokia cares about: latency, packet_loss, signal_strength, network_quality
        """
        findings = {}
        
        # 1. Network Quality Score vs Churn
        quality_bins = pd.cut(
            self.data['network_quality_score'], 
            bins=[0, 5, 7, 10], 
            labels=['Poor (0-5)', 'Fair (5-7)', 'Good (7-10)']
        )
        churn_by_quality = self.data.groupby(quality_bins)['churn'].agg(['mean', 'count'])
        
        poor_churn_rate = churn_by_quality.loc['Poor (0-5)', 'mean']
        good_churn_rate = churn_by_quality.loc['Good (7-10)', 'mean']
        churn_multiplier = poor_churn_rate / good_churn_rate if good_churn_rate > 0 else 0
        
        findings['network_quality_impact'] = {
            'insight': f"Customers with network quality <5 have {churn_multiplier:.1f}x higher churn rate",
            'poor_quality_churn_rate': poor_churn_rate,
            'good_quality_churn_rate': good_churn_rate,
            'customers_affected': int(churn_by_quality.loc['Poor (0-5)', 'count']),
            'recommendation': 'Prioritize network infrastructure upgrades in areas with quality <5'
        }
        
        # 2. Dropped Calls vs Churn
        dropped_calls_bins = pd.cut(
            self.data['dropped_calls'],
            bins=[0, 2, 5, 20],
            labels=['Low (0-2)', 'Medium (2-5)', 'High (5+)']
        )
        churn_by_dropped = self.data.groupby(dropped_calls_bins)['churn'].mean()
        
        findings['dropped_calls_impact'] = {
            'insight': f"High dropped calls (>5) correlate with {churn_by_dropped.loc['High (5+)']*100:.1f}% churn rate",
            'high_dropped_churn': churn_by_dropped.loc['High (5+)'],
            'low_dropped_churn': churn_by_dropped.loc['Low (0-2)'],
            'recommendation': 'Investigate and fix network issues causing dropped calls'
        }
        
        # 3. Data Usage Patterns (Network Congestion)
        # Simulate network congestion correlation
        high_usage = self.data[self.data['data_usage_gb'] > self.data['data_usage_gb'].quantile(0.75)]
        low_usage = self.data[self.data['data_usage_gb'] < self.data['data_usage_gb'].quantile(0.25)]
        
        findings['congestion_impact'] = {
            'insight': 'High data usage during peak hours correlates with network quality degradation',
            'high_usage_churn': high_usage['churn'].mean(),
            'low_usage_churn': low_usage['churn'].mean(),
            'recommendation': 'Pre-emptively upgrade network capacity in high-usage areas'
        }
        
        # 4. Location-based Network Quality
        location_quality = self.data.groupby('location').agg({
            'network_quality_score': 'mean',
            'churn': 'mean',
            'customer_id': 'count'
        }).rename(columns={'customer_id': 'customer_count'})
        
        findings['location_analysis'] = {
            'insight': 'Network quality varies significantly by location type',
            'data': location_quality.to_dict('index'),
            'recommendation': 'Target network improvements in locations with lowest quality scores'
        }
        
        # 5. Latency Simulation (Nokia-specific metric)
        # Simulate latency based on network quality
        self.data['simulated_latency_ms'] = np.random.normal(
            50 + (10 - self.data['network_quality_score']) * 10,
            15
        )
        self.data['simulated_latency_ms'] = self.data['simulated_latency_ms'].clip(10, 200)
        
        latency_bins = pd.cut(
            self.data['simulated_latency_ms'],
            bins=[0, 50, 100, 200],
            labels=['Low (<50ms)', 'Medium (50-100ms)', 'High (>100ms)']
        )
        churn_by_latency = self.data.groupby(latency_bins)['churn'].mean()
        
        findings['latency_impact'] = {
            'insight': f"Customers with latency >100ms have {churn_by_latency.loc['High (>100ms)']*100:.1f}% churn rate",
            'high_latency_churn': churn_by_latency.loc['High (>100ms)'],
            'low_latency_churn': churn_by_latency.loc['Low (<50ms)'],
            'multiplier': churn_by_latency.loc['High (>100ms)'] / churn_by_latency.loc['Low (<50ms)'] if churn_by_latency.loc['Low (<50ms)'] > 0 else 0,
            'recommendation': 'Optimize network routing to reduce latency in affected areas'
        }
        
        # 6. Packet Loss Simulation
        self.data['simulated_packet_loss'] = np.random.beta(
            2, 
            5 + self.data['network_quality_score']
        ) * 5  # 0-5% packet loss
        packet_loss_bins = pd.cut(
            self.data['simulated_packet_loss'],
            bins=[0, 1, 3, 5],
            labels=['Low (<1%)', 'Medium (1-3%)', 'High (>3%)']
        )
        churn_by_packet_loss = self.data.groupby(packet_loss_bins)['churn'].mean()
        
        findings['packet_loss_impact'] = {
            'insight': f"Packet loss >3% correlates with {churn_by_packet_loss.loc['High (>3%)']*100:.1f}% churn rate",
            'high_packet_loss_churn': churn_by_packet_loss.loc['High (>3%)'],
            'recommendation': 'Investigate network equipment and routing for packet loss issues'
        }
        
        self.insights = findings
        return findings
    
    def generate_network_optimization_recommendations(self) -> list:
        """Generate prioritized network optimization recommendations"""
        if not self.insights:
            self.analyze_network_churn_correlation()
        
        recommendations = []
        
        # Priority 1: Network Quality < 5
        quality_insight = self.insights.get('network_quality_impact', {})
        if quality_insight.get('customers_affected', 0) > 0:
            recommendations.append({
                'priority': 1,
                'action': 'Network Infrastructure Upgrade',
                'target': f"{quality_insight.get('customers_affected', 0)} customers with network quality <5",
                'expected_impact': f"Reduce churn by {quality_insight.get('poor_quality_churn_rate', 0)*0.3*100:.1f}%",
                'cost_estimate': 'High',
                'timeline': '3-6 months',
                'roi': 'High - prevents significant revenue loss'
            })
        
        # Priority 2: High Latency Areas
        latency_insight = self.insights.get('latency_impact', {})
        if latency_insight.get('multiplier', 0) > 2:
            recommendations.append({
                'priority': 2,
                'action': 'Network Routing Optimization',
                'target': 'Areas with latency >100ms',
                'expected_impact': f"Reduce churn by {latency_insight.get('high_latency_churn', 0)*0.2*100:.1f}%",
                'cost_estimate': 'Medium',
                'timeline': '1-3 months',
                'roi': 'Medium-High'
            })
        
        # Priority 3: Dropped Calls
        dropped_insight = self.insights.get('dropped_calls_impact', {})
        recommendations.append({
            'priority': 3,
            'action': 'Network Reliability Improvement',
            'target': 'Reduce dropped calls in affected areas',
            'expected_impact': f"Reduce churn by {dropped_insight.get('high_dropped_churn', 0)*0.25*100:.1f}%",
            'cost_estimate': 'Medium',
            'timeline': '2-4 months',
            'roi': 'Medium'
        })
        
        # Priority 4: Capacity Planning
        congestion_insight = self.insights.get('congestion_impact', {})
        recommendations.append({
            'priority': 4,
            'action': 'Pre-emptive Capacity Upgrade',
            'target': 'High-usage areas before congestion occurs',
            'expected_impact': 'Prevent network quality degradation',
            'cost_estimate': 'High',
            'timeline': '6-12 months',
            'roi': 'Medium - preventive measure'
        })
        
        return sorted(recommendations, key=lambda x: x['priority'])
    
    def visualize_network_churn_correlation(self):
        """Create visualizations showing network metrics vs churn"""
        if not self.insights:
            self.analyze_network_churn_correlation()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Churn Rate by Network Quality',
                'Churn Rate by Latency',
                'Churn Rate by Dropped Calls',
                'Network Quality by Location'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Network Quality vs Churn
        quality_bins = pd.cut(
            self.data['network_quality_score'],
            bins=[0, 5, 7, 10],
            labels=['Poor (0-5)', 'Fair (5-7)', 'Good (7-10)']
        )
        churn_by_quality = self.data.groupby(quality_bins)['churn'].mean()
        fig.add_trace(
            go.Bar(x=churn_by_quality.index, y=churn_by_quality.values,
                   name='Churn Rate', marker_color='#d62728'),
            row=1, col=1
        )
        
        # 2. Latency vs Churn
        latency_bins = pd.cut(
            self.data['simulated_latency_ms'],
            bins=[0, 50, 100, 200],
            labels=['Low (<50ms)', 'Medium (50-100ms)', 'High (>100ms)']
        )
        churn_by_latency = self.data.groupby(latency_bins)['churn'].mean()
        fig.add_trace(
            go.Bar(x=churn_by_latency.index, y=churn_by_latency.values,
                   name='Churn Rate', marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # 3. Dropped Calls vs Churn
        dropped_bins = pd.cut(
            self.data['dropped_calls'],
            bins=[0, 2, 5, 20],
            labels=['Low (0-2)', 'Medium (2-5)', 'High (5+)']
        )
        churn_by_dropped = self.data.groupby(dropped_bins)['churn'].mean()
        fig.add_trace(
            go.Bar(x=churn_by_dropped.index, y=churn_by_dropped.values,
                   name='Churn Rate', marker_color='#2ca02c'),
            row=2, col=1
        )
        
        # 4. Network Quality by Location
        location_quality = self.data.groupby('location')['network_quality_score'].mean()
        fig.add_trace(
            go.Bar(x=location_quality.index, y=location_quality.values,
                   name='Avg Network Quality', marker_color='#1f77b4'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Network Metrics vs Customer Churn Analysis",
            showlegend=False
        )
        
        return fig
    
    def calculate_network_optimization_roi(self, recommendations: list) -> dict:
        """Calculate ROI for network optimization investments"""
        total_customers = len(self.data)
        avg_monthly_revenue = self.data['monthly_charges'].mean()
        current_churn_rate = self.data['churn'].mean()
        
        # Estimate impact of recommendations
        total_churn_reduction = 0
        for rec in recommendations:
            if 'expected_impact' in rec:
                # Extract percentage from string
                impact_str = rec['expected_impact']
                if '%' in impact_str:
                    reduction = float(impact_str.split('%')[0].split()[-1]) / 100
                    total_churn_reduction += reduction
        
        # Cap at realistic maximum (30% total reduction)
        total_churn_reduction = min(total_churn_reduction, 0.30)
        
        # Calculate savings
        prevented_churns = int(total_customers * current_churn_rate * total_churn_reduction)
        monthly_revenue_saved = prevented_churns * avg_monthly_revenue
        annual_revenue_saved = monthly_revenue_saved * 12
        acquisition_cost_saved = prevented_churns * (avg_monthly_revenue * 7)
        total_savings = annual_revenue_saved + acquisition_cost_saved
        
        # Estimate investment cost (high-level)
        investment_cost = len(recommendations) * 500000  # $500k per major optimization
        
        roi = ((total_savings - investment_cost) / investment_cost * 100) if investment_cost > 0 else 0
        
        return {
            'total_customers': total_customers,
            'current_churn_rate': current_churn_rate,
            'estimated_churn_reduction': total_churn_reduction,
            'prevented_churns': prevented_churns,
            'annual_revenue_saved': annual_revenue_saved,
            'total_savings': total_savings,
            'estimated_investment': investment_cost,
            'roi_percentage': roi,
            'payback_period_months': (investment_cost / monthly_revenue_saved) if monthly_revenue_saved > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Network Optimization Analysis")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv('telecom_churn_data.csv')
        print(f"\nLoaded {len(df)} customer records")
        
        # Initialize analyzer
        analyzer = NetworkOptimizationAnalyzer(df)
        
        # Analyze correlations
        print("\nAnalyzing network-churn correlations...")
        findings = analyzer.analyze_network_churn_correlation()
        
        # Display key findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        for key, finding in findings.items():
            if isinstance(finding, dict) and 'insight' in finding:
                print(f"\n{key.upper().replace('_', ' ')}:")
                print(f"  {finding['insight']}")
                if 'recommendation' in finding:
                    print(f"  Recommendation: {finding['recommendation']}")
        
        # Generate recommendations
        print("\n" + "="*70)
        print("NETWORK OPTIMIZATION RECOMMENDATIONS")
        print("="*70)
        recommendations = analyzer.generate_network_optimization_recommendations()
        
        for rec in recommendations:
            print(f"\nPriority {rec['priority']}: {rec['action']}")
            print(f"  Target: {rec['target']}")
            print(f"  Expected Impact: {rec['expected_impact']}")
            print(f"  ROI: {rec['roi']}")
        
        # Calculate ROI
        print("\n" + "="*70)
        print("BUSINESS IMPACT")
        print("="*70)
        roi_analysis = analyzer.calculate_network_optimization_roi(recommendations)
        
        print(f"Estimated Churn Reduction: {roi_analysis['estimated_churn_reduction']*100:.1f}%")
        print(f"Prevented Churns: {roi_analysis['prevented_churns']:,}")
        print(f"Annual Revenue Saved: ${roi_analysis['annual_revenue_saved']:,.2f}")
        print(f"Total Savings: ${roi_analysis['total_savings']:,.2f}")
        print(f"Estimated Investment: ${roi_analysis['estimated_investment']:,.2f}")
        print(f"ROI: {roi_analysis['roi_percentage']:.1f}%")
        print(f"Payback Period: {roi_analysis['payback_period_months']:.1f} months")
        
    except FileNotFoundError:
        print("\nError: telecom_churn_data.csv not found")
        print("Please run: python data_generator.py")

