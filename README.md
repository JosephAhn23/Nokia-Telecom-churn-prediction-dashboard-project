# Telecom Churn Prediction Dashboard

**A comprehensive machine learning solution for predicting customer churn in telecom services, featuring real-time predictions, business impact analysis, and MLOps-ready deployment.**

---

##  **PROJECT DOCUMENTATION**

**[ View Full Project Documentation (PDF)](./Telecom%20churn%20prediction%20-%20Joseph%20Ahn.pdf)**

*Click the link above or find "Telecom churn prediction - Joseph Ahn.pdf" in the repository files to view the complete project summary, technical details, business impact analysis, and alignment with Nokia's ML Engineer role requirements.*

**The PDF includes:**
- Executive summary with key metrics
- Complete technical implementation details
- Business impact calculations ($39M+ revenue preservation)
- Model performance analysis
- Nokia role alignment
- Interview talking points

---

## Quick Overview

This project demonstrates a complete end-to-end ML pipeline for telecom churn prediction, built specifically to address Nokia's ML Engineer role requirements. The system achieves **94.35% accuracy** and **91.06% ROC-AUC** using XGBoost, with the potential to preserve **$39M+ in annual revenue** for a 10M customer base.

### Key Metrics

- **Model Accuracy**: 94.35% (XGBoost)
- **ROC-AUC Score**: 91.06%
- **Business Impact**: $39M+ annual revenue preservation potential
- **ROI**: 1,235% (assuming $5M implementation cost)
- **Churn Reduction**: 0.5% monthly reduction = 50,000 customers retained

### Key Features

- **Real-time Churn Prediction**: Interactive dashboard for instant customer risk assessment
- **Agentic AI Workflow**: Autonomous decision-making with full transparency
- **Network Optimization**: Connects churn prediction to network quality metrics
- **What-if Analysis**: Test retention strategies before implementation
- **Business Impact**: Revenue calculations and ROI analysis
- **MLOps Pipeline**: Production-ready with Kubernetes deployment

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JosephAhn23/Nokia---Telecom-churn-prediction-dashboard-project.git
   cd Nokia---Telecom-churn-prediction-dashboard-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run complete pipeline**
   ```bash
   python run_pipeline.py
   ```
   This will generate data and train all ML models.

4. **Launch the dashboard**
   ```bash
   streamlit run dashboard.py
   ```
   The dashboard will open at `http://localhost:8501`

---

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 0.9435 | 0.5882 | 0.2459 | 0.3468 | 0.9106 |
| Random Forest | 0.9435 | 0.7368 | 0.1148 | 0.1986 | 0.9062 |
| Neural Network | 0.9425 | 0.7692 | 0.0820 | 0.1481 | 0.8870 |

### Top Churn Drivers

1. **Network Quality Score** (3x more predictive than call duration)
2. **Complaints Count** (strongest negative indicator)
3. **Contract Type** (Two-year contracts: 40% lower churn)
4. **Customer Service Calls** (4+ calls = high risk)
5. **Payment History** (Score <5: 2x churn probability)

---

## Business Impact

### ROI Calculation

**For 10M Customer Base:**
- 50,000 customers retained (0.5% churn reduction)
- $39,000,000 annual revenue preserved
- $22,750,000 acquisition cost saved
- **Total Annual Savings: $61,750,000**
- **ROI: 1,235%**

---

## Technical Stack

- **Python 3.9+**: Core programming language
- **pandas/scikit-learn**: Data processing and ML
- **TensorFlow 2.15**: Deep learning models
- **XGBoost**: Gradient boosting
- **Streamlit**: Interactive dashboard
- **Plotly**: Advanced visualizations
- **Docker**: Containerization
- **Kubernetes**: Production orchestration

---

## Docker Deployment

```bash
# Build the Docker image
docker build -t telecom-churn-dashboard .

# Run the container
docker run -p 8501:8501 telecom-churn-dashboard
```

Or use Docker Compose:
```bash
docker-compose up -d
```

---

## Alignment with Nokia ML Engineer Role

This project directly addresses Nokia's requirements:

- ✅ **Python & ML Libraries**: pandas, scikit-learn, TensorFlow
- ✅ **Churn Prediction**: Core focus - 94%+ accuracy
- ✅ **Docker & Kubernetes**: Complete containerization + K8s deployment
- ✅ **Telecom Domain**: Realistic telecom-specific features
- ✅ **Business Impact**: Revenue calculations and ROI analysis
- ✅ **Agentic AI Systems**: Autonomous decision-making workflow
- ✅ **MLOps Pipeline**: Production-ready with drift detection

---
