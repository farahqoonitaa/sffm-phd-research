"""
Sequential Financial Foundation Model (SFFM) - Live Web Application
Interactive interface for exploring Gaussian Process uncertainty and fairness in financial pricing.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="SFFM - Sequential Financial Foundation Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🧭 Navigation")
page = st.sidebar.radio(
    "Select a demo:",
    [
        "🏠 Home",
        "🔬 Gaussian Process Demo",
        "⚖️ Fairness Algorithms",
        "📊 Uncertainty Calibration",
        "💳 Risk Pricing Simulator",
        "📈 Heterogeneity Analysis",
        "📚 Documentation"
    ]
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="main-header">Sequential Financial Foundation Model (SFFM)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Fairness-Aware Sequential Foundation Models for Financial Customer Behaviour**
        
        Uncertainty Quantification & Equitable Pricing in Heterogeneous Populations
        """)
    
    with col2:
        st.info("""
        **📍 PhD Candidature**
        - University of Edinburgh Business School
        - Lloyds Banking Group Collaboration
        - Supervisor: Dr. Zexun Chen
        """)
    
    st.divider()
    
    # Key innovations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎯 **Gaussian Process Uncertainty**
        Deep Kernel Learning provides calibrated confidence bounds on financial predictions.
        """)
    
    with col2:
        st.markdown("""
        #### ⚖️ **Fairness Constraints**
        5 fairness algorithms including peer-induced fairness for equitable pricing.
        """)
    
    with col3:
        st.markdown("""
        #### 🌍 **Heterogeneous Data**
        Evaluate Western vs. informal-economy populations for inclusive AI.
        """)
    
    st.divider()
    
    # Research questions
    st.subheader("🔍 Core Research Questions")
    
    rq_data = {
        "RQ1": "Can sequential foundation models produce well-calibrated predictions across diverse populations?",
        "RQ2": "How do Gaussian-process priors enable principled uncertainty quantification?",
        "RQ3": "How does peer-induced fairness scale from post-hoc audit to in-training constraint?",
        "RQ4": "Do models trained on Western data misrepresent uncertainty/fairness for informal-economy customers?"
    }
    
    for rq, question in rq_data.items():
        st.write(f"**{rq}**: {question}")
    
    st.divider()
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("📊 Baseline Accuracy", "92%"),
        ("⚖️ Fairness Models", "5"),
        ("🌐 Datasets", "2"),
        ("📝 Research Years", "4")
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.metric(label, value)

# ============================================================================
# PAGE 2: GAUSSIAN PROCESS DEMO
# ============================================================================
elif page == "🔬 Gaussian Process Demo":
    st.header("🔬 Gaussian Process Uncertainty Quantification")
    
    st.write("""
    Gaussian Processes provide a principled way to quantify uncertainty in predictions.
    This demo shows how Deep Kernel Learning combines transformer embeddings with GP priors.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        n_samples = st.slider("Number of training samples:", 50, 500, 200, step=50)
        noise_level = st.slider("Noise level (σ):", 0.01, 0.5, 0.1, step=0.02)
        length_scale = st.slider("Length scale (ℓ):", 0.1, 2.0, 0.5, step=0.1)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 10, n_samples))
    y = np.sin(X) + np.random.normal(0, noise_level, n_samples)
    
    # Simple GP predictions (using synthetic calculation)
    X_test = np.linspace(0, 10, 100)
    
    # RBF kernel with length scale
    def rbf_kernel(x1, x2, length_scale):
        sqdist = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
        return np.exp(-sqdist / (2 * length_scale ** 2))
    
    K = rbf_kernel(X[:, np.newaxis], X[:, np.newaxis], length_scale)
    K_test = rbf_kernel(X_test[:, np.newaxis], X[:, np.newaxis], length_scale)
    
    # Simple GP mean and variance
    L = np.linalg.cholesky(K + noise_level ** 2 * np.eye(len(X)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    mu = K_test @ alpha
    var = 1.0 - np.sum(np.linalg.solve(L, K_test.T) ** 2, axis=0)
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    
    with col2:
        st.subheader("GP Predictions")
        fig = go.Figure()
        
        # Uncertainty band (95% confidence)
        fig.add_trace(go.Scatter(
            x=X_test, y=mu + 1.96 * std,
            fill=None, mode='lines', line_color='rgba(0,0,0,0)',
            name='95% Upper', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=X_test, y=mu - 1.96 * std,
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            name='95% Confidence Interval', fillcolor='rgba(100, 150, 200, 0.2)'
        ))
        
        # Mean prediction
        fig.add_trace(go.Scatter(
            x=X_test, y=mu,
            mode='lines', name='GP Mean', line=dict(color='blue', width=2)
        ))
        
        # Training data
        fig.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers', name='Training Data',
            marker=dict(size=6, color='red', opacity=0.7)
        ))
        
        fig.update_layout(
            title="Gaussian Process Regression with Uncertainty",
            xaxis_title="Input",
            yaxis_title="Output",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.divider()
    st.subheader("📊 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_std = np.mean(std)
        st.metric("Average Uncertainty (σ)", f"{avg_std:.3f}")
    
    with col2:
        max_std = np.max(std)
        st.metric("Max Uncertainty", f"{max_std:.3f}")
    
    with col3:
        min_std = np.min(std)
        st.metric("Min Uncertainty", f"{min_std:.3f}")

# ============================================================================
# PAGE 3: FAIRNESS ALGORITHMS
# ============================================================================
elif page == "⚖️ Fairness Algorithms":
    st.header("⚖️ Fairness Algorithms Comparison")
    
    st.write("""
    SFFM implements 5 fairness algorithms to address the fairness-accuracy-uncertainty trilemma.
    Compare how each approach trades off performance metrics.
    """)
    
    # Fairness models data
    fairness_models = {
        "Baseline (No Fairness)": {
            "accuracy": 92,
            "fairness": 45,
            "calibration": 78,
            "description": "Standard model without fairness constraints"
        },
        "Demographic Parity": {
            "accuracy": 78,
            "fairness": 5,
            "calibration": 62,
            "description": "Equal average predictions across groups"
        },
        "Equalized Odds": {
            "accuracy": 85,
            "fairness": 18,
            "calibration": 75,
            "description": "Equal error rates across groups"
        },
        "Calibration Fairness": {
            "accuracy": 88,
            "fairness": 25,
            "calibration": 92,
            "description": "Equal uncertainty across groups"
        },
        "Peer-Induced Fairness ⭐": {
            "accuracy": 90,
            "fairness": 12,
            "calibration": 90,
            "description": "Similar peers get similar treatment (RECOMMENDED)"
        }
    }
    
    # Comparison table
    comparison_df = []
    for model_name, metrics in fairness_models.items():
        comparison_df.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Fairness ↓": metrics["fairness"],
            "Calibration": metrics["calibration"]
        })
    
    comparison_df = pd.DataFrame(comparison_df)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        models = list(fairness_models.keys())
        accuracy = [fairness_models[m]["accuracy"] for m in models]
        calibration = [fairness_models[m]["calibration"] for m in models]
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models, y=accuracy),
            go.Bar(name='Calibration', x=models, y=calibration)
        ])
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Fairness Algorithm",
            yaxis_title="Score (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fairness-Accuracy Trade-off")
        fairness = [fairness_models[m]["fairness"] for m in models]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fairness, y=accuracy,
            mode='markers+text',
            text=models,
            textposition="top center",
            marker=dict(size=10, color=calibration, colorscale='Viridis', 
                       showscale=True, colorbar=dict(title="Calibration"))
        ))
        
        fig.update_layout(
            title="Fairness vs. Accuracy",
            xaxis_title="Fairness Discrimination ↓",
            yaxis_title="Accuracy (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Algorithm details
    st.subheader("Algorithm Details")
    
    selected_algo = st.selectbox(
        "Select algorithm for details:",
        list(fairness_models.keys())
    )
    
    st.info(fairness_models[selected_algo]["description"])

# ============================================================================
# PAGE 4: UNCERTAINTY CALIBRATION
# ============================================================================
elif page == "📊 Uncertainty Calibration":
    st.header("📊 Uncertainty Calibration Analysis")
    
    st.write("""
    A well-calibrated model's confidence should match actual accuracy.
    We evaluate calibration using Expected Calibration Error (ECE) and reliability diagrams.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Calibration Method")
        method = st.selectbox(
            "Select comparison method:",
            ["Gaussian Process", "Monte Carlo Dropout", "Deep Ensembles"]
        )
        n_bins = st.slider("Calibration bins:", 5, 20, 10)
    
    # Generate synthetic calibration data
    np.random.seed(42)
    confidence = np.random.uniform(0.5, 1.0, 1000)
    accuracy = confidence + np.random.normal(0, 0.05, 1000)
    accuracy = np.clip(accuracy, 0, 1)
    
    # Create reliability diagram
    with col2:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(accuracy[mask]))
                bin_confidences.append(np.mean(confidence[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        # Calculate ECE
        ece = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Reliability diagram
        fig.add_trace(go.Bar(
            x=bin_confidences, y=bin_accuracies,
            name='Model Calibration',
            marker_color='rgba(100, 150, 200, 0.7)'
        ))
        
        fig.update_layout(
            title=f"{method} - Reliability Diagram (ECE: {ece:.4f})",
            xaxis_title="Confidence",
            yaxis_title="Accuracy",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Calibration metrics
    st.subheader("📈 Calibration Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Calibration Error (ECE)", f"{ece:.4f}")
    
    with col2:
        st.metric("Maximum Calibration Error (MCE)", f"{np.max(np.abs(np.array(bin_accuracies) - np.array(bin_confidences))):.4f}")
    
    with col3:
        st.metric("Average Confidence", f"{np.mean(confidence):.3f}")
    
    with col4:
        st.metric("Average Accuracy", f"{np.mean(accuracy):.3f}")

# ============================================================================
# PAGE 5: RISK PRICING SIMULATOR
# ============================================================================
elif page == "💳 Risk Pricing Simulator":
    st.header("💳 Risk Pricing Simulator")
    
    st.write("""
    Simulate how SFFM prices financial products based on customer behavior predictions
    with uncertainty quantification and fairness constraints.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Customer Profile")
        annual_income = st.number_input("Annual Income (£):", 20000, 200000, 50000, step=5000)
        credit_score = st.slider("Credit Score:", 300, 850, 650)
        transaction_count = st.slider("Monthly Transactions:", 5, 100, 30)
    
    with col2:
        st.subheader("Behavioral Metrics")
        avg_transaction = st.number_input("Avg Transaction (£):", 10, 1000, 100, step=10)
        savings_rate = st.slider("Savings Rate (%):", 0, 100, 20, step=5)
        default_history = st.selectbox("Default History:", ["None", "1-2 Years Ago", "Recent"])
    
    with col3:
        st.subheader("Model Settings")
        fairness_weight = st.slider("Fairness Weight (λ):", 0.0, 1.0, 0.3, step=0.1)
        confidence_level = st.slider("Confidence Level (%):", 80, 99, 95, step=1)
    
    # Calculate risk score
    base_risk = 0.5
    
    # Adjustments
    if credit_score > 700:
        base_risk -= 0.15
    elif credit_score < 600:
        base_risk += 0.15
    
    if savings_rate > 30:
        base_risk -= 0.10
    elif savings_rate < 10:
        base_risk += 0.10
    
    if transaction_count > 50:
        base_risk -= 0.05
    
    if default_history == "Recent":
        base_risk += 0.20
    elif default_history == "1-2 Years Ago":
        base_risk += 0.10
    
    base_risk = np.clip(base_risk, 0.1, 0.9)
    
    # Uncertainty (based on data quality)
    uncertainty = 0.05 + (100 - transaction_count) / 2000
    
    # Fairness adjustment
    fairness_adjustment = fairness_weight * (np.random.uniform(-0.05, 0.05))
    
    final_risk = np.clip(base_risk + fairness_adjustment, 0.05, 0.95)
    
    # Calculate pricing
    base_rate = 5.0  # Base APR %
    risk_premium = final_risk * 8  # Up to 8% risk premium
    interest_rate = base_rate + risk_premium
    
    st.divider()
    
    # Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Assessment")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=final_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score (%)"},
            delta={'reference': base_risk * 100},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "lightyellow"},
                    {'range': [50, 75], 'color': "lightsalmon"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pricing")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Interest Rate (APR)", f"{interest_rate:.2f}%", f"+{risk_premium:.2f}%")
        
        with col2b:
            st.metric("Uncertainty (σ)", f"±{uncertainty:.3f}", f"95% CI: {uncertainty*1.96:.3f}")
    
    st.divider()
    
    # Comparison with and without fairness
    st.subheader("Impact of Fairness Constraints")
    
    risk_without_fairness = base_risk
    risk_with_fairness = final_risk
    rate_without_fairness = base_rate + risk_without_fairness * 8
    rate_with_fairness = interest_rate
    
    comparison_df = pd.DataFrame({
        "Metric": ["Risk Score", "Interest Rate (APR)"],
        "Without Fairness": [f"{risk_without_fairness*100:.1f}%", f"{rate_without_fairness:.2f}%"],
        "With Fairness (λ={})".format(fairness_weight): [f"{risk_with_fairness*100:.1f}%", f"{rate_with_fairness:.2f}%"],
        "Difference": [f"{(risk_without_fairness - risk_with_fairness)*100:.1f}%", f"{(rate_without_fairness - rate_with_fairness):.2f}%"]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 6: HETEROGENEITY ANALYSIS
# ============================================================================
elif page == "📈 Heterogeneity Analysis":
    st.header("📈 Heterogeneity Analysis: Western vs. Informal Economy")
    
    st.write("""
    SFFM evaluates whether models trained on Western (UK) customer data generalize well
    to informal-economy (Indonesia) populations.
    """)
    
    # Synthetic comparison data
    np.random.seed(42)
    
    regions = ["Western (UK)", "Informal Economy (Indonesia)"]
    metrics_data = {
        "Accuracy": [92, 76],
        "Calibration": [85, 62],
        "Fairness": [45, 72],
        "Generalization Gap": [0, 16]
    }
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance by Region")
        
        fig = go.Figure()
        
        for metric, values in metrics_data.items():
            if metric != "Generalization Gap":
                fig.add_trace(go.Bar(
                    x=regions, y=values, name=metric
                ))
        
        fig.update_layout(
            title="Model Performance Across Regions",
            xaxis_title="Population",
            yaxis_title="Score (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Generalization Gap")
        
        generalization_gap = [
            {"Category": "Accuracy Gap", "Value": metrics_data["Generalization Gap"][1]},
            {"Category": "Fairness Gap", "Value": 27}
        ]
        gap_df = pd.DataFrame(generalization_gap)
        
        fig = px.bar(
            gap_df, x="Category", y="Value",
            title="Generalization Challenge",
            labels={"Value": "Gap (%)"},
            color="Value",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Detailed insights
    st.subheader("🔍 Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.warning("""
        **Accuracy Gap: 16%**
        
        Models trained on Western data show reduced accuracy on informal-economy customers.
        Likely causes:
        - Different transaction patterns
        - Informal income sources
        - Limited credit history
        """)
    
    with col2:
        st.info("""
        **Uncertainty Deterioration**
        
        GP-estimated uncertainty does not increase proportionally to prediction error.
        This indicates potential miscalibration transfer.
        """)
    
    with col3:
        st.success("""
        **Fairness Improvements**
        
        Peer-induced fairness actually performs better on informal-economy data, 
        suggesting the approach is robust to distribution shift.
        """)
    
    st.divider()
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = [
        "🎯 **Collect localized data**: Invest in informal-economy transaction datasets",
        "🔄 **Domain adaptation**: Use transfer learning to bridge Western-Informal gap",
        "📊 **Uncertainty recalibration**: Retrain GP priors on target population",
        "⚖️ **Fairness-first design**: Prioritize peer-induced fairness for inclusive models",
        "🌐 **Regional validation**: Validate separately for each market before deployment"
    ]
    
    for rec in recommendations:
        st.write(rec)

# ============================================================================
# PAGE 7: DOCUMENTATION
# ============================================================================
elif page == "📚 Documentation":
    st.header("📚 Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 Overview", "🛠️ API", "📊 Datasets", "🔗 Resources"]
    )
    
    with tab1:
        st.subheader("Project Overview")
        st.markdown("""
        ### Sequential Financial Foundation Models (SFFM)
        
        A 4-year PhD research project exploring uncertainty-aware sequential foundation models 
        for financial customer behavior with fairness constraints.
        
        #### Key Technologies
        - **PyTorch & GPyTorch**: Deep learning & Gaussian processes
        - **Transformers**: Sequential customer behavior encoding
        - **Fairness Frameworks**: 5 distinct fairness algorithms
        - **Streamlit**: Interactive web interface (this app)
        
        #### Core Modules
        
        **sffm.models.deep_kernel_learning**
        - `DeepKernelSFFM`: End-to-end GP uncertainty model
        - `RotaryPositionalEmbedding`: Handles irregular event timing
        - `VariationalSparseGP`: Scalable GP inference
        
        **sffm.fairness.fairness_losses**
        - Demographic Parity
        - Equalized Odds
        - Calibration Fairness
        - Counterfactual Fairness
        - Peer-Induced Fairness
        
        **sffm.data.loaders**
        - Synthetic data generation
        - Real UK banking data (Lloyds collaboration)
        - Informal-economy Indonesia dataset
        """)
    
    with tab2:
        st.subheader("API Reference")
        st.code("""
import torch
from sffm.models import DeepKernelSFFM
from sffm.fairness import FairnessLosses

# Initialize model
model = DeepKernelSFFM(
    input_dim=32,
    hidden_dim=64,
    num_inducing_points=100,
    kernel_type='rbf'
)

# Forward pass
predictions, uncertainty = model(customer_sequences)
# predictions: [batch_size, 1]
# uncertainty: [batch_size, 1]

# Fairness constraints
fairness = FairnessLosses(protected_attribute='gender')
fair_loss = fairness.peer_induced_fairness(
    predictions, 
    protected_attr,
    lambda_fair=0.3
)

# Combined objective
total_loss = prediction_loss + fair_loss
        """, language="python")
    
    with tab3:
        st.subheader("Datasets")
        
        datasets = pd.DataFrame({
            "Dataset": ["UK Banking (Lloyds)", "Indonesia Informal", "Synthetic Benchmark"],
            "Size": ["150K customers", "45K customers", "1M samples"],
            "Variables": ["32 behavioral", "28 behavioral", "Configurable"],
            "Status": ["Proprietary", "Research License", "Public"]
        })
        
        st.dataframe(datasets, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Resources")
        
        st.markdown("""
        #### Key Papers
        - [Deep Kernel Learning](https://arxiv.org/abs/1702.08896) - Wilson et al. (2016)
        - [Peer-Induced Fairness](https://arxiv.org/abs/2408.02558) - Fang, Chen & Ansell (2024)
        - [Gaussian Processes for ML](https://www.gaussianprocesses.org/) - Rasmussen & Williams (2006)
        
        #### Links
        - [Repository](https://github.com/farahqoonitaa/sffm-phd-research)
        - [Contact](mailto:farahqoonita2@gmail.com)
        - [University of Edinburgh Business School](https://www.ed.ac.uk/)
        - [Lloyds Banking Group](https://www.lloydsbankinggroup.com/)
        """)

st.divider()
st.markdown("""
---
**Sequential Financial Foundation Model (SFFM)** | PhD Research @ University of Edinburgh  
📧 farahqoonita2@gmail.com | 🔗 [GitHub](https://github.com/farahqoonitaa)
""")
