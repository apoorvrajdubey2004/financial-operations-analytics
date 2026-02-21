"""
Financial Operations Analytics - Complete Analysis Script
========================================================
This script performs comprehensive financial analytics including:
- Revenue forecasting using ARIMA and Prophet
- Customer churn prediction
- Customer segmentation (RFM)
- Cohort analysis
- KPI calculation

Author: Financial Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Analysis libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             precision_score, recall_score, f1_score, roc_curve)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("=" * 80)
print("FINANCIAL OPERATIONS ANALYTICS")
print("=" * 80)

# Load data
print("\n[1/8] Loading data...")
customers = pd.read_csv('financial_customers.csv')
transactions = pd.read_csv('financial_transactions.csv')
monthly_revenue = pd.read_csv('monthly_revenue.csv')

# Convert dates
customers['join_date'] = pd.to_datetime(customers['join_date'])
customers['transaction_date'] = pd.to_datetime(customers['transaction_date'])
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
monthly_revenue['year_month'] = pd.to_datetime(monthly_revenue['year_month'])

print(f"   Customers: {len(customers):,}")
print(f"   Transactions: {len(transactions):,}")
print(f"   Monthly Revenue Records: {len(monthly_revenue):,}")

# ============================================================================
# REVENUE ANALYSIS AND FORECASTING
# ============================================================================

print("\n[2/8] Analyzing revenue trends...")

# Calculate key revenue metrics
total_revenue = monthly_revenue['total_revenue'].sum()
current_mrr = monthly_revenue['total_revenue'].iloc[-1]
arr = current_mrr * 12
avg_revenue_per_account = total_revenue / len(customers)

# Calculate growth rate (last 6 months)
revenue_6mo_ago = monthly_revenue['total_revenue'].iloc[-7]
revenue_now = monthly_revenue['total_revenue'].iloc[-1]
revenue_growth_rate = ((revenue_now - revenue_6mo_ago) / revenue_6mo_ago) * 100

print(f"   Total Revenue: ${total_revenue:,.2f}")
print(f"   Current MRR: ${current_mrr:,.2f}")
print(f"   ARR: ${arr:,.2f}")
print(f"   Growth Rate (6mo): {revenue_growth_rate:+.2f}%")

# Time Series Decomposition
print("\n[3/8] Performing time series analysis...")

# Set index for time series
ts_data = monthly_revenue.set_index('year_month')['total_revenue']

# Seasonal decomposition
decomposition = seasonal_decompose(ts_data, model='multiplicative', period=12)

# ADF Test for stationarity
adf_result = adfuller(ts_data)
print(f"   ADF Statistic: {adf_result[0]:.4f}")
print(f"   p-value: {adf_result[1]:.4f}")
print(f"   Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")

# ARIMA Forecasting
print("\n   Training ARIMA model...")

# Split data
train_size = int(len(ts_data) * 0.8)
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(2, 1, 2))
arima_fit = arima_model.fit()

# Forecast next 12 months
forecast = arima_fit.forecast(steps=12)
forecast_df = pd.DataFrame({
    'month': pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS'),
    'forecasted_revenue': forecast.values
})

# Calculate forecast metrics
from sklearn.metrics import mean_absolute_percentage_error
test_forecast = arima_fit.forecast(steps=len(test_data))
mape = mean_absolute_percentage_error(test_data, test_forecast) * 100
accuracy = 100 - mape

print(f"   Forecast Sum (12mo): ${forecast.sum():,.2f}")
print(f"   Model Accuracy (MAPE): {mape:.2f}%")

# Prophet Forecasting (if available)
if PROPHET_AVAILABLE:
    print("\n   Training Prophet model...")
    
    prophet_data = pd.DataFrame({
        'ds': ts_data.index,
        'y': ts_data.values
    })
    
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    prophet_model.fit(prophet_data)
    
    # Create future dataframe
    future = prophet_model.make_future_dataframe(periods=12, freq='MS')
    prophet_forecast = prophet_model.predict(future)
    
    # Get Prophet components for visualization
    prophet_components = prophet_forecast.tail(12)
    
    # Save Prophet forecast figure
    fig, ax = plt.subplots(figsize=(14, 6))
    prophet_model.plot(prophet_forecast, ax=ax)
    ax.set_title('Prophet Revenue Forecast', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue ($)')
    plt.tight_layout()
    plt.savefig('financial_viz/05_prophet_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save Prophet components figure
    fig = prophet_model.plot_components(prophet_forecast)
    fig.savefig('financial_viz/06_prophet_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   Prophet visualizations saved!")

# ============================================================================
# CUSTOMER CHURN ANALYSIS
# ============================================================================

print("\n[4/8] Analyzing customer churn...")

# Create churn indicator (based on last purchase recency)
reference_date = customers['transaction_date'].max() + timedelta(days=30)
customers['recency'] = (reference_date - customers['transaction_date']).dt.days

# Define churn threshold (90 days inactive)
churn_threshold = 90
customers['is_churned'] = (customers['recency'] > churn_threshold).astype(int)

# Calculate churn rate
churned_customers = customers[customers['is_churned'] == 1]
active_customers = customers[customers['is_churned'] == 0]
churn_rate = len(churned_customers) / len(customers) * 100

print(f"   Total Customers: {len(customers):,}")
print(f"   Churned Customers: {len(churned_customers):,}")
print(f"   Active Customers: {len(active_customers):,}")
print(f"   Churn Rate: {churn_rate:.2f}%")

# Prepare features for churn prediction
feature_cols = ['age', 'usage_score', 'nps_score', 'support_tickets', 
                'plan', 'mrr', 'tenure_months']

# Create a copy for modeling
df_model = customers.copy()
df_model['plan_encoded'] = LabelEncoder().fit_transform(df_model['plan'])

# Calculate additional features
df_model['clv'] = df_model['mrr'] * df_model['tenure_months']
df_model['engagement_score'] = (df_model['usage_score'] + df_model['nps_score']) / 2
df_model['support_per_tenure'] = df_model['support_tickets'] / (df_model['tenure_months'] + 1)

# Features for model
X = df_model[['age', 'usage_score', 'nps_score', 'support_tickets', 
               'plan_encoded', 'mrr', 'tenure_months', 'engagement_score', 
               'support_per_tenure', 'clv']].copy()
y = df_model['is_churned']

# Handle missing values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n   Training churn prediction models...")

# Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_roc = roc_auc_score(y_test, lr_proba)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_roc = roc_auc_score(y_test, rf_proba)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_proba = gb_model.predict_proba(X_test)[:, 1]
gb_roc = roc_auc_score(y_test, gb_proba)

# Find best model
models = {
    'Logistic Regression': {'model': lr_model, 'roc': lr_roc, 'proba': lr_proba, 'pred': lr_pred},
    'Random Forest': {'model': rf_model, 'roc': rf_roc, 'proba': rf_proba, 'pred': rf_pred},
    'Gradient Boosting': {'model': gb_model, 'roc': gb_roc, 'proba': gb_proba, 'pred': gb_pred}
}

best_model_name = max(models, key=lambda x: models[x]['roc'])
best_model = models[best_model_name]['model']
best_roc = models[best_model_name]['roc']
best_proba = models[best_model_name]['proba']
best_pred = models[best_model_name]['pred']

print(f"   Best Model: {best_model_name}")
print(f"   ROC AUC Score: {best_roc:.4f}")

# Feature importance
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(gb_model.feature_importances_)
    }).sort_values('importance', ascending=False)

print("\n   Top Churn Predictors:")
for i, row in feature_importance.head(5).iterrows():
    print(f"      - {row['feature']}: {row['importance']:.4f}")

# Predict churn probability for all customers
all_proba = best_model.predict_proba(X)[:, 1]
customers['churn_probability'] = all_proba

# Identify at-risk customers
at_risk = customers[customers['churn_probability'] > 0.5].copy()
at_risk_mrr = at_risk['mrr'].sum()
potential_revenue_loss = at_risk_mrr * 12

print(f"\n   At-Risk Customers (>50%): {len(at_risk):,}")
print(f"   At-Risk MRR: ${at_risk_mrr:,.2f}")
print(f"   Potential Annual Loss: ${potential_revenue_loss:,.2f}")

# Save at-risk customers
at_risk_save = at_risk[['customer_id', 'segment', 'industry', 'plan', 
                          'mrr', 'clv', 'usage_score', 'nps_score', 
                          'support_tickets', 'churn_probability']].copy()
at_risk_save['risk_category'] = at_risk_save['churn_probability'].apply(
    lambda x: 'Very High Risk' if x > 0.9 else ('High Risk' if x > 0.7 else 'Medium Risk')
)
at_risk_save.to_csv('at_risk_customers.csv', index=False)

# ============================================================================
# RFM SEGMENTATION
# ============================================================================

print("\n[5/8] Performing RFM segmentation...")

# Calculate RFM metrics
reference_date = customers['transaction_date'].max() + timedelta(days=1)

# Group by customer
rfm = customers.groupby('customer_id').agg({
    'transaction_date': lambda x: (reference_date - x.max()).days,
    'transaction_id': 'count',
    'amount': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Merge with customer data
rfm = rfm.merge(customers[['customer_id', 'segment', 'plan', 'join_date']], on='customer_id')

# Create RFM scores (1-5, higher is better)
rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# Handle any NaN scores
rfm['r_score'] = rfm['r_score'].fillna(3).astype(int)
rfm['f_score'] = rfm['f_score'].fillna(3).astype(int)
rfm['m_score'] = rfm['m_score'].fillna(3).astype(int)

# Create customer segments
def segment_customer(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    r, f, m = int(r), int(f), int(m)
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Promising'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Hibernating High Value'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Need Attention'

rfm['customer_segment'] = rfm.apply(segment_customer, axis=1)

# Segment distribution
segment_counts = rfm['customer_segment'].value_counts()
print(f"   Customer Segments:")
for segment, count in segment_counts.items():
    print(f"      - {segment}: {count:,}")

# Save RFM segmentation
rfm_save = rfm[['customer_id', 'segment', 'plan', 'recency', 'frequency', 
                  'monetary', 'r_score', 'f_score', 'm_score', 'customer_segment']]
rfm_save.to_csv('rfm_segmentation.csv', index=False)

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

print("\n[6/8] Performing customer clustering...")

# Prepare features for clustering
cluster_features = customers[['age', 'usage_score', 'nps_score', 'mrr', 'tenure_months']].copy()
cluster_features = cluster_features.fillna(cluster_features.median())

# Scale features
cluster_scaled = scaler.fit_transform(cluster_features)

# Find optimal number of clusters
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_scaled)
    score = silhouette_score(cluster_scaled, labels)
    silhouette_scores.append(score)
    print(f"   K={k}: Silhouette Score = {score:.4f}")

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n   Optimal Number of Clusters: {optimal_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers['cluster'] = kmeans_final.fit_predict(cluster_scaled)

# Cluster profiles
cluster_profile = customers.groupby('cluster').agg({
    'age': 'mean',
    'usage_score': 'mean',
    'nps_score': 'mean',
    'mrr': 'mean',
    'tenure_months': 'mean',
    'customer_id': 'count'
}).round(2)
cluster_profile.columns = ['Avg_Age', 'Avg_Usage', 'Avg_NPS', 'Avg_MRR', 'Avg_Tenure', 'Count']

print("\n   Cluster Profiles:")
print(cluster_profile.to_string())

# ============================================================================
# COHORT ANALYSIS
# ============================================================================

print("\n[7/8] Performing cohort analysis...")

# Create cohort groups
customers['cohort_month'] = customers['join_date'].dt.to_period('M')
customers['cohort_index'] = (customers['transaction_date'].dt.to_period('M') - 
                              customers['cohort_month']).apply(lambda x: x.n)

# Cohort retention
cohort_data = customers.groupby(['cohort_month', 'cohort_index']).agg({
    'customer_id': 'nunique'
}).reset_index()
cohort_data.columns = ['cohort_month', 'cohort_index', 'customers']

# Pivot for retention matrix
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customers')

# Calculate retention percentages
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

# Average retention by month
avg_retention = retention_matrix.mean()

print("\n   Average Retention by Month:")
for month, retention in avg_retention.items():
    if month <= 12:
        print(f"      Month {month}: {retention:.1f}%")

# ============================================================================
# KPI SUMMARY
# ============================================================================

print("\n[8/8] Generating KPI summary...")

# Calculate additional KPIs
avg_clv = customers['clv'].mean()
median_clv = customers['clv'].median()
cac = 500  # Assumed
clv_cac_ratio = avg_clv / cac
avg_payback = (customers['mrr'] / customers['clv']).mean() * 12
avg_lifetime = customers['tenure_months'].mean()

# Create KPI summary
kpi_summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     KEY PERFORMANCE INDICATORS                            ║
╚══════════════════════════════════════════════════════════════════════════╝

REVENUE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total All-Time Revenue         : ${total_revenue:,.2f}
  • Current Monthly MRR            : ${current_mrr:,.2f}
  • Annual Recurring Revenue (ARR) : ${arr:,.2f}
  • Average Revenue per Account   : ${avg_revenue_per_account:,.2f}
  • Revenue Growth Rate (6mo)     : {revenue_growth_rate:+.2f}%
  
CUSTOMER METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total Customers                : {len(customers):,}
  • Active Customers              : {len(active_customers):,}
  • Churned Customers             : {len(churned_customers):,}
  • Overall Churn Rate            : {churn_rate:.2f}%
  • Average Customer Lifetime     : {avg_lifetime:.1f} months
  
CUSTOMER VALUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Average CLV                   : ${avg_clv:,.2f}
  • Median CLV                    : ${median_clv:,.2f}
  • CLV to CAC Ratio              : {clv_cac_ratio:.2f}x (assumed CAC: ${cac})
  • Average Payback Period        : {avg_payback:.1f} months

FORECASTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Forecasted Revenue (Next 12mo): ${forecast.sum():,.2f}
  • Expected Monthly Average      : ${forecast.mean():,.2f}
  • Model Accuracy (MAPE)        : {mape:.2f}%

RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • At-Risk Customers (>50% prob) : {len(at_risk):,}
  • At-Risk MRR                  : ${at_risk_mrr:,.2f}
  • Potential Annual Revenue Loss : ${potential_revenue_loss:,.2f}
"""

# Save KPI summary
with open('kpi_summary.txt', 'w') as f:
    f.write(kpi_summary)

print(kpi_summary)

# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

executive_summary = f"""
================================================================================
                    EXECUTIVE SUMMARY - FINANCIAL OPERATIONS
================================================================================

EXECUTIVE OVERVIEW
------------------
This analysis examines {len(customers):,} customers across {len(transactions):,} transactions
spanning a 5-year period (2020-2024). The company has demonstrated strong growth
with current Monthly Recurring Revenue (MRR) of ${current_mrr:,.2f}.

KEY FINDINGS
------------

1. REVENUE PERFORMANCE
   • Total Revenue: ${total_revenue:,.2f}
   • Current MRR: ${current_mrr:,.2f} | ARR: ${arr:,.2f}
   • 6-Month Growth Rate: {revenue_growth_rate:+.2f}%
   • 12-Month Revenue Forecast: ${forecast.sum():,.2f}
   • Forecast Accuracy (MAPE): {mape:.2f}%

2. CUSTOMER HEALTH
   • Total Customers: {len(customers):,}
   • Active Customers: {len(active_customers):,} ({100-len(churned_customers)/len(customers)*100:.1f}%)
   • Churn Rate: {churn_rate:.2f}%
   • Average Customer Lifetime: {avg_lifetime:.1f} months
   • Average CLV: ${avg_clv:,.2f}

3. CHURN RISK ANALYSIS
   • High-Risk Customers: {len(at_risk):,}
   • At-Risk MRR: ${at_risk_mrr:,.2f}
   • Potential Revenue at Risk: ${potential_revenue_loss:,.2f}/year
   • Best Churn Model: {best_model_name} (ROC AUC: {best_roc:.2%})
   
4. CUSTOMER SEGMENTS
   • Champions: {segment_counts.get('Champions', 0):,}
   • Loyal Customers: {segment_counts.get('Loyal Customers', 0):,}
   • At Risk: {segment_counts.get('At Risk', 0):,}
   • Promising: {segment_counts.get('Promising', 0):,}
   • Need Attention: {segment_counts.get('Need Attention', 0):,}
   • Hibernating: {segment_counts.get('Hibernating High Value', 0):,}
   • Lost: {segment_counts.get('Lost', 0):,}

STRATEGIC RECOMMENDATIONS
-------------------------

IMMEDIATE ACTIONS (Next 30 Days):
1. Contact {len(at_risk):,} at-risk customers with personalized retention offers
2. Implement early warning system for customers with declining usage scores
3. Review and address support tickets for high-risk accounts

SHORT-TERM (1-3 Months):
1. Launch targeted win-back campaign for "At Risk" segment
2. Develop segment-specific customer success playbooks
3. Implement usage monitoring alerts (threshold: usage_score < 50)
4. Create NPS improvement programs for detractors (NPS < 50)

LONG-TERM (6-12 Months):
1. Target 20% churn reduction → Save ${potential_revenue_loss * 0.2:,.0f}/year
2. Expand "Champions" segment through referral programs
3. Build automated churn prediction system
4. Achieve {revenue_growth_rate * 1.2:.0f}% revenue growth

FINANCIAL IMPACT
-----------------
• Current Annual Revenue: ${arr:,.0f}
• Projected Revenue (with 20% churn reduction): ${arr + (potential_revenue_loss * 0.2):,.0f}
• Potential Savings from Retention: ${potential_revenue_loss * 0.2:,.0f}/year

================================================================================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: 2020-2024
Next Forecast Period: 2025
================================================================================
"""

# Save executive summary
with open('EXECUTIVE_SUMMARY_FINANCIAL.txt', 'w') as f:
    f.write(executive_summary)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFiles Generated:")
print("  - kpi_summary.txt")
print("  - EXECUTIVE_SUMMARY_FINANCIAL.txt")
print("  - at_risk_customers.csv")
print("  - rfm_segmentation.csv")
print("  - financial_viz/05_prophet_forecast.png")
print("  - financial_viz/06_prophet_components.png")
print("\nAnalysis completed successfully!")
