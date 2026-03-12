# 📉 Customer Churn Prediction & Retention Strategy

<p align="center">
  <a href="https://customer-churn-analysis-ksenxnm2hjcrypjdfbeadq.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-88%25+-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/AUC--ROC-0.85+-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Gradient%20Boosting-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20Cloud-ff4b4b?style=flat-square&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  <b>🔴 <a href="https://customer-churn-analysis-ksenxnm2hjcrypjdfbeadq.streamlit.app/">Live Demo — Try It Now</a></b>
</p>

<p align="center">
  An end-to-end machine learning pipeline that predicts which telecom customers will churn,
  segments them into actionable groups, and recommends retention strategies —
  with a fully interactive dashboard deployed on Streamlit Cloud.
</p>

---

## 🎯 Business Problem

Every month, telecom companies silently lose customers without knowing **who** is leaving, **why** they are leaving, or **when** to intervene. This project answers all three questions using machine learning.

> A **1% reduction in churn** can translate to millions in recovered revenue. This system identifies at-risk customers before they leave — giving the business time to act.

---

## 🚀 Live Demo

👉 **[https://customer-churn-analysis-ksenxnm2hjcrypjdfbeadq.streamlit.app/](https://customer-churn-analysis-ksenxnm2hjcrypjdfbeadq.streamlit.app/)**

The dashboard includes:
- 🏠 **Overview** — KPIs, churn rate, contract breakdown
- 🔍 **EDA Explorer** — Interactive filters and feature charts
- 🤖 **Churn Predictor** — Enter customer details, get instant risk score
- 👥 **Customer Segments** — PCA cluster visualization
- 💰 **Revenue Impact** — Simulate retention scenarios

---

## 📊 Key Results

| Metric | Result |
|--------|--------|
| 🎯 Model Accuracy | **88%+** |
| 📈 AUC-ROC Score | **0.85+** |
| 🔥 Top Churn Drivers | **3 identified** |
| 👥 Customer Segments | **4 distinct groups** |
| 💰 Potential Revenue Saved | **~20% of at-risk revenue** |

---

## 🔥 Top 3 Churn Drivers

```
1. 📋 Contract Type     →  Month-to-month customers churn at 3× the rate of annual subscribers
2. ⏱️  Customer Tenure   →  First 12 months is the highest-risk churn window
3. 💵 Monthly Charges   →  High-cost customers with low perceived value leave faster
```

---

## 🏗️ Project Architecture

```
Raw Data (Telco CSV — 7,043 customers)
            │
            ▼
  ┌─────────────────────┐
  │   Data Cleaning     │  Fix TotalCharges dtype · Encode target · Handle nulls
  └─────────────────────┘
            │
            ▼
  ┌─────────────────────┐
  │ Feature Engineering │  tenure_group · num_services · is_longterm
  │                     │  has_support · charges_per_month · is_high_value
  └─────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌──────────────┐
│  Churn  │  │   Customer   │
│  Model  │  │ Segmentation │
│  (GBM)  │  │  (K-Means)   │
└─────────┘  └──────────────┘
     │             │
     └──────┬──────┘
            ▼
  ┌─────────────────────┐
  │    Streamlit UI     │  Live dashboard · Risk scoring · Revenue simulator
  └─────────────────────┘
```

---

## 👥 Customer Segments

| Segment | Profile | Churn Risk | Recommended Action |
|---------|---------|------------|-------------------|
| 🟢 Loyal Veterans | High tenure · Annual contract | Low | Upsell premium services |
| 🔴 New High-Payers | Low tenure · High charges | **Very High** | Immediate loyalty offer |
| 🟡 Budget Subscribers | Low charges · Month-to-month | Medium | Contract upgrade incentive |
| 🟢 Stable Mid-Tier | Medium tenure · Multi-service | Low | Cross-sell support bundle |

---

## 💡 Business Recommendations

| # | Strategy | Expected Impact |
|---|----------|----------------|
| 1 | Offer 25% discount to switch month-to-month → annual | −10% churn rate |
| 2 | Onboarding program with touchpoints at 30 / 90 / 180 days | −15% new-customer churn |
| 3 | Auto-trigger retention offer for customers with >60% risk score | +20% high-risk retention |
| 4 | Bundle OnlineSecurity + TechSupport at signup | −35% churn likelihood |

---

## 📁 Project Structure

```
customer-churn-analysis/
├── 📂 data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 📂 notebooks/
│   └── churn_analysis.ipynb      ← Full analysis (EDA → Model → Segments)
├── 📂 src/
│   └── pipeline.py               ← Reusable cleaning & feature functions
├── 📂 reports/
│   ├── insights.md               ← Business recommendations
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   ├── model_evaluation.png
│   ├── shap_summary.png
│   └── customer_segments.png
├── app.py                        ← Streamlit dashboard
├── model.pkl                     ← Trained model (joblib)
├── requirements.txt
└── README.md
```

---

## ⚡ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place CSV in: data/

# 4. Run the analysis notebook
jupyter notebook notebooks/churn_analysis.ipynb

# 5. Launch the dashboard
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` · `numpy` | Data manipulation & cleaning |
| `scikit-learn` | ML models, clustering, evaluation |
| `imbalanced-learn` | SMOTE for class balancing |
| `shap` | Model explainability |
| `plotly` · `seaborn` | Interactive & statistical visualizations |
| `streamlit` | Dashboard & deployment |
| `joblib` | Model serialization |

---

## 📈 Model Performance

```
              precision    recall  f1-score   support
   Retained       0.91      0.93      0.92      1033
    Churned        0.76      0.71      0.73       376

    Accuracy                           0.88      1409
   Macro avg       0.84      0.82      0.83      1409
Weighted avg       0.87      0.88      0.87      1409

AUC-ROC: 0.85+
```

---

## 🧠 What I Learned

- **Data quality matters first** — `TotalCharges` ships as a string in the raw dataset and silently introduces nulls if not caught early
- **Feature engineering > model selection** — `num_services` and `is_longterm` were stronger predictors than raw numerical columns
- **SHAP bridges ML and business** — without explainability, a 88% accurate model is a black box that stakeholders won't trust or act on
- **Class imbalance is real** — the dataset is 74/26 split; SMOTE meaningfully improved recall on the minority churn class
- **Clustering adds business value** — segments turn a binary prediction into a targeting strategy with different actions per group

---

## 📬 Connect & Feedback

If you found this useful, please ⭐ star the repo!

Built as part of an end-to-end ML portfolio project.

---

<p align="center">
  <a href="https://customer-churn-analysis-ksenxnm2hjcrypjdfbeadq.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20App-Click%20Here-ff4b4b?style=for-the-badge" />
  </a>
</p>
