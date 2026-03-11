# Business Insights & Recommendations: Customer Churn Prediction

This document summarizes the findings from our machine learning analysis on customer churn, focusing on actionable business insights and revenue impact.

## 1. Key Drivers of Customer Churn
Based on our SHAP value analysis from the high-accuracy Gradient Boosting model, the top drivers for customer churn are:
1. **Tenure & Contract Type**: Customers with Month-to-month contracts and less than a year of tenure are significantly more likely to churn. Long-term contracts lock in loyalty and reduce churn dramatically.
2. **Monthly Charges & Fiber Optic**: Higher monthly charges, particularly for Fiber Optic internet customers, are strongly correlated with churn. This indicates high price sensitivity or dissatisfaction with the value proposition of the premium internet tiers.
3. **Lack of Support Services**: Customers without Online Security, Tech Support, or Device Protection are leaving at much higher rates. These services appear to act as 'sticky' features that increase switching costs and improve customer satisfaction.

## 2. Customer Segmentation Analysis
Using K-Means Clustering (K=4), we separated our customer base into four distinct behavioral segments:
- **Segment A (High-Value, High-Risk)**: These are mostly Month-to-Month customers paying high monthly fees for Fiber Optic. They have the highest churn rate among all segments.
- **Segment B (Loyal Long-Term)**: Customers with 2+ year contracts and multiple support services. Their churn rate is near zero.
- **Segment C (Budget-Conscious)**: Lower monthly charges, usually DSL or No Internet. They have moderate churn but are highly price-sensitive to sudden price hikes.
- **Segment D (New Joiners)**: High churn risk purely due to lack of tenure. They are testing the service and require immediate onboarding support.

## 3. Financial/Revenue Impact
Our dataset reveals that churned customers represent a massive risk to annual recurring revenue. 
- **Current Annual Revenue at Risk**: *(See final notebook output for exact calculation)*
- **Estimated Savings**: By implementing preemptive retention campaigns targeting the top 10% highest-risk customers identified by our model, we estimate a **20% improvement in retention**, saving a substantial portion of the annualized revenue that would otherwise be lost.

## 4. Strategic Recommendations
To mitigate revenue loss, the following strategies are proposed:

1. **Incentivize Long-Term Contracts**: Offer a 1-month or 2-month discount when switching from a Month-to-Month to a 1-year contract. The short-term revenue hit is far outweighed by the increased lifetime value (LTV).
2. **Bundle "Sticky" Features**: Include basic Online Security or Tech Support for free or at a steep discount during the first 6 months of a Fiber Optic subscription. This will artificially create stickiness and reduce the initial churn spike.
3. **Targeted Price Reductions**: Use the ML model's probability scores to proactively contact high-risk, high-value customers (Segment A) with personalized discounts *before* they initiate the cancellation process.

---
*Generated as part of the production-ready Customer Churn Data Science portfolio.*
