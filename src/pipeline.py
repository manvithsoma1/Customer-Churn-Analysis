import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Telco Customer Churn dataset.
    1. Fix TotalCharges dtype
    2. Encode Churn
    3. Handle nulls
    """
    data = df.copy()
    
    # 1. Fix TotalCharges dtype (replace blank spaces with NaN, then convert to float)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', np.nan))
    
    # 2. Handle nulls (fill NaN in TotalCharges with 0 or drop, but 0 is logical for 0 tenure)
    # We will just fill them with 0 since there are only ~11 missing where tenure is 0.
    data['TotalCharges'] = data['TotalCharges'].fillna(0)
    
    # Drop customerID as it's not a feature
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
        
    # 3. Encode Churn (Yes=1, No=0)
    if 'Churn' in data.columns and data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        
    # Binary encode other Yes/No columns for simplicity
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns and data[col].dtype == 'object':
            if col == 'gender':
                data[col] = data[col].map({'Female': 1, 'Male': 0})
            else:
                data[col] = data[col].map({'Yes': 1, 'No': 0})
                
    return data

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features:
    - tenure_group: binting tenure into groups
    - num_services: count of services the user has
    - is_longterm: binary flag for tenure > 24 months
    - has_support: binary flag indicating if they have any support services
    - charges_per_month: ratio of TotalCharges to tenure
    """
    data = df.copy()
    
    # 1. tenure_group
    # Group into buckets: 0-12, 13-24, 25-48, 49-60, >60
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
    # Include lowest=True so 0 is included if any
    data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # 2. num_services
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Have to be careful: Some are binary (Yes/No), some have "No internet service"
    count = np.zeros(len(data))
    for s in services:
        if s in data.columns:
            # Count standard 'Yes' or if PhoneService is 1 (encoded)
            count += ((data[s] == 'Yes') | (data[s] == 1)).astype(int)
    data['num_services'] = count

    # 3. is_longterm
    data['is_longterm'] = (data['tenure'] > 24).astype(int)
    
    # 4. has_support
    support_cols = ['OnlineSecurity', 'TechSupport']
    supt_count = np.zeros(len(data))
    for c in support_cols:
        if c in data.columns:
            supt_count += ((data[c] == 'Yes') | (data[c] == 1)).astype(int)
    data['has_support'] = (supt_count > 0).astype(int)
    
    # 5. charges_per_month interaction
    safe_tenure = data['tenure'].replace(0, 1)
    data['charges_per_month'] = data['TotalCharges'] / safe_tenure
    
    # 6. Additional Advanced interactions to boost model performance
    # Monthly Charges Bins
    charge_bins = [0, 30, 60, 90, np.inf]
    charge_labels = ['Low', 'Medium', 'High', 'Very High']
    data['monthly_charges_bin'] = pd.cut(data['MonthlyCharges'], bins=charge_bins, labels=charge_labels, include_lowest=True)
    
    # Contract * Tenure interaction (Numeric)
    if 'Contract' in data.columns:
        # Ordinal encode contract
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        data['contract_numeric'] = data['Contract'].map(contract_map)
        data['tenure_contract_interaction'] = data['tenure'] * data['contract_numeric']

    return data
