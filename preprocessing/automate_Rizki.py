import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing():
    df = pd.read_csv('../autisme_raw.csv')
    df = df.dropna()
    df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    df.to_csv('autisme_processed.csv', index=False)
    return df

if __name__ == "__main__":
    run_preprocessing()