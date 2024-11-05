# scripts/eda.py
import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['description'])  # Drop description for Step 1
    # Additional cleaning if necessary
    return df

if __name__ == "__main__":
    data = load_and_clean_data('/home/mohsn/ml_optimization_multimodal/data/candidates_data.csv')
    print(data.describe())
