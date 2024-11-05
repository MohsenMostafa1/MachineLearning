import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_tabular_model(data_path):
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Encode categorical columns if any
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions and calculate MAE
    y_pred = model.predict(X_val)
    error = mean_absolute_error(y_val, y_pred)

    print("MAE:", error)
    return model

if __name__ == "__main__":
    train_tabular_model('/home/mohsn/ml_optimization_multimodal/data/candidates_data.csv')
