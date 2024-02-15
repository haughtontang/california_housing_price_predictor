from sklearn.datasets import fetch_california_housing
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def house_price_predictor():
    california_housing = fetch_california_housing(as_frame=True)

    X = california_housing.data
    y = california_housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = (np.mean(y_pred == y_train) * 100)
    test_y_pred = model.predict(x_test)
    test_accuracy = (np.mean(test_y_pred == y_test) * 100)
    print(f"Training accuracy: {accuracy}")
    print(f"Test accuracy: {test_accuracy}")





if __name__ == '__main__':
    house_price_predictor()
