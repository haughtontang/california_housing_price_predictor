from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def house_price_predictor():
    california_housing = fetch_california_housing(as_frame=True)

    x = california_housing.data
    y = california_housing.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    # Make predictions
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    # Evaluate the model
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Training set evaluation:")
    print(f"Mean Absolute Error: {train_mae}")
    print(f"Mean Squared Error: {train_mse}")
    print(f"R-squared: {train_r2}")

    print("\nTest set evaluation:")
    print(f"Mean Absolute Error: {test_mae}")
    print(f"Mean Squared Error: {test_mse}")
    print(f"R-squared: {test_r2}")


if __name__ == '__main__':
    house_price_predictor()
