
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def print_lr(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Step 1: Generate synthetic data
    np.random.seed(18)
    num_samples = 25
    area_sqft = np.random.randint(100, 500, size=num_samples)
    base_price_per_sqft = 20
    price_noise = np.random.normal(loc=0, scale=300, size=num_samples)
    cost_inr = base_price_per_sqft * area_sqft + price_noise

    # Step 2: Create DataFrame
    df = pd.DataFrame({
        'Area_sqft': area_sqft,
        'Noise': price_noise.astype(int),
        'Cost_INR': cost_inr.astype(int)
    })
    print("📊 Sample Data:\n", df)

    # Step 3: Fit Linear Regression Model
    X = area_sqft.reshape(-1, 1)  # Reshape for sklearn
    Y = cost_inr
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    # Step 4: Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Actual Cost')
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title('Linear Regression: House Cost vs Area')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Cost (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print model parameters
    print(f"📈 Slope (Coefficient): {model.coef_[0]:.2f}")
    print(f"📍 Intercept: {model.intercept_:.2f}")

    # Evaluate model
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, Y_pred)

    # Print results
    print(f"📊 Model Performance Metrics:")
    print(f"MAE  = {mae:.2f}")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R²   = {r2:.2f}")



