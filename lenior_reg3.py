import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def print_lr3(name, num_iterations=5):
    print(f'Hello, {name} 👋')

    # Step 1: Generate synthetic data
    num_records = 10
    np.random.seed(42)
    X_full = np.random.randint(100, 500, size=num_records).reshape(-1, 1)
    noise = np.random.normal(0, 300, size=num_records)
    Y_full = 20 * X_full.flatten() + noise

    # Step 2: Track metrics
    slopes = []
    intercepts = []

    # Step 3: Iterative training and visualization
    for i in range(num_iterations):
        idx = np.random.choice(num_records, size=num_records, replace=False)  # sample 30 from 50        X_sample = X_full[idx]
        X_sample = X_full[idx]
        Y_sample = Y_full[idx]

        model = LinearRegression()
        model.fit(X_sample, Y_sample)

        slope = model.coef_[0]
        intercept = model.intercept_
        Y_pred = model.predict(X_full)

        mse = mean_squared_error(Y_full, Y_pred)
        r2 = r2_score(Y_full, Y_pred)

        slopes.append(slope)
        intercepts.append(intercept)

        # Console output
        print(f"\n📈 Iteration {i + 1}")
        print(f"Slope      = {slope:.2f}")
        print(f"Intercept  = {intercept:.2f}")
        print(f"MSE        = {mse:.2f}")
        print(f"R² Score   = {r2:.4f}")
        print("-" * 30)

        # Graph output
        plt.figure(figsize=(8, 6))
        plt.scatter(X_full, Y_full, color='blue', label='Actual Data')
        plt.plot(X_full, Y_pred, color='red', linewidth=2, label='Regression Line')
        plt.title(f'Iteration {i + 1}: Linear Regression\nSlope={slope:.2f}, Intercept={intercept:.2f}, MSE={mse:.2f}, R²={r2:.4f}')
        plt.xlabel('Area (sqft)')
        plt.ylabel('Cost (INR)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.waitforbuttonpress()  # Wait for key press
        plt.close()  # Close plot before next iteration

    # Step 4: Evolution of slope and intercept
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_iterations + 1), slopes, marker='o', color='blue')
    plt.title('Slope Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Slope')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_iterations + 1), intercepts, marker='o', color='green')
    plt.title('Intercept Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Intercept')

    plt.tight_layout()
    plt.show()
