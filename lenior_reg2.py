import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def print_lr2(name):
    print(f'Hello, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # Generate synthetic data
    np.random.seed(42)
    X_full = np.random.randint(100, 500, size=50).reshape(-1, 1)
    noise = np.random.normal(0, 300, size=50)
    Y_full = 20 * X_full.flatten() + noise

    # Track slope and intercept
    slopes = []
    intercepts = []

    # Train on 5 random subsets
    for i in range(5):
        idx = np.random.choice(range(50), size=30, replace=False)
        X = X_full[idx]
        Y = Y_full[idx]

        model = LinearRegression()
        model.fit(X, Y)

        slope = model.coef_[0]
        intercept = model.intercept_

        slopes.append(slope)
        intercepts.append(intercept)

        print(f"📈 Iteration {i + 1}: Slope = {slope:.2f}, Intercept = {intercept:.2f}")

    # Plot slope and intercept evolution
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 6), slopes, marker='o', color='blue')
    plt.title('Slope Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Slope')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 6), intercepts, marker='o', color='green')
    plt.title('Intercept Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Intercept')

    plt.tight_layout()
    plt.show()
