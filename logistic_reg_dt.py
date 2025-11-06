import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree


def print_dt(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Step 1: Create dataset
    data = {
        'StudyHours': [11, 2, 13, 4, 5, 6, 7, 8, 9, 10, 12, 4, 6, 18, 10],
        'Attendance': [70, 65, 70, 75, 80, 85, 50, 95, 100, 100, 55, 70, 85, 95, 100],
        'Pass': [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]
    }

    df_students = generate_student_data()
    print(df_students.head())

    df = pd.DataFrame(data) # static data

    df = df_students
    print("📊 Dataset:\n", df)

    # Step 2: Split features and target
    X = df[['StudyHours', 'Attendance']]
    y = df['Pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train and evaluate across depths
    for depth in range(1, 5):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Compare predictions with actual labels
        incorrect = X_test.copy()
        incorrect['Actual'] = y_test.values
        incorrect['Predicted'] = y_pred_test
        incorrect_misclassified = incorrect[incorrect['Actual'] != incorrect['Predicted']]

        # Print misclassified records
        print(f"❌ Misclassified Records (Depth {depth}):")
        print(incorrect_misclassified)

        acc_test = accuracy_score(y_test, y_pred_test)
        acc_train = accuracy_score(y_train, y_pred_train)

        print(f"\n🌳 Depth {depth}")
        print(f"✅ Train Accuracy: {acc_train:.2f}")
        print(f"🧪 Test Accuracy:  {acc_test:.2f}")
        print("📋 Classification Report (Test):")
        print(classification_report(y_test, y_pred_test))

        # Step 4: Visualize tree structure
        plt.figure(figsize=(6, 4))
        plot_tree(model, feature_names=X.columns, class_names=['Fail', 'Pass'], filled=True)
        plt.title(f'Decision Tree (Depth {depth})')
        plt.tight_layout()
        plt.show()

        # Step 5: Visualize predictions
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test['StudyHours'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['StudyHours'], y_pred_test, color='red', marker='x', label='Predicted')
        plt.title(f'Predictions (Depth {depth})')
        plt.xlabel('Study Hours')
        plt.ylabel('Pass (1=Yes, 0=No)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def generate_student_data(num_records=100, seed=60):
    np.random.seed(seed)

    # Random study hours between 1 and 20
    study_hours = np.random.randint(1, 25, size=num_records)

    # Random attendance between 50% and 100%
    attendance = np.random.randint(40, 101, size=num_records)

    # Simple rule-based label with noise
    # Pass if (study_hours * 2 + attendance) > threshold
    threshold = 100 + np.random.randint(-10, 10)
    score = study_hours * 2 + attendance
    noise = np.random.normal(0, 10, size=num_records)
    pass_label = ((score + noise) > threshold).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'StudyHours': study_hours,
        'Attendance': attendance,
        'Pass': pass_label
    })
    return df