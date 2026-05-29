"""
Project: AI-Based Student Performance Prediction System
Role: Senior Machine Learning Engineer (Internship-Ready Portfolio)
Description: A supervised classification project using Decision Trees and
             Logistic Regression to predict student performance categories (Grades, Pass/Fail, Performance Level).
             This project demonstrates the full ML pipeline: EDA, 
             preprocessing, model comparison, and interactive prediction.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities for Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set plot style for professional looking graphs
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 2. DATASET CREATION (Synthetic Data Generation)
# ---------------------------------------------------------
def generate_student_data(samples=500):
    """Generates a realistic dataset for student performance."""
    np.random.seed(42)

    data = {
        'StudyHours': np.random.randint(1, 21, samples), # Hours per week (1-20)
        'Attendance': np.random.randint(60, 101, samples), # 60-100%
        'AssignmentsCompleted': np.random.randint(0, 11, samples), # 0-10
        'InternalMarks': np.random.randint(30, 101, samples) # 30-100
    }

    df = pd.DataFrame(data)

    # Simulate weighted logic mapped onto a 0-100 scale for a realistic dataset
    score = (df['StudyHours'] / 20 * 15) + \
            (df['Attendance'] * 0.20) + \
            (df['AssignmentsCompleted'] / 10 * 15) + \
            (df['InternalMarks'] * 0.50)

    # Define thresholds for multiple performance categories (Grades) per instructions
    def assign_grade(s):
        if s >= 90:
            return 'Grade A'
        elif 75 <= s < 90:
            return 'Grade B'
        elif 60 <= s < 75:
            return 'Grade C'
        elif 40 <= s < 60:
            return 'Grade D'
        else:
            return 'Grade F' # Represents a failing grade

    df['Grade'] = score.apply(assign_grade)

    # Derive other performance categories from 'Grade' for comprehensive prediction
    df['Pass_Fail'] = df['Grade'].apply(lambda g: 'PASS' if g in ['Grade A', 'Grade B', 'Grade C', 'Grade D'] else 'FAIL')
    df['Performance_Category'] = df['Grade'].apply(lambda g: 'GOOD' if g in ['Grade A', 'Grade B'] else
                                                              'AVERAGE' if g == 'Grade C' else
                                                              'POOR') # Grade D or F

    # Reorder columns for better readability
    df = df[['StudyHours', 'Attendance', 'AssignmentsCompleted', 'InternalMarks', 'Grade', 'Pass_Fail', 'Performance_Category']]

    print(f"\n[Info] Generated {samples} synthetic student records.")
    print(f"       Unique Grades: {df['Grade'].unique()}")
    print(f"       Unique Pass/Fail: {df['Pass_Fail'].unique()}")
    print(f"       Unique Performance Categories: {df['Performance_Category'].unique()}")
    return df

# ---------------------------------------------------------
# 3. USER INPUT & PREDICTION SYSTEM
# ---------------------------------------------------------
def get_user_prediction(dt_model, encoder):
    """Collects validated user input and predicts the outcome."""
    print("\n" + "="*70)
    print("   STEP 7: REAL-TIME STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*70)
    print("\nPlease enter the student details below:")

    def get_validated_input(prompt, min_val, max_val, input_type=float):
        """Helper function to ensure input is within the correct range and type."""
        while True:
            try:
                val = input_type(input(prompt))
                if min_val <= val <= max_val:
                    return val
                else:
                    print(f"   [!] Error: Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("   [!] Error: Invalid input. Please enter numerical values only.")
            except Exception as e:
                print(f"   [!] An unexpected error occurred: {e}")

    # Collect inputs with strict validation loops
    # Study Hours: positive only (let's set a max of 20 for realism)
    hours = get_validated_input("-> Enter Study Hours per week (1-20): ", 1, 20, input_type=int)
    attendance = get_validated_input("-> Enter Attendance Percentage (0-100): ", 0, 100, input_type=int)
    assignments = get_validated_input("-> Enter Number of Assignments Completed (0-10): ", 0, 10, input_type=int)
    marks = get_validated_input("-> Enter Internal Marks (0-100): ", 0, 100, input_type=int)

    # Create a DataFrame with proper feature names to prevent sklearn warnings
    user_data = pd.DataFrame(
        [[hours, attendance, assignments, marks]], 
        columns=['StudyHours', 'Attendance', 'AssignmentsCompleted', 'InternalMarks']
    )

    # Immediate Prediction using the trained Decision Tree model
    pred_encoded = dt_model.predict(user_data)
    predicted_grade = encoder.inverse_transform(pred_encoded)[0]

    # Derive other performance categories from the predicted grade
    predicted_pass_fail = 'PASS' if predicted_grade in ['Grade A', 'Grade B', 'Grade C', 'Grade D'] else 'FAIL'
    predicted_performance_category = 'GOOD' if predicted_grade in ['Grade A', 'Grade B'] else \
                                     'AVERAGE' if predicted_grade == 'Grade C' else \
                                     'POOR' # Grade D or F

    # Professional and clear output for presentations
    print("\n" + "*"*70)
    print(f"Grade Prediction: {predicted_grade}")
    print(f"Result: {predicted_pass_fail}")
    print(f"Performance Level: {predicted_performance_category}")
    print("*"*70 + "\n")

# ---------------------------------------------------------
# 4. VISUALIZATION SUITE
# ---------------------------------------------------------
def show_visualizations(df, X, y_test, dt_model, acc_dt, acc_log, encoder, sorted_grade_names):
    """Generates professional, multi-window visualization dashboards."""
    print("\n[Step 9] Generating Presentation-Quality Visualizations...")
    
    # Ensure consistent test set for confusion matrix
    _, X_test_local, _, _ = train_test_split(X, df['Grade_Encoded'], test_size=0.2, random_state=42)
    y_pred_dt = dt_model.predict(X_test_local)
    conf_matrix = confusion_matrix(y_test, y_pred_dt)

    # ---------------------------------------------------------
    # FIGURE 1: AI MODEL PERFORMANCE DASHBOARD
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(1, 3, figsize=(20, 7))
    fig1.suptitle("AI Model Performance Dashboard", fontsize=24, fontweight='bold', y=0.98)

    # 1. Confusion Matrix Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1[0],
                xticklabels=sorted_grade_names, yticklabels=sorted_grade_names, annot_kws={"size": 14})
    ax1[0].set_title('Confusion Matrix: Actual vs Predicted Grades', fontsize=16, pad=20, fontweight='bold')
    ax1[0].set_xlabel('Predicted Grade', fontsize=13)
    ax1[0].set_ylabel('Actual Grade', fontsize=13)

    # 2. Feature Importance Graph
    importances = pd.Series(dt_model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh', color='#2ca02c', edgecolor='black', ax=ax1[1])
    ax1[1].set_title('Feature Importance Analysis', fontsize=16, pad=20, fontweight='bold')
    ax1[1].set_xlabel('Importance Score', fontsize=13)
    ax1[1].grid(axis='x', linestyle='--', alpha=0.7)

    # 3. Model Accuracy Comparison
    algorithms = ['Decision Tree', 'Logistic Regression']
    accuracies = [acc_dt * 100, acc_log * 100]
    bars = ax1[2].bar(algorithms, accuracies, color=['#1f77b4', '#ff7f0e'], edgecolor='black', alpha=0.8)
    ax1[2].set_ylabel('Accuracy (%)', fontsize=13)
    ax1[2].set_title('Model Accuracy Comparison', fontsize=16, pad=20, fontweight='bold')
    ax1[2].set_ylim(0, 110)
    
    # Percentage labels
    for i, val in enumerate(accuracies):
        ax1[2].text(i, val + 2, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # ---------------------------------------------------------
    # FIGURE 2: STUDENT DATA ANALYSIS & INSIGHTS
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 3, figsize=(20, 7))
    fig2.suptitle("Student Data Analysis & Insights", fontsize=24, fontweight='bold', y=0.98)

    # 4. Student Grade Distribution
    # Ensure grades are plotted in a logical order (e.g., F, D, C, B, A)
    sns.countplot(x='Grade', data=df, hue='Grade', palette='viridis', edgecolor='black', ax=ax2[0], legend=False, order=sorted_grade_names)
    ax2[0].set_title('Distribution of Student Grades', fontsize=15, pad=20, fontweight='bold')
    ax2[0].set_xlabel('Academic Grade', fontsize=13)
    ax2[0].set_ylabel('Number of Students', fontsize=13)

    # 5. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=1, ax=ax2[1], cbar_kws={'shrink': .8})
    ax2[1].set_title('Feature Correlation Analysis', fontsize=16, pad=20, fontweight='bold')

    # 6. Scatter Plot (Study Hours vs Internal Marks, colored by Grade)
    sns.scatterplot(x='StudyHours', y='InternalMarks', hue='Grade', data=df,
                    palette='viridis', s=120, alpha=0.6, ax=ax2[2], hue_order=sorted_grade_names)
    ax2[2].set_title('Study Hours vs Internal Marks Relationship by Grade', fontsize=15, pad=20, fontweight='bold')
    ax2[2].set_xlabel('Study Hours per Week', fontsize=13)
    ax2[2].set_ylabel('Internal Marks', fontsize=13)
    ax2[2].legend(title='Final Grade', loc='best', frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # ---------------------------------------------------------
    # FIGURE 3: DECISION TREE LEARNING STRUCTURE
    # ---------------------------------------------------------
    plt.figure(figsize=(22, 12))
    plt.suptitle("Decision Tree Learning Structure", fontsize=26, fontweight='bold', y=0.98)
    plot_tree(dt_model, feature_names=list(X.columns), class_names=sorted_grade_names,
              filled=True, rounded=True, fontsize=13, max_depth=3, precision=2)

    plt.title("\nHow the AI model predicts student grades", fontsize=18, fontstyle='italic', pad=30)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("\n[Success] Visualization windows opened. Close them to finish the execution.")
    plt.show()

# ---------------------------------------------------------
# 5. MAIN EXECUTION FLOW
# ---------------------------------------------------------
def main():
    # 1. Generate and Load Dataset
    df = generate_student_data(samples=500)
    
    # 2. Display Dataset Preview and Basic Statistics
    print("\n" + "="*60)
    print("   STEP 2: DATASET PREVIEW & STATISTICS")
    print("="*60)
    print("\n[First 5 Rows of Dataset]:")
    print(df.head())
    
    print("\n[Dataset Information]:")
    df.info()
    
    print("\n[Statistical Description]:")
    print(df.describe())
    
    print("\n[Checking for Missing Values]:")
    print(df.isnull().sum())

    # 3. Preprocess Data
    encoder = LabelEncoder()
    df['Grade_Encoded'] = encoder.fit_transform(df['Grade'])
    sorted_grade_names = list(encoder.classes_)
    
    # Feature and Label Separation
    X = df[['StudyHours', 'Attendance', 'AssignmentsCompleted', 'InternalMarks']]
    y = df['Grade_Encoded']
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for Logistic Regression to ensure convergence and accuracy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Machine Learning Models
    print("\n[Step 4] Training Decision Tree & Logistic Regression models...")
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate Models
    y_pred_dt = dt_model.predict(X_test)
    y_pred_log = log_model.predict(X_test_scaled)
    
    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_log = accuracy_score(y_test, y_pred_log)
    
    print("\n[Step 5] Model Evaluation Complete.")
    print(f"   -> Decision Tree Accuracy: {acc_dt*100:.2f}%")
    print(f"   -> Logistic Regression Accuracy: {acc_log*100:.2f}%")

    # Optional: Cross-validation for more robust evaluation
    dt_cv_scores = cross_val_score(dt_model, X, y, cv=5)
    log_cv_scores = cross_val_score(log_model, X_scaled, y, cv=5)
    print(f"   -> Decision Tree Cross-Validation Accuracy (5-fold): {dt_cv_scores.mean()*100:.2f}%")
    print(f"   -> Logistic Regression Cross-Validation Accuracy (5-fold): {log_cv_scores.mean()*100:.2f}%")

    # 6, 7, 8. User Interaction & Prediction Result
    get_user_prediction(dt_model, encoder)
    
    # 9. Final Step: Visualizations
    input("\n-> Press Enter to view all visualizations and final report...")
    
    # Print Detailed Classification Report before showing graphs
    print("\n" + "="*70)
    print("   DETAILED CLASSIFICATION REPORT (Decision Tree Model)")
    print("="*70)
    print(classification_report(y_test, y_pred_dt, target_names=sorted_grade_names))

    show_visualizations(df, X, y_test, dt_model, acc_dt, acc_log, encoder, sorted_grade_names)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Info] System execution interrupted by user. Exiting gracefully...")
        sys.exit(0)
