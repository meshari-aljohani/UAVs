


import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve

task_types = ["takeoff", "navigate_to_target", "surveillance", "engagement_decision", "strike_execution", "damage_assessment"]

data = pd.read_csv("AB8_20000.csv")
X = data.drop(columns=['Time_of_Day', 'Overall_Success', 'Mission_ID'] + [f'{task_type}_Success_Ratio' for task_type in task_types]+ [f'{task_type}_Battery_Level' for task_type in task_types])
y = data['Overall_Success']

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models with adjusted hyperparameters
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    'SVM (LibSVM)': SVC(kernel='rbf', C=0.5, probability=True, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Bagging with Decision Trees': BaggingClassifier(n_estimators=100, base_estimator=RandomForestClassifier(max_depth=4), random_state=42)
}


plot_index = 0
letters = 'abcde'
plot_index2 = 0
letters2 = 'abcd'
# Training and evaluating each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")



    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix - {name}:\n", conf_mat)  # Print the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Use the plot_index to get the corresponding letter
    plt.title(f'({letters[plot_index]}): Confusion Matrix - {name}')
    plt.show()
    plot_index += 1


    # ROC Curve
    if name != 'Naive Bayes':  
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'({letters2[plot_index2]}): ROC - {name}')
        plt.legend(loc='lower right')
        plt.show()
        plot_index2 += 1



# Boxplots for Cross-Validation Scores
model_names = []
scores_list = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    model_names.append(name)
    scores_list.append(scores)

plt.figure(figsize=(10, 6))
plt.boxplot(scores_list, labels=model_names, showmeans=True)
plt.title('Cross-Validation Scores for Each Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
