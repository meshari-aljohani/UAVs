#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 05 23:44:41 2023

@author: Meshari
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import csv
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt

for r in range(0,50,3):
    # Load the dataset
    #data = pd.read_excel("dr1.xlsx")
    #train_data = pd.read_csv("sim_data.csv")
    #r = 10
    data = pd.read_csv("F"+str(r)+".csv")
    print("Data:")
    data.head(7)

    # Extract features (X) and target (y) from the dataset
    #X = data.drop(["Mission_ID","Success", "takeoff_Battery_Level","search_Battery_Level", "Target_Detected_Battery_Level", "attack_Battery_Level", "Elimination_Proof_Battery_Level","land_Battery_Level" ,"Target_Detected_Latitude"  ], axis=1)
    X = data.drop(["Mission_ID","Success"], axis=1)

    y = data["Success"]

    print("X head:")
    X.head()


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    #X_test = X[0]
    #y_test = y[1,-1]

    print("X_train:")
    print(X_train)
    print("X_test:")
    print(X_test)

    # Standardize the features (optional, but often improves performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=20, random_state=3, max_depth= 2)

    # Train the classifier using the training data
    classifier.fit(X_train, y_train.ravel())

    # Make predictions on the testing set
    y_pred = classifier.predict(X_test)
    print("y_pred:")
    print(y_pred)

    # Evaluate the classifier's performance
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    import seaborn as sns

    sns.heatmap(confusion_matrix(y_test, y_pred), fmt='1')

    print(classifier.feature_importances_)
    print(max(classifier.feature_importances_))

    #feature_names =  ["Latitude", "Longitude", "Altitude", "temperature","wind_speed","humidity", "Acceleration", "Battery_Level", "Obstacle_Detected"]

    #feature_names = ["takeoff_Latitude", "takeoff_Longitude","takeoff_Altitude", "takeoff_Temperature","takeoff_Wind_Speed","takeoff_Humidity","takeoff_Acceleration","takeoff_Battery_Level","takeoff_Obstacle_Detected","search_Latitude","search_Longitude","search_Altitude","search_Temperature","search_Wind_Speed","search_Humidity","search_Acceleration","search_Battery_Level","search_Obstacle_Detected","Target_Detected_Latitude","Target_Detected_Longitude","Target_Detected_Altitude","Target_Detected_Temperature","Target_Detected_Wind_Speed","Target_Detected_Humidity","Target_Detected_Acceleration","Target_Detected_Battery_Level","Target_Detected_Obstacle_Detected","attack_Latitude","attack_Longitude","attack_Altitude","attack_Temperature","attack_Wind_Speed","attack_Humidity","attack_Acceleration","attack_Battery_Level","attack_Obstacle_Detected","Elimination_Proof_Latitude","Elimination_Proof_Longitude","Elimination_Proof_Altitude","Elimination_Proof_Temperature","Elimination_Proof_Wind_Speed","Elimination_Proof_Humidity","Elimination_Proof_Acceleration","Elimination_Proof_Battery_Level","Elimination_Proof_Obstacle_Detected","land_Latitude","land_Longitude","land_Altitude","land_Temperature","land_Wind_Speed","land_Humidity","land_Acceleration","land_Battery_Level","land_Obstacle_Detected"]

    feature_names = ["takeoff_Latitude", "takeoff_Longitude","takeoff_Altitude", "takeoff_Temperature","takeoff_Wind_Speed","takeoff_Humidity","takeoff_Acceleration","takeoff_Obstacle_Detected","search_Latitude","search_Longitude","search_Altitude","search_Temperature","search_Wind_Speed","search_Humidity","search_Acceleration","search_Obstacle_Detected","Target_Detected_Latitude","Target_Detected_Longitude","Target_Detected_Altitude","Target_Detected_Temperature","Target_Detected_Wind_Speed","Target_Detected_Humidity","Target_Detected_Acceleration","Target_Detected_Obstacle_Detected","attack_Latitude","attack_Longitude","attack_Altitude","attack_Temperature","attack_Wind_Speed","attack_Humidity","attack_Acceleration","attack_Obstacle_Detected","Elimination_Proof_Latitude","Elimination_Proof_Longitude","Elimination_Proof_Altitude","Elimination_Proof_Temperature","Elimination_Proof_Wind_Speed","Elimination_Proof_Humidity","Elimination_Proof_Acceleration","Elimination_Proof_Obstacle_Detected","land_Latitude","land_Longitude","land_Altitude","land_Temperature","land_Wind_Speed","land_Humidity","land_Acceleration","land_Obstacle_Detected"]




    feature_importances = dict(zip(feature_names, classifier.feature_importances_))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_importances:
        print(f"{feature}: {importance}")


    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

    # Assuming you already have the true labels (y_true) and predicted labels (y_pred) from your model

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate recall (sensitivity)
    recall = recall_score(y_test, y_pred)

    # Calculate specificity
    specificity = tn / (tn + fp)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    # Calculate G-means
    g_means = (specificity * recall) ** 0.5

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the metrics
    print(f"Precision: {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1}")
    print(f"G-means: {g_means}")
    print(f"Accuracy: {accuracy}")


    aaa = (tp+tn)/(tp+tn+fp+fn)
    print(aaa)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc


    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("ROC_RF_flaten"+str(r)+".pdf", dpi = 100)

    if r==0:
        # Create a dataframe to store your results
        mission = pd.DataFrame({"F1 Score": f1, "Precision": precision, "Recall": recall, "Specificity": specificity, "Accuracy": accuracy, "DatasetSize":len(data), "TrainSampleSize":len(X_train), "TestSampleSize":len(X_test)},  index=["0"])

        # Save your dataframe
        mission.to_csv("flaten.csv")

    else:
        # Open the CSV file in append mode
        with open("flaten.csv", "a") as f:

            # Write the new result to the file
            writer = csv.writer(f)
            #writer.writerow(["run", "F1 Score", "Precision", "Recall", "Specificity", "Accuracy"])
            writer.writerow([r, str(f1), str(precision), str(recall), str(specificity), str(accuracy), str(len(X)), str(len(X_train)), str(len(X_test))])

        # Close the file
        f.close()