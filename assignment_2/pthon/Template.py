#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on a cloudy day


Name: Asif
Student ID: R00201303
Cohort: SD3A

"""

# Use the below few lines to ignore the warnning messages
"""warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn"""


"""TASK 1"""
import pandas as pd
import matplotlib.pyplot as plt


def task1():
    # Read the csv file
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')
    # Find number of unique locations in 'weather.csv'
    unique_locations = df['Location'].nunique()
    print("The total number of unique locations are", unique_locations)

    # Five locations with the fewest rows/records
    count = df['Location'].value_counts().sort_values()
    top_locations = count.head(5)

    # Visualisation
    plt.bar(top_locations.index, top_locations.values)
    plt.xlabel('Locations')
    plt.ylabel('Records')
    plt.title('Locations with fewest Records')
    plt.show()

    # Calculate then display the result for percentage in each section
    records = len(df)
    percent = (top_locations / records) * 100
    print("Following are the percentage for each section:")
    print(percent)

    """# Cleaning
    # Drop rows with missing values in relevant columns
    relevant_columns = ['Location', 'OtherRelevantColumn1', 'OtherRelevantColumn2']
    data_cleaned = df.dropna(subset=relevant_columns)

    # Drop duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Additional cleaning steps as needed...

    # Display cleaned data summary
    cleaned_records = len(data_cleaned)
    print(f"\nAfter cleaning, the dataset has {cleaned_records} records.")"""

#task1()


"""TASK 2"""

import pandas as pd


def task2():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Check for columns named 'Pressure9am', 'Pressure3pm', and 'RainTomorrow'
    if 'Pressure9am' not in df.columns or 'Pressure3pm' not in df.columns or 'RainTomorrow' not in df.columns:
        print("Error: The 'Weather.CSV' file must have columns named 'Pressure9am', 'Pressure3pm', and 'RainTomorrow'.")
        return

    # New column 'PressureDifference' representing the absolute difference between 9 am and 3 pm pressures
    df['PressureDifference'] = abs(df['Pressure9am'] - df['Pressure3pm'])

    # Store results for each Day value
    results = {}

    # Iterate over the range of D values from 1 to 12
    for Day in range(1, 13):
        # Extract rows with the minimum difference Day
        min_diff = df[df['PressureDifference'] == Day]

        # Count the number of rainy & non-rainy days
        rainy_days = min_diff[min_diff['RainTomorrow'] == 'Yes'].shape[0]
        non_rainy_days = min_diff[min_diff['RainTomorrow'] == 'No'].shape[0]

        # Ratio of rainy days to non-rainy days
        calculate_ratio = rainy_days / non_rainy_days if non_rainy_days != 0 else 0

        # Store results for this Day value
        results[Day] = calculate_ratio

    return results

result = task2()

# Print the results
for Day, ratio in result.items():
    print(f'For Day {Day}: Rainy/Non-Rainy Ratio is  {ratio}')


"""TASK 3"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass

# Suppress other warnings
warnings.warn = warn

def task3():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Check DataFrame has columns including these
    req_columns = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                   'Pressure9am', 'Temp9am', 'Temp3pm', 'RainTomorrow']

    for column in req_columns:
        if column not in df.columns:
            print(f"Error: The 'weather.csv' file must have a column named '{column}'.")
            return

    # Sub-DataFrame with the specified attributes
    sub_df = df[req_columns]

    # Encode the Rain Tomorrow column to numerical values (0 = 'No', 1 = 'Yes')
    label_end = preprocessing.LabelEncoder()
    sub_df['RainTomorrow'] = label_end.fit_transform(sub_df['RainTomorrow'])

    # Separate features & target variable
    x = sub_df.drop('RainTomorrow', axis=1)
    y = sub_df['RainTomorrow']

    # Decision tree classifier
    dtc = DecisionTreeClassifier()

    # Lists for importance scores of each maximum depth
    feature_importance_scr = []

    # Maximum depths ranging 1 to 35
    max_dpt_val = list(range(1, 36))

    for max_dpt in max_dpt_val:
        # Set maximum depth(decision tree)
        dtc.set_params(max_depth=max_dpt)

        # Fit(decision tree classifier)
        dtc.fit(x, y)

        # Get feature importance
        feature_importance = dtc.feature_importances_

        # Append feature importance in list
        feature_importance_scr.append(feature_importance)

    # Convert list into DataFrame
    feature_importance_df = pd.DataFrame(feature_importance_scr, columns=x.columns)

    # Visualization
    plt.figure(figsize=(12, 8))

    for feature in feature_importance_df.columns:
        plt.plot(max_dpt_val, feature_importance_df[feature], label=feature, marker='o', linestyle='-')

    plt.xlabel('Max Depth')
    plt.ylabel('Feature Importance')
    plt.title('Effect of Maximum Depth on Feature Importance')
    plt.legend()
    plt.grid(True)
    plt.show()
    # To improve the readability of the visualization:
    # Added markers to the plot for each data point.
    # Used solid lines for better visibility.
    # Added a grid for better reference.

# Execution
#task3()


"""TASK 4"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):

    pass
import warnings

warnings.warn = warn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def task4():
    # Read the csv file
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Sub-DataFrame with specified attributes
    sub_df_p = df[['WindDir9am', 'WindDir3pm', 'Pressure9am', 'Pressure3pm', 'RainTomorrow']]
    sub_df_w = df[['WindDir9am', 'WindDir3pm', 'RainTomorrow']]

    # Drop NaN values rows
    sub_df_p = sub_df_p.dropna()
    sub_df_w = sub_df_w.dropna()

    # Classification algorithm & calculate accuracy function
    def classification(df, features, target):
        # Separate features & target variables
        x = df[features]
        y = df[target]

        # Using label encoding for categorical features
        for feature in features:
            x[feature] = x[feature].astype('category').cat.codes

        # Split into training & testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Decision Tree Classifier initialising
        dt_classifier = DecisionTreeClassifier(random_state=42)

        # Train Decision Tree Classifier
        dt_classifier.fit(x_train, y_train)

        # Predictions(training & test sets)
        y_train_pred = dt_classifier.predict(x_train)
        y_test_pred = dt_classifier.predict(x_test)

        # Calculate accuracy(training & test sets)
        training_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        return training_accuracy, test_accuracy

    # Classification(Pressure)
    p_features = ['Pressure9am', 'Pressure3pm']
    p_target = 'RainTomorrow'
    p_train_acc, p_test_acc = classification(sub_df_p, p_features, p_target)

    # Classification(Wind)
    w_features = ['WindDir9am', 'WindDir3pm']
    w_target = 'RainTomorrow'
    w_train_acc, w_test_acc = classification(sub_df_w, w_features, w_target)

    # Results
    print("\nThe Results")
    print("\nFor Wind attributes:")
    print(f"Training Accuracy is: {w_train_acc:.2f}")
    print(f"Test Accuracy is: {w_test_acc:.2f}")

    print("\nFor Pressure attributes:")
    print(f"Training Accuracy is: {p_train_acc:.2f}")
    print(f"Test Accuracy is: {p_test_acc:.2f}")

    # Explanation
    print("\nFollowing are few explanations based on the Evaluation:")
    print("1. Two separate sets of attributes are used when running the classification method.")
    print("2. To ascertain which model is superior for forecasting 'RainTomorrow' the accuracy results are compared.")
    print("3. Taking into account elements like overfitting and achieving a balance between test accuracy and training.")


#task4()



"""TASK 5"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def task5():
    # Read csv file
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Sub-DataFrame with specified attributes
    # Avoid the SettingWithCopyWarning ".copy()"
    sub_df = df[['RainTomorrow', 'WindDir9am', 'WindGustDir',
                        'WindDir3pm']].copy()

    # Change the relevant columns to string(.loc)
    sub_df.loc[:, 'WindDir9am'] = sub_df['WindDir9am'].astype(str)
    sub_df.loc[:, 'WindGustDir'] = sub_df['WindGustDir'].astype(str)
    sub_df.loc[:, 'WindDir3pm'] = sub_df['WindDir3pm'].astype(str)

    # Rows with three-letter wind directions
    sub_df = sub_df[~sub_df['WindDir9am'].str.match(r'\b\w{3}\b')]
    sub_df = sub_df[~sub_df['WindGustDir'].str.match(r'\b\w{3}\b')]
    sub_df = sub_df[~sub_df['WindDir3pm'].str.match(r'\b\w{3}\b')]

    # Map target variable to binary values, handling NaN
    sub_df['RainTomorrow'] = sub_df['RainTomorrow'].map({'No': 0, 'Yes': 1}).fillna(0)

    # Modeling: map categorical wind directions to numerical values.
    w_direction = set(sub_df['WindDir9am'].unique()) | set(sub_df['WindGustDir'].unique()) | set(
        sub_df['WindDir3pm'].unique())
    wd_map = {direction: i for i, direction in enumerate(w_direction)}

    sub_df['WindDir9am'] = sub_df['WindDir9am'].map(wd_map)
    sub_df['WindGustDir'] = sub_df['WindGustDir'].map(wd_map)
    sub_df['WindDir3pm'] = sub_df['WindDir3pm'].map(wd_map)

    # Separate features (X) and target variable (y)
    X = sub_df.drop('RainTomorrow', axis=1)
    y = sub_df['RainTomorrow']

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_train_acc = []
    dt_test_acc = []
    kn_train_acc = []
    kn_test_acc = []

    # Depths for Decision Tree Classifier
    for depths in range(1, 11):
        dt_class = DecisionTreeClassifier(max_depth=depths, random_state=42)
        dt_train_score = cross_val_score(dt_class, X_train, y_train, cv=5, scoring='accuracy')
        dt_test_score = cross_val_score(dt_class, X_test, y_test, cv=5, scoring='accuracy')

        dt_train_acc.append(dt_train_score.mean())
        dt_test_acc.append(dt_test_score.mean())

    # Neighbors for K-Neighbors Classifier
    for neighbors in range(1, 11):
        kn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
        kn_train_score = cross_val_score(kn_classifier, X_train, y_train, cv=5, scoring='accuracy')
        kn_test_score = cross_val_score(kn_classifier, X_test, y_test, cv=5, scoring='accuracy')

        kn_train_acc.append(kn_train_score.mean())
        kn_test_acc.append(kn_test_score.mean())

    # Visualisation
    plt.figure(figsize=(10, 8))

    # Decision Tree Classifier
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 11), dt_train_acc, label='Decision Tree: Training Accuracy')
    plt.plot(range(1, 11), dt_test_acc, label='Decision Tree: Test Accuracy')
    plt.title('Decision Tree Classifier - Training & Test Accuracy vs Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()

    # KNeighbors Classifier
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 11), kn_train_acc, label='K-Neighbor: Training Accuracy')
    plt.plot(range(1, 11), kn_test_acc, label='K-Neighbors: Test Accuracy')
    plt.title('K-Neighbor Classifier- Training & Test Accuracy vs Number of Neighbors')
    plt.xlabel('Number of Neighbor')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

    # Explanation
    print("\nFollowing are few explanations based on the Evaluation:")
    print("1. Decision Tree Classifier: Training accuracy is generally increased with depth, but too much depth might result in overfitting.")
    print("2. K-Neighbors Classifier: A smaller number of neighbors(1-3) typically results in higher training accuracy however, overfitting may have an impact on test accuracy.")

    # Observations from the visaulisation
    # Decision Tree Classifier:
    # 1.Test accuracy decreases as the maximum depth of the decision tree increases.
    # 2.Training accuracy also decreases with an increase in the maximum depth, but the decrease is less pronounced.
    # KNeighbors Classifier:
    # 1. Both test and training accuracy start high when the number of neighbors is low.
    # 2. As the number of neighbors increases, accuracy shows a zig-zag pattern.
    # 3. The zig-zag pattern indicates that as the number of neighbors grows, there is fluctuation in accuracy but not a drastic decline.


task5()



"""TASK 6"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def task6():
    # Read the csv file
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Check non-numerical values and drop corresponding rows
    df = df.dropna().select_dtypes(include='number')

    # Standardization
    scaler = StandardScaler()
    df_scaler = scaler.fit_transform(df)

    # Range of clusters to test
    cluster_list = [2, 3, 4, 5, 6, 7, 8]

    # Inertia value for each cluster
    inertia_value = []

    # K-Means clustering
    for n_clusters in cluster_list:
        k_mean = KMeans(n_clusters=n_clusters, random_state=42)
        k_mean.fit(df_scaler)
        inertia_value.append(k_mean.inertia_)

    # Elbow method: optimal number of clusters
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_list, inertia_value, marker='o')
    plt.title('Elbow Method: Optimal Number of Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Elbow point
    opt_cluster = int(input("Elbow Method: Enter the optimal number of clusters: "))

    # K-Means clustering (optimal number of clusters)
    k_mean_opt = KMeans(n_clusters=opt_cluster, random_state=42)
    df['Cluster'] = k_mean_opt.fit_predict(df_scaler)

    # Visualize the clusters with centroids in a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['MinTemp'], df['MaxTemp'], c=df['Cluster'], cmap='viridis', alpha=0.5)
    plt.scatter(k_mean_opt.cluster_centers_[:, 0], k_mean_opt.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering with {opt_cluster} Clusters')
    plt.xlabel('MinTemp')
    plt.ylabel('MaxTemp')
    plt.legend()
    plt.show()

    # Provide an explanation based on the findings
    print(f"\nFollowing are few explanations based on the Evaluation:")
    print(f"1. Optimal number of clusters based on the elbow method is chosen as {opt_cluster}.")
    print("2. Visualisation shows how data points are grouped into clusters based on min temp and max temp.")

    # For ease of visualisation: use scatter plot with centroids marked for each cluster.
    # 1. Centroids offers a visual confirmation of the cluster assignments.
    # 2. Users can see how well the algorithm has grouped data points around these central positions.
    # 3. Easy Comparison: With centroids, users can easily compare the center points of different clusters,
    # making it straightforward to understand the characteristics of each cluster in terms of the features being analyzed.

#task6()


"""TASK 7"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def task7():
    # This task is a weather prediction task that utilizes a machine learning model, specifically a Random Forest Classifier,
    # to predict whether it will rain tomorrow based on various weather attributes mentioned in the 'task7' below.

    # Read the csv file
    df = pd.read_csv(r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv')

    # Check non-numerical values and drop corresponding rows
    df = df.dropna().select_dtypes(include='number')

    # Target variable 'RainTomorrow' based on 'Rainfall'
    df['RainTomorrow'] = (df['Rainfall'] > 0).astype(int)

    # Define features & target variable
    X = df[['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm']]
    y = df['RainTomorrow']

    # Split into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_class = RandomForestClassifier(random_state=42)

    # Train the classifier
    rf_class.fit(X_train, y_train)

    # Prediction(test set)
    y_prediction = rf_class.predict(X_test)

    # Model
    acc = accuracy_score(y_test, y_prediction)
    class_rep = classification_report(y_test, y_prediction)
    conf_mat = confusion_matrix(y_test, y_prediction)

    # Visualization(confusion matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
    plt.xlabel('The Predicted Labels')
    plt.ylabel('The True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Display metrics
    print("\nFrom Weather Prediction Model:")
    print(f"The Accuracy is: {acc*100:.2f}%")
    print("\nThe Classification Report:\n", class_rep)
    print("\nFor Confusion Matrix:\n", conf_mat)


    # Conclusion(confusion matrix)
    tn, fp, fn, tp = conf_mat.ravel()
    print("\nConclusion:")
    print(f"True Negative (No Rain correctly predicted): {tn}")
    print(f"False Positive (Rain incorrectly predicted): {fp}")
    print(f"False Negative (No Rain incorrectly predicted): {fn}")
    print(f"True Positive (Rain correctly predicted): {tp}")

    # Following are few explanations based on the Evaluation:
    # 1. Random Forest Regressor is trained to predict the amount of rainfall based on weather attributes.
    # 2. Mean Squared Error is measure of the model's accuracy. Also higher R-squared score indicates better prediction.
    # 3. The heatmap visually represents the predicted labels against the true labels, providing an insightful view of the model's performance.

task7()
