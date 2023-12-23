



def task1():
    import pandas as pd
    import matplotlib.pyplot as plt

    def analyze_weather_data(file_path):
        # Read the weather data from the CSV file
        weather_data = pd.read_csv(file_path)

        # Task 1: Find the number of unique locations present in the dataset
        unique_locations = weather_data['Location'].nunique()
        print("Number of unique locations:", unique_locations)

        # Task 2: Display the five locations with the fewest records or rows
        location_counts = weather_data['Location'].value_counts().sort_values()
        top_5_locations = location_counts.head(5)

        # Visualize the five locations with the fewest records
        plt.bar(top_5_locations.index, top_5_locations.values)
        plt.xlabel('Location')
        plt.ylabel('Number of Records')
        plt.title('Locations with the Fewest Records')
        plt.show()

        # Calculate and display the percentage for each section
        total_records = len(weather_data)
        percentages = (top_5_locations / total_records) * 100
        print("Percentage for each section:")
        print(percentages)

        # Data Cleaning
        # Drop rows with missing values in relevant columns
        relevant_columns = ['Location', 'OtherRelevantColumn1', 'OtherRelevantColumn2']
        weather_data_cleaned = weather_data.dropna(subset=relevant_columns)

        # Drop duplicate rows
        weather_data_cleaned = weather_data_cleaned.drop_duplicates()

        # Additional cleaning steps as needed...

        # Display cleaned data summary
        cleaned_records = len(weather_data_cleaned)
        print(f"\nAfter cleaning, the dataset has {cleaned_records} records.")

        # Perform further analysis on the cleaned data
        # ...

    # Example usage
    file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
    analyze_weather_data(file_path)


task1()

def task2():
    import pandas as pd

    def analyze_pressure_effect(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure that the DataFrame has columns named 'Pressure9am', 'Pressure3pm', and 'RainTomorrow'
        if 'Pressure9am' not in df.columns or 'Pressure3pm' not in df.columns or 'RainTomorrow' not in df.columns:
            print("Error: The CSV file must have columns named 'Pressure9am', 'Pressure3pm', and 'RainTomorrow'.")
            return

        # Create a new column 'PressureDifference' representing the absolute difference between 9 am and 3 pm pressures
        df['PressureDifference'] = abs(df['Pressure9am'] - df['Pressure3pm'])

        # Initialize a dictionary to store results for each D value
        results = {}

        # Iterate over the range of D values from 1 to 12
        for D in range(1, 13):
            # Extract rows with the minimum difference D
            min_diff_rows = df[df['PressureDifference'] == D]

            # Count the number of rainy and non-rainy days
            rainy_days = min_diff_rows[min_diff_rows['RainTomorrow'] == 'Yes'].shape[0]
            non_rainy_days = min_diff_rows[min_diff_rows['RainTomorrow'] == 'No'].shape[0]

            # Calculate the ratio of rainy days to non-rainy days
            ratio = rainy_days / non_rainy_days if non_rainy_days != 0 else 0

            # Store the result for this D value
            results[D] = ratio

        return results

    # Example usage
    file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
    result_dict = analyze_pressure_effect(file_path)

    # Print the results
    for D, ratio in result_dict.items():
        print(f'D={D}: Rainy/Non-Rainy Ratio = {ratio}')
#task2()
def task3():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    def visualize_feature_importance(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure that the DataFrame has columns including those specified
        required_columns = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                            'Pressure9am', 'Temp9am', 'Temp3pm', 'RainTomorrow']

        for col in required_columns:
            if col not in df.columns:
                print(f"Error: The CSV file must have a column named '{col}'.")
                return

        # Create a sub-DataFrame with the specified attributes
        sub_df = df[required_columns]

        # Encode the 'RainTomorrow' column to numerical values (0 for 'No', 1 for 'Yes')
        le = preprocessing.LabelEncoder()
        sub_df['RainTomorrow'] = le.fit_transform(sub_df['RainTomorrow'])

        # Separate the features and the target variable
        X = sub_df.drop('RainTomorrow', axis=1)
        y = sub_df['RainTomorrow']

        # Initialize a decision tree classifier
        dt_classifier = DecisionTreeClassifier()

        # Initialize lists to store feature importance scores for each maximum depth
        feature_importance_scores = []

        # Experiment with different maximum depths ranging from 1 to 35
        max_depth_values = list(range(1, 36))

        for max_depth in max_depth_values:
            # Set the maximum depth for the decision tree
            dt_classifier.set_params(max_depth=max_depth)

            # Fit the decision tree classifier
            dt_classifier.fit(X, y)

            # Get feature importances
            feature_importances = dt_classifier.feature_importances_

            # Append the feature importances to the list
            feature_importance_scores.append(feature_importances)

        # Convert the list of feature importance scores into a DataFrame
        feature_importance_df = pd.DataFrame(feature_importance_scores, columns=X.columns)

        # Plot the feature importances for each maximum depth
        plt.figure(figsize=(12, 8))
        for feature in feature_importance_df.columns:
            plt.plot(max_depth_values, feature_importance_df[feature], label=feature)

        plt.xlabel('Maximum Depth')
        plt.ylabel('Feature Importance')
        plt.title('Impact of Maximum Depth on Feature Importance')
        plt.legend()
        plt.show()

    # Example usage
    file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
    visualize_feature_importance(file_path)


#task3()
def task4():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    def analyze_weather_data(file_path):
        # Read the weather data from the CSV file
        data = pd.read_csv(file_path)

        # Create the first sub-dataset with the specified attributes
        sub_dataset1 = data[['WindDir9am', 'WindDir3pm', 'Pressure9am', 'Pressure3pm', 'RainTomorrow']]

        # Split the data into training and testing sets for the first model
        X1 = sub_dataset1.drop('RainTomorrow', axis=1)
        y1 = sub_dataset1['RainTomorrow']
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)

        # Train the classification model and make predictions for the first model
        model1 = DecisionTreeClassifier()
        model1.fit(X1_train, y1_train)
        y1_train_pred = model1.predict(X1_train)
        y1_test_pred = model1.predict(X1_test)

        # Calculate the accuracy for the test and training datasets for the first model
        accuracy_train1 = accuracy_score(y1_train, y1_train_pred)
        accuracy_test1 = accuracy_score(y1_test, y1_test_pred)

        # Create the second sub-dataset with the specified attributes
        sub_dataset2 = data[['WindDir3pm', 'WindDir9am', 'Pressure9am', 'Pressure3pm', 'RainTomorrow']]

        # Split the data into training and testing sets for the second model
        X2 = sub_dataset2.drop('RainTomorrow', axis=1)
        y2 = sub_dataset2['RainTomorrow']
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.33, random_state=42)

        # Train the classification model and make predictions for the second model
        model2 = DecisionTreeClassifier()
        model2.fit(X2_train, y2_train)
        y2_train_pred = model2.predict(X2_train)
        y2_test_pred = model2.predict(X2_test)

        # Calculate the accuracy for the test and training datasets for the second model
        accuracy_train2 = accuracy_score(y2_train, y2_train_pred)
        accuracy_test2 = accuracy_score(y2_test, y2_test_pred)

        return accuracy_train1, accuracy_test1, accuracy_train2, accuracy_test2

    # Example usage
    file_path = r"C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv"
    accuracy_train1, accuracy_test1, accuracy_train2, accuracy_test2 = analyze_weather_data(file_path)
    print("Model 1:")
    print("Training accuracy:", accuracy_train1)
    print("Test accuracy:", accuracy_test1)
    print()
    print("Model 2:")
    print("Training accuracy:", accuracy_train2)
    print("Test accuracy:", accuracy_test2)

#task4()

def task5():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    def analyze_weather_data(file_path):
        # Load the data
        data = pd.read_csv(file_path)

        # Create a sub-DataFrame containing required attributes
        sub_data = data[['RainTomorrow', 'WindDir9am', 'WindGustDir', 'WindDir3pm                     ~sub_data['
                         WindGustDir(r'\b\w{3}\b') &
                         ~sub_data['WindDirains(r'\b\w
        {3}\b')]

        # Convert wind direction attributes to numerical categories
        directions = np.unique(
            sub_data[['WindDir9am', 'WindGustDir', 'WindDir {direction: i for i, direction in enumerate(directions)}
                      sub_data[['WindDir9am', 'WindGustDir', 'WindDir3', 'WindDir3t the data into features and target
                                X = sub_data.drop('RainTomorrow', axis=1)
        y = sub_data

        # Initialize lists to store accuracy values
        dt_train_acc =
        dt_test_acc =
        knn_train_acc =
        knn_test_acc =

        # Train and evaluate Decision Tree Classifier
        for depth in range(1, 11):
            dt = DecisionTreeClassifier(max_depth=depth)
        dt_scores = cross_val_score(dt, X, y, cv=5)
        dt_train_acc.append(np.mean(dt_scores))
        dt_test_acc.append(np.mean(1 - dt_scores))

        # Train and evaluate K Neighbors Classifier
        for neighbors in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn_scores = cross_val_score(knn, X, y, cv=5)
        knn_train_acc.append(np.mean(knn_scores))
        knn_test_acc.append(np.mean(1 - knn_scores))

        # Generate plots
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(range(1, 11), dt_train_acc, label='Training')
        plt.plot(range(1, 11), dt_test_acc, label='Test')
        plt.xlabel('Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree Classifier')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(range(1, 11), knn_train_acc, label='Training')
        plt.plot(range(1, 11), knn_test_acc, label='Test')
        plt.xlabel('Neighbors')
        plt.ylabel('Accuracy')
        plt.title('K Neighbors Classifier')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Find optimal values for depth and number of neighbors
        dt_optimal_depth = dt_train_acc.index(max(dt_train_acc)) + 1
        knn_optimal_neighbors = knn_train_acc.index(max(knn_train_acc)) + 1

        print('Optimal Depth for Decision Tree Classifier:', dt_optimal_depth)
        print('Optimal Neighbors for K Neighbors Classifier:', knn_optimal_neighbors)

        # Provide the file path to the analyze_weather_data function
        file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
        analyze_weather_data(file_path)

#task5()
def task6():
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

    def kmeans_clustering(file_path):
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check for non-numerical values and drop corresponding rows
        df = df.dropna().select_dtypes(include='number')

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Define the range of clusters to test
        n_clusters_list = [2, 3, 4, 5, 6, 7, 8]

        # Store inertia values for each number of clusters
        inertias = []

        # Apply K-Means clustering for different numbers of clusters
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(df_scaled)
            inertias.append(kmeans.inertia_)

        # Determine the optimal number of clusters using the elbow method
        plt.figure(figsize=(8, 4))
        plt.plot(n_clusters_list, inertias, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()

        # Choose the optimal number of clusters (elbow point)
        optimal_clusters = int(input("Enter the optimal number of clusters based on the elbow method: "))

        # Apply K-Means clustering with the optimal number of clusters
        kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
        df['Cluster'] = kmeans_optimal.fit_predict(df_scaled)

        # Visualize the clusters in a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['MinTemp'], df['MaxTemp'], c=df['Cluster'], cmap='viridis', alpha=0.5)
        plt.title(f'K-Means Clustering with {optimal_clusters} Clusters')
        plt.xlabel('MinTemp')
        plt.ylabel('MaxTemp')
        plt.show()

        # Provide an explanation based on the findings
        print(f"\nExplanation based on the findings:")
        print(f"The optimal number of clusters based on the elbow method is chosen as {optimal_clusters}.")
        print("The scatter plot shows how the data points are grouped into clusters based on MinTemp and MaxTemp.")

    # Example usage
    file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
    kmeans_clustering(file_path)


#task6()
def task7():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    def rainfall_prediction(file_path):
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check for non-numerical values and drop corresponding rows
        df = df.dropna().select_dtypes(include='number')

        # Define features (X) and target variable (y)
        X = df.drop('Rainfall', axis=1)
        y = df['Rainfall']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the Random Forest Regressor
        rf_regressor = RandomForestRegressor(random_state=42)

        # Train the regressor
        rf_regressor.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_regressor.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display evaluation metrics
        print("Mean Squared Error:", mse)
        print("R-squared Score:", r2)

        # Visualize the actual vs predicted rainfall
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.title('Actual vs Predicted Rainfall')
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
        plt.show()

        # Provide an explanation based on the findings
        print("\nExplanation based on the findings:")
        print("The Random Forest Regressor is trained to predict the amount of rainfall based on weather attributes.")
        print(
            f"The Mean Squared Error (MSE) is a measure of the model's accuracy, and a higher R-squared score indicates better prediction.")
        print("The scatter plot visually compares the actual vs predicted rainfall values.")

    # Example usage
    file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
    rainfall_prediction(file_path)

#task7()