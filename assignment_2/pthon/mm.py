import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_confusion_matrix(conf_matrix):
    """
    Visualizes the confusion matrix using a heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', linewidths=.5, square=True, cbar=True,
                xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'], linecolor='white', linewidth=1)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def display_model_evaluation(accuracy, classification_rep, conf_matrix):
    """
    Displays the model evaluation metrics and explanation.
    """
    print("\nWeather Prediction Model Evaluation:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Provide a clear summary based on the confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    print("\nSummary:")
    print(f"True Negative (No Rain correctly predicted): {tn}")
    print(f"False Positive (Rain incorrectly predicted): {fp}")
    print(f"False Negative (No Rain incorrectly predicted): {fn}")
    print(f"True Positive (Rain correctly predicted): {tp}")

def weather_prediction_task(file_path):
    # Read the csv file
    df = pd.read_csv(file_path)

    # Drop rows with missing values and select numerical columns
    df = df.dropna().select_dtypes(include='number')

    # Create a binary target variable 'RainTomorrow' based on 'Rainfall'
    df['RainTomorrow'] = (df['Rainfall'] > 0).astype(int)

    # Define features (X) and target variable (y)
    X = df[['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm']]
    y = df['RainTomorrow']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Train the classifier
    rf_clf.fit(X_train, y_train)

    # Predictions on the test set
    y_prediction = rf_clf.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_prediction)
    classification_rep = classification_report(y_test, y_prediction)
    conf_matrix = confusion_matrix(y_test, y_prediction, normalize='all')

    # Visualize confusion matrix
    visualize_confusion_matrix(conf_matrix)

    # Display metrics and explanation
    display_model_evaluation(accuracy, classification_rep, conf_matrix)

# Example usage
file_path = r'C:\Users\asifl\OneDrive\Desktop\pthon\weather.csv'
weather_prediction_task(file_path)
