if __name__ == "__main__":
    # Define dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    X = iris.data
    y = iris.target

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7)

    # Define model
    from KNN_Classifier import KNN_Classifier
    myClassifier = KNN_Classifier()

    # Training
    myClassifier.fit(X_train, y_train)

    # Measure accuracy
    predictions = myClassifier.predict(X_valid)
    from sklearn.metrics import accuracy_score # % of correct predictions
    print(f"Accuracy Score: {accuracy_score(y_valid, predictions):.5f}\n")